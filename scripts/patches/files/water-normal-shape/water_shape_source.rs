//! Wave shape from the water normal map (local patch set, aurora-render
//! water_shape.rs): the height map from the region's water normal map.
//! While the shape takes part (a Ratio above 0 with the geometric waves),
//! the frame asks the texture streamer to keep that texture's decoded
//! texels on the CPU, as it keeps a sculpt map's (`want_sculpt`: the full
//! resolution, no block compression for that one texture), and when they
//! arrive or change, a copy goes to the thread pool, which integrates the
//! slopes into the height map. The new map replaces the old one when it is
//! done; the frame never waits for it and the GPU is never read back.
//! While an environment change cross-fades two normal maps (patch 998,
//! world/eep_fade.rs; never a day cycle's keyframes) the second one's
//! height map is made the same way, in a slot of its own, and becomes the
//! first one's when the fade ends: nothing is made again. Nothing is sent
//! and nothing is written.
// aurora-local-patch: water-normal-shape-source-module

use super::Scene;
use super::jobs::SculptMap;
use super::textures::TextureStreamer;
use aurora_render::WaterEffects;
use aurora_render::water_effects::shape::{self, HeightMap, Shape};
use std::sync::{Arc, Mutex, mpsc};
use uuid::Uuid;

/// One height map and what it is made from.
#[derive(Default)]
struct Slot {
    /// The normal map the height map is (or will be) made from (nil: none).
    id: Uuid,
    /// The decoded texels it was made from (another decode: again).
    source: Option<Arc<SculptMap>>,
    /// (in a mutex only so the scene stays `Sync`: it is read through
    /// `get_mut`, never locked)
    pending: Option<Mutex<mpsc::Receiver<Option<HeightMap>>>>,
    map: Option<Arc<HeightMap>>,
}

impl Slot {
    fn busy(&self) -> bool {
        self.source.is_some() || self.pending.is_some() || self.map.is_some()
    }

    /// The height map back from the pool, once it is done.
    fn receive(&mut self) {
        let Some(pending) = &mut self.pending else {
            return;
        };
        let receiver = pending.get_mut().unwrap_or_else(std::sync::PoisonError::into_inner);
        match receiver.try_recv() {
            Ok(map) => {
                if map.is_none() {
                    log::warn!("water shape: the normal map {} gave no height map (not three channels?)", self.id);
                }
                self.map = map.map(Arc::new);
                self.pending = None;
            }
            Err(mpsc::TryRecvError::Disconnected) => self.pending = None,
            Err(mpsc::TryRecvError::Empty) => {}
        }
    }

    /// The decoded texels to make the height map from: new ones, with no
    /// height map on its way.
    fn wanted(&self, textures: &TextureStreamer) -> Option<Arc<SculptMap>> {
        if self.id.is_nil() || self.pending.is_some() {
            return None;
        }
        let source = textures.sculpt_map(&self.id)?;
        if self.source.as_ref().is_some_and(|old| Arc::ptr_eq(old, &source)) {
            None
        } else {
            Some(source)
        }
    }
}

#[derive(Default)]
pub struct Builder {
    /// The first normal map's height map, then the second one's while the
    /// environment cross-fades two (patch 998).
    slots: [Slot; 2],
    /// The cross-fade's second normal map (nil: none), set before `feed`.
    pub(super) next_id: Uuid,
    /// The normal maps' own wave directions: wave1 xy, wave2 zw.
    waves: [f32; 4],
    next_waves: [f32; 4],
    /// The second map's share (0: the first alone).
    blend: f32,
}

impl Builder {
    /// Each frame before the water effects: `id` the environment's water
    /// normal map (nil: none), `next_id` the second one of a cross-fade.
    pub fn feed(&mut self, textures: &mut TextureStreamer, id: Uuid, settings: WaterEffects) {
        if !shape::on(settings) || id.is_nil() {
            if self.slots.iter().any(Slot::busy) {
                self.slots = Default::default();
            }
            return;
        }
        // a fade's second map becoming the first (the fade ends, or a fade
        // begun over it): its height map goes with it
        let next = if self.next_id == id { Uuid::nil() } else { self.next_id };
        if id != self.slots[0].id && id == self.slots[1].id {
            self.slots.swap(0, 1);
        }
        for (slot, want) in self.slots.iter_mut().zip([id, next]) {
            if slot.id != want {
                *slot = Slot {
                    id: want,
                    ..Default::default()
                };
            }
            if !want.is_nil() {
                textures.want_sculpt(&want);
            }
            slot.receive();
        }
        // one height map sent off a frame, the first map's first
        let Some((index, source)) = (0..2).find_map(|i| self.slots[i].wanted(textures).map(|s| (i, s))) else {
            return;
        };
        self.slots[index].source = Some(source.clone());
        let (sender, receiver) = mpsc::channel();
        rayon::spawn(move || {
            let map = HeightMap::from_normal_map(source.width, source.height, source.components, &source.pixels);
            // (a map no longer wanted finds its receiver gone)
            let _ = sender.send(map);
        });
        self.slots[index].pending = Some(Mutex::new(receiver));
        log::debug!("water shape: height map of {} on its way", self.slots[index].id);
    }

    /// This frame's shape (`Ocean::shape`): the newest height map while the
    /// shape takes part, and a cross-fade's second one once it is made.
    pub fn shape(&self, settings: WaterEffects) -> Shape {
        let on = shape::on(settings);
        let next = self.slots[1].map.clone().filter(|_| on && self.blend > 0.0);
        Shape {
            map: self.slots[0].map.clone().filter(|_| on),
            waves: self.waves,
            blend: if next.is_some() { self.blend } else { 0.0 },
            next,
            next_waves: self.next_waves,
        }
    }
}

impl Scene {
    /// The environment's wave directions, which scroll the normal map's
    /// layers (the shading's `frame.water_waves`), before the water effects.
    /// While the water cross-fades (patch 998) each map's own, and the
    /// second one's share (0 while it loads).
    pub fn set_water_shape_waves(&mut self, water: &aurora_render::WaterParams) {
        let finite = |w: [f32; 4]| if w.iter().all(|v| v.is_finite()) { w } else { [0.0; 4] };
        let f = &water.fade;
        let (w1, w2) = if f.on {
            (f.first_wave1, f.first_wave2)
        } else {
            (water.wave1, water.wave2)
        };
        let b = &mut self.water_effects.water_shape;
        b.waves = finite([w1.x, w1.y, w2.x, w2.y]);
        b.next_waves = finite([f.next_wave1.x, f.next_wave1.y, f.next_wave2.x, f.next_wave2.y]);
        b.blend = if f.on && f.next.blend.is_finite() {
            f.next.blend.clamp(0.0, 1.0)
        } else {
            0.0
        };
    }

    /// Hands the frame's normal map to the builder (`update_water_effects`).
    pub(super) fn feed_water_shape(&mut self, effects: &mut super::water_effects::Effects, settings: WaterEffects) {
        // a cross-fade's second normal map (patch 998), kept by the scene
        // beside the first one
        effects.water_shape.next_id = self.env_fade_tex[3].0;
        effects.water_shape.feed(&mut self.textures, self.env_tex[3].0, settings);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat() -> Option<Arc<HeightMap>> {
        HeightMap::from_heights(vec![0.0; (shape::SIZE * shape::SIZE) as usize]).map(Arc::new)
    }

    #[test]
    fn the_waves_are_kept_and_the_shape_is_off_without_a_ratio() {
        let mut b = Builder {
            waves: [0.6, -0.8, 0.1, 0.9],
            ..Default::default()
        };
        b.slots[0].map = flat();
        let on = WaterEffects {
            shape_ratio: 0.5,
            ..Default::default()
        };
        assert!(b.shape(on).map.is_some());
        assert_eq!(b.shape(on).waves, [0.6, -0.8, 0.1, 0.9]);
        assert!(b.shape(on).next.is_none() && b.shape(on).blend == 0.0, "no fade");
        assert!(b.shape(WaterEffects::default()).map.is_none(), "Ratio 0: no map handed over");
        let mut textures = TextureStreamer::new(std::path::PathBuf::new());
        b.feed(&mut textures, Uuid::from_u128(7), WaterEffects::default());
        assert!(b.slots[0].map.is_none() && b.slots[0].pending.is_none(), "off: dropped");
        assert_eq!(b.waves, [0.6, -0.8, 0.1, 0.9], "the directions stay");
    }

    #[test]
    fn a_cross_fade_keeps_both_height_maps_and_hands_the_second_over() {
        let (a, c) = (Uuid::from_u128(7), Uuid::from_u128(8));
        let on = WaterEffects {
            shape_ratio: 0.5,
            ..Default::default()
        };
        let mut textures = TextureStreamer::new(std::path::PathBuf::new());
        let mut b = Builder::default();
        b.slots[0] = Slot {
            id: a,
            map: flat(),
            ..Default::default()
        };
        // the fade starts: a slot of its own for the second map, the first kept
        b.next_id = c;
        b.feed(&mut textures, a, on);
        assert_eq!((b.slots[0].id, b.slots[1].id), (a, c));
        assert!(b.slots[0].map.is_some() && b.slots[1].map.is_none());
        b.blend = 0.4;
        assert!(
            b.shape(on).next.is_none() && b.shape(on).blend == 0.0,
            "not made yet: the first alone"
        );
        b.slots[1].map = flat();
        assert!(b.shape(on).next.is_some() && b.shape(on).blend == 0.4);
        let second = b.slots[1].map.clone();
        // the fade ends: the second map's height map is the first one's, not
        // made again
        b.next_id = Uuid::nil();
        b.feed(&mut textures, c, on);
        assert_eq!((b.slots[0].id, b.slots[1].id), (c, Uuid::nil()));
        assert!(b.slots[0].map.as_ref().zip(second.as_ref()).is_some_and(|(x, y)| Arc::ptr_eq(x, y)));
        assert!(!b.slots[1].busy());
        // a fade to the same map is none
        b.next_id = c;
        b.feed(&mut textures, c, on);
        assert!(b.slots[1].id.is_nil());
    }
}
