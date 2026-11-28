//! Open-sea swell (local patch set, aurora-render ocean_swell.rs): the
//! land-distance field around the camera. Terrain and water heights are
//! sampled on the frame (in parallel, every `RESAMPLE` seconds or when the
//! field has to move); the distance transform runs on the thread pool and
//! the new field replaces the old one when it is done. Nothing is sent and
//! nothing is written.
// aurora-local-patch: ocean-swell-field-module

use crate::world::World;
use aurora_render::WaterEffects;
use aurora_render::water_effects::ocean::{self, Cell, Field, Ocean};
use glam::{Vec2, Vec3};
use rayon::prelude::*;
use std::sync::{Arc, Mutex, mpsc};

/// The field moves by this step (m) and is centred on the step's square
/// that holds the camera: the camera is at least 896 m from its edges.
const STEP: f32 = 256.0;
/// How often the heights are sampled again (s): a terrain edit, a region
/// that arrives or a new water height rebuilds the field.
const RESAMPLE: f32 = 2.0;

#[derive(Default)]
pub struct Builder {
    /// The main region and its offset the field was sampled in: another
    /// one (a crossing, a teleport) drops it, its coordinates are stale.
    anchor: Option<(u64, Vec3)>,
    origin: Option<Vec2>,
    cells: Vec<Cell>,
    sampled_at: f32,
    /// (in a mutex only so the scene stays `Sync`: it is read through
    /// `get_mut`, never locked)
    pending: Option<Mutex<mpsc::Receiver<Field>>>,
    field: Option<Arc<Field>>,
}

/// One cell: land (terrain at or above the water, or a region without its
/// terrain), region water and its depth, or void water (open sea).
pub fn cell(level: Option<f32>, terrain: Option<f32>, void_level: f32) -> Cell {
    let water = |level: f32, ground: f32| {
        if ground >= level - ocean::LAND_MARGIN {
            Cell::Land
        } else {
            Cell::Water(level - ground)
        }
    };
    match (level, terrain) {
        (None, None) => Cell::Void,
        (None, Some(ground)) => water(void_level, ground),
        (Some(_), None) => Cell::Land,
        (Some(level), Some(ground)) => water(level, ground),
    }
}

/// The field's origin for a camera at `eye` (m).
pub fn origin(eye: Vec2) -> Vec2 {
    let half = ocean::FIELD_SIZE as f32 * ocean::FIELD_CELL * 0.5;
    (eye / STEP).floor() * STEP + Vec2::splat(STEP * 0.5 - half)
}

fn sample(world: &World, origin: Vec2) -> Vec<Cell> {
    let n = ocean::FIELD_SIZE as usize;
    let void_level = world.main_water_height();
    (0..n * n)
        .into_par_iter()
        .map(|i| {
            let xy = origin + (Vec2::new((i % n) as f32, (i / n) as f32) + Vec2::splat(0.5)) * ocean::FIELD_CELL;
            let p = xy.extend(0.0);
            cell(super::water_effects::water_at(world, p), world.ground_height(p), void_level)
        })
        .collect()
}

impl Builder {
    /// This frame's ocean state (`WaterGround::ocean`): the newest field,
    /// the swell's direction from the environment's big wave `wave1`.
    /// While the swell is off nothing is sampled and the state is empty.
    pub fn update(&mut self, world: &World, eye: Vec3, time: f32, settings: WaterEffects, wave1: Vec2) -> Ocean {
        if !ocean::swell_on(settings) {
            if self.anchor.is_some() || self.origin.is_some() || self.pending.is_some() || self.field.is_some() {
                *self = Self::default();
            }
            return Ocean::default();
        }
        let anchor = world.main_region.and_then(|h| Some((h, world.region_offset(h)?)));
        if anchor != self.anchor {
            *self = Self {
                anchor,
                ..Default::default()
            };
        }
        if let Some(pending) = &mut self.pending {
            let receiver = pending.get_mut().unwrap_or_else(std::sync::PoisonError::into_inner);
            match receiver.try_recv() {
                Ok(field) => {
                    self.field = Some(Arc::new(field));
                    self.pending = None;
                }
                Err(mpsc::TryRecvError::Disconnected) => self.pending = None,
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        let origin = origin(eye.truncate());
        let due = self.origin != Some(origin) || !(0.0..RESAMPLE).contains(&(time - self.sampled_at));
        if due && self.pending.is_none() && eye.is_finite() && anchor.is_some() {
            self.sampled_at = time;
            let cells = sample(world, origin);
            if self.origin != Some(origin) || cells != self.cells {
                self.origin = Some(origin);
                self.cells.clone_from(&cells);
                let (sender, receiver) = mpsc::channel();
                rayon::spawn(move || {
                    if let Some(field) = Field::build(origin.to_array(), &cells) {
                        // (a field no longer wanted finds its receiver gone)
                        let _ = sender.send(field);
                    }
                });
                self.pending = Some(Mutex::new(receiver));
                log::debug!("open-sea swell: land-distance field at {origin:?} on its way");
            }
        }
        Ocean {
            wind: ocean::wind(wave1).to_array(),
            field: self.field.clone(),
            ..Default::default()
        }
    }
}

/// The big water waves (EEP wave1) the open-sea swell follows: the
/// environment's, and no second swell. With patch 998, while the
/// environment cross-fades the water (the viewer's world/eep_fade.rs), its
/// step env-texture-fade-ocean-waves returns first the first layer's own
/// and the second's with its share, the one the water normal maps take, so
/// that each swell keeps its direction and arrives with its normal map.
pub fn swell_waves(water: &aurora_render::WaterParams) -> (Vec2, Option<(Vec2, f32)>) {
    (water.wave1, None)
}

impl super::Scene {
    /// The environment's big water waves, whose directions the open-sea
    /// swell follows ([`swell_waves`]).
    pub fn set_ocean_wave(&mut self, water: &aurora_render::WaterParams) {
        let (wave1, next) = swell_waves(water);
        self.water_effects.ocean_wave1 = wave1;
        self.water_effects.ocean_next = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cells_are_land_water_or_void() {
        assert_eq!(cell(None, None, 20.0), Cell::Void);
        assert_eq!(cell(Some(20.0), None, 20.0), Cell::Land, "a missing terrain patch is land");
        assert_eq!(cell(Some(20.0), Some(19.8), 20.0), Cell::Land, "within the margin");
        assert_eq!(cell(Some(20.0), Some(25.0), 20.0), Cell::Land);
        assert_eq!(cell(Some(20.0), Some(12.0), 20.0), Cell::Water(8.0));
        assert_eq!(cell(None, Some(5.0), 20.0), Cell::Water(15.0));
        assert_eq!(cell(None, Some(21.0), 20.0), Cell::Land);
    }

    #[test]
    fn the_field_moves_by_steps_and_keeps_the_camera_far_from_its_edges() {
        let size = ocean::FIELD_SIZE as f32 * ocean::FIELD_CELL;
        for eye in [
            Vec2::ZERO,
            Vec2::new(255.9, -0.1),
            Vec2::new(-700.0, 1300.0),
            Vec2::new(128.0, 128.0),
        ] {
            let o = origin(eye);
            assert_eq!(o, origin(eye + Vec2::splat(0.001)));
            let margin = (eye - o).min(o + Vec2::splat(size) - eye).min_element();
            assert!(margin >= 0.5 * (size - STEP) - 1e-3, "{eye} {o} {margin}");
            assert_eq!((o + Vec2::splat(size * 0.5 - STEP * 0.5)) % STEP, Vec2::ZERO);
        }
    }

    #[test]
    fn nothing_is_sampled_while_the_swell_is_off() {
        let world = World::new(Arc::new(crate::scene::avatar::AvatarLibrary::load()));
        let mut builder = Builder::default();
        let off = WaterEffects::default();
        let ocean = builder.update(&world, Vec3::ZERO, 1.0, off, Vec2::X);
        assert!(ocean.field.is_none() && builder.origin.is_none() && builder.pending.is_none());
        // on, but no region yet: nothing either
        let on = WaterEffects {
            ocean_depth: 3.0,
            ..Default::default()
        };
        let ocean = builder.update(&world, Vec3::ZERO, 1.0, on, Vec2::X);
        assert!(ocean.field.is_none() && builder.pending.is_none());
        assert_eq!(ocean.wind, [-1.0, 0.0]);
    }

    #[test]
    fn the_swell_takes_the_frames_big_wave_and_a_second_swells_direction_and_share() {
        let water = aurora_render::WaterParams {
            wave1: Vec2::new(0.4, 0.4),
            ..Default::default()
        };
        assert_eq!(swell_waves(&water), (Vec2::new(0.4, 0.4), None), "no fade: the frame's");
        // a second swell (patch 998's cross-fade gives it): its direction
        // and the share
        let o = Ocean::default().with_next(Some((Vec2::new(0.0, -2.0), 0.25)));
        assert_eq!((o.next_wind, o.share), ([0.0, 1.0], 0.25));
    }
}
