//! Calm water (local patch set, aurora-render ocean_swell.rs): the fine
//! calm-distance map around the camera. Every second while the calming is
//! on, the frame lists the static objects whose box reaches the water line
//! (in parallel) and the terrain revisions of the regions under the map;
//! when they, the camera's step or a water level changed, a copy of those
//! regions' heightmaps and the footprints go to the thread pool, which
//! marks the land and the footprints on a 1 m grid and runs the distance
//! transform. The new map replaces the old one when it is done; the frame
//! never waits for it. Nothing is sent and nothing is written.
//!
//! A footprint is the object's drawn box (its scale, turned and placed as
//! it is drawn) where it crosses the water's level, give or take `BAND`:
//! exact for the boxes and cylinders that make most walls, piers and
//! pilings, a little generous for round or cut shapes and for meshes
//! (whose box is their bounds), and it costs a few multiplications per
//! cell. Physical objects, moving keyframed ones (ocean_ride.rs
//! `moving_root`, held `KFM_HOLD` seconds after they stop), avatars and
//! attachments never calm the water.
// aurora-local-patch: ocean-swell-calm-module

use super::Scene;
use super::ocean_ride::{self, Movers};
use crate::world::World;
use crate::world::objects::Object;
use crate::world::terrain::Heightmap;
use aurora_render::WaterEffects;
use aurora_render::water_effects::ocean::{self, FineMap};
use glam::{Mat3, Quat, Vec2, Vec3};
use rayon::prelude::*;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Instant;

/// The map moves by this step (m) and is centred on the step's square
/// that holds the camera: the camera is at least 224 m from its edges.
pub const STEP: f32 = 64.0;
/// How often the objects and the terrain are looked at again (s).
const CHECK: f32 = 1.0;
/// How far above or below the water's level (m) a box still touches it.
pub const BAND: f32 = 0.15;

/// An object's drawn box where it may meet the water.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Footprint {
    pub center: Vec3,
    pub rotation: Quat,
    pub half: Vec3,
    /// The water's level under it (m).
    pub level: f32,
}

impl Footprint {
    fn contains(&self, p: Vec3) -> bool {
        let local = self.rotation.inverse() * (p - self.center);
        local.abs().cmple(self.half + Vec3::splat(1e-4)).all()
    }

    /// Whether the box meets the water's line at `xy`.
    pub fn at_water(&self, xy: Vec2) -> bool {
        [-BAND, 0.0, BAND].into_iter().any(|dz| self.contains(xy.extend(self.level + dz)))
    }

    /// Half its extent along each world axis.
    fn extent(&self) -> Vec3 {
        let r = Mat3::from_quat(self.rotation);
        r.x_axis.abs() * self.half.x + r.y_axis.abs() * self.half.y + r.z_axis.abs() * self.half.z
    }

    /// Whether the box reaches into the band around the water's level.
    pub fn touches_water(&self) -> bool {
        let e = self.extent().z;
        self.center.z - e <= self.level + BAND && self.center.z + e >= self.level - BAND
    }
}

/// A region's terrain, copied for the thread pool.
pub struct Terrain {
    pub offset: Vec2,
    pub map: Heightmap,
    pub water: f32,
}

/// Land: terrain at or above its region's water (`ocean::LAND_MARGIN`) or a
/// patch not received yet; water with no region under it is open sea.
fn terrain_land(terrain: &[Terrain], xy: Vec2) -> bool {
    for t in terrain {
        let local = xy - t.offset;
        let (w, h) = (t.map.size_x as f32, t.map.size_y as f32);
        if local.x < 0.0 || local.y < 0.0 || local.x >= w || local.y >= h {
            continue;
        }
        let patch = (local.y as u32 / 16) * t.map.patches_x + local.x as u32 / 16;
        if !t.map.patch_received.get(patch as usize).copied().unwrap_or(false) {
            return true;
        }
        return t.map.sample(local.x, local.y) >= t.water - ocean::LAND_MARGIN;
    }
    false
}

/// The map's land cells (`ocean::FINE_SIZE`², row by row from `origin`):
/// the terrain's and the footprints'.
pub fn rasterize(origin: Vec2, terrain: &[Terrain], footprints: &[Footprint]) -> Vec<bool> {
    let n = ocean::FINE_SIZE as usize;
    let cell = ocean::FINE_CELL;
    let centre = |x: usize, y: usize| origin + (Vec2::new(x as f32, y as f32) + Vec2::splat(0.5)) * cell;
    let mut land: Vec<bool> = (0..n * n)
        .into_par_iter()
        .map(|i| terrain_land(terrain, centre(i % n, i / n)))
        .collect();
    for f in footprints {
        let e = f.extent().truncate();
        let lo = ((f.center.truncate() - e - origin) / cell).floor().max(Vec2::ZERO);
        let hi = ((f.center.truncate() + e - origin) / cell).ceil().min(Vec2::splat(n as f32));
        for y in lo.y as usize..hi.y as usize {
            for x in lo.x as usize..hi.x as usize {
                if !land[y * n + x] && f.at_water(centre(x, y)) {
                    land[y * n + x] = true;
                }
            }
        }
    }
    land
}

/// The map's origin for a camera at `eye` (m).
pub fn origin(eye: Vec2) -> Vec2 {
    let half = ocean::FINE_SIZE as f32 * ocean::FINE_CELL * 0.5;
    (eye / STEP).floor() * STEP + Vec2::splat(STEP * 0.5 - half)
}

/// The root of `o`'s linkset (itself for a root).
pub(super) fn root_of<'a>(world: &'a World, o: &'a Object) -> Option<&'a Object> {
    let mut current = o;
    for _ in 0..256 {
        if current.parent_id == 0 {
            return Some(current);
        }
        let key = current.parent_key()?;
        current = world.objects.get(world.objects.index_of(&key)?)?;
    }
    None
}

/// The static objects whose drawn box meets the water within `[lo, hi]`.
fn footprints(world: &World, lo: Vec2, hi: Vec2, time: f32, movers: &Movers) -> Vec<Footprint> {
    let now = Instant::now();
    let main_level = world.main_water_height();
    world
        .objects
        .slots
        .par_iter()
        .enumerate()
        .filter_map(|(idx, o)| {
            let o = o.as_ref()?;
            if o.is_avatar() || o.pcode != aurora_prim::params::LL_PCODE_VOLUME || o.is_attachment() {
                return None;
            }
            let root = root_of(world, o)?;
            if root.is_avatar() || !ocean_ride::is_static(root, time, movers) {
                return None;
            }
            let (center, rotation, hud) = Scene::object_transform_unshifted(world, idx, now)?;
            if hud || !center.is_finite() {
                return None;
            }
            let level = super::water_effects::water_at(world, center).unwrap_or(main_level);
            let f = Footprint {
                center,
                rotation,
                half: o.scale.abs() * 0.5,
                level,
            };
            let e = f.extent().truncate();
            let inside = (center.truncate() + e).cmpge(lo).all() && (center.truncate() - e).cmple(hi).all();
            (inside && f.touches_water() && f.half.is_finite()).then_some(f)
        })
        .collect()
}

#[derive(Default)]
pub struct Builder {
    /// The main region and its offset the map was made in: another one (a
    /// crossing, a teleport) drops it, its coordinates are stale.
    anchor: Option<(u64, Vec3)>,
    /// What the newest map (or the one on its way) was made from.
    key: Option<u64>,
    checked_at: Option<f32>,
    /// (in a mutex only so the scene stays `Sync`: it is read through
    /// `get_mut`, never locked)
    pending: Option<Mutex<mpsc::Receiver<FineMap>>>,
    fine: Option<Arc<FineMap>>,
    movers: Movers,
}

impl Builder {
    /// This frame's fine map (`Ocean::fine`), the newest one; while the
    /// calming is off nothing is looked at and there is none.
    pub fn update(&mut self, world: &World, eye: Vec3, time: f32, settings: WaterEffects) -> Option<Arc<FineMap>> {
        if !ocean::calm_on(settings) {
            if self.anchor.is_some() || self.key.is_some() || self.pending.is_some() || self.fine.is_some() {
                *self = Self::default();
            }
            return None;
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
                Ok(map) => {
                    self.fine = Some(Arc::new(map));
                    self.pending = None;
                }
                Err(mpsc::TryRecvError::Disconnected) => self.pending = None,
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        let origin = origin(eye.truncate());
        let moved = self.fine.as_ref().is_some_and(|f| Vec2::from_array(f.origin) != origin);
        let due = moved || self.checked_at.is_none_or(|at| !(0.0..CHECK).contains(&(time - at)));
        if due && self.pending.is_none() && eye.is_finite() && anchor.is_some() {
            self.checked_at = Some(time);
            self.rebuild(world, origin, time);
        }
        self.fine.clone()
    }

    fn rebuild(&mut self, world: &World, origin: Vec2, time: f32) {
        let size = ocean::FINE_SIZE as f32 * ocean::FINE_CELL;
        let (lo, hi) = (origin, origin + Vec2::splat(size));
        // the keyframed roots seen moving now (held a while after they stop)
        let moving: Vec<uuid::Uuid> = world
            .objects
            .slots
            .par_iter()
            .filter_map(|o| o.as_ref().filter(|o| ocean_ride::moving_root(o)).map(|o| o.full_id))
            .collect();
        self.movers.note(moving, time);
        let prints = footprints(world, lo, hi, time, &self.movers);
        let regions: Vec<_> = world
            .regions
            .values()
            .filter_map(|r| {
                let offset = world.region_offset(r.handle)?.truncate();
                let far = offset + Vec2::new(r.heightmap.size_x as f32, r.heightmap.size_y as f32);
                let water = r.info.as_ref().map_or(world.main_water_height(), |i| i.water_height);
                (far.cmpge(lo).all() && offset.cmple(hi).all()).then_some((r, offset, water))
            })
            .collect();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        origin.to_array().map(f32::to_bits).hash(&mut hasher);
        for (r, offset, water) in &regions {
            (r.handle, r.terrain_revision, offset.to_array().map(f32::to_bits), water.to_bits()).hash(&mut hasher);
        }
        for f in &prints {
            let q = |v: f32| (v * 100.0).round() as i64;
            (
                q(f.center.x),
                q(f.center.y),
                q(f.center.z),
                q(f.half.x),
                q(f.half.y),
                q(f.half.z),
                q(f.level),
            )
                .hash(&mut hasher);
            f.rotation.to_array().map(|v| (v * 1000.0).round() as i64).hash(&mut hasher);
        }
        let key = hasher.finish();
        if self.key == Some(key) {
            return;
        }
        self.key = Some(key);
        let terrain: Vec<Terrain> = regions
            .into_iter()
            .map(|(r, offset, water)| Terrain {
                offset,
                map: r.heightmap.clone(),
                water,
            })
            .collect();
        let (sender, receiver) = mpsc::channel();
        let count = prints.len();
        rayon::spawn(move || {
            let land = rasterize(origin, &terrain, &prints);
            if let Some(map) = FineMap::build(origin.to_array(), &land) {
                // (a map no longer wanted finds its receiver gone)
                let _ = sender.send(map);
            }
        });
        self.pending = Some(Mutex::new(receiver));
        log::debug!("calm water: distance map at {origin:?} with {count} static footprints on its way");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(level: f32, ground: f32) -> Terrain {
        let mut map = Heightmap::new(256, 256);
        map.heights.iter_mut().for_each(|h| *h = ground);
        map.patch_received.iter_mut().for_each(|r| *r = true);
        Terrain {
            offset: Vec2::ZERO,
            map,
            water: level,
        }
    }

    #[test]
    fn the_map_holds_terrain_and_static_footprints_at_the_water_line() {
        let origin = Vec2::new(-128.0, -128.0);
        // a region of deep water, a beach on its west edge (x < 20 is land)
        let mut region = flat(20.0, 5.0);
        for y in 0..256u32 {
            for x in 0..20u32 {
                region.map.heights[(y * 256 + x) as usize] = 25.0;
            }
        }
        // a pier 2 m wide crossing the water at x = 100, a deck floating
        // high above the water (does not touch it), a rock under the water
        // touching its line
        let pier = Footprint {
            center: Vec3::new(100.0, 60.0, 19.0),
            rotation: Quat::from_rotation_z(0.0),
            half: Vec3::new(1.0, 20.0, 2.0),
            level: 20.0,
        };
        let deck = Footprint {
            center: Vec3::new(160.0, 60.0, 25.0),
            half: Vec3::new(5.0, 5.0, 0.2),
            ..pier
        };
        let rock = Footprint {
            center: Vec3::new(60.0, 30.0, 18.9),
            half: Vec3::splat(1.0),
            ..pier
        };
        assert!(pier.touches_water() && rock.touches_water() && !deck.touches_water());
        let land = rasterize(origin, &[region], &[pier, rock]);
        let n = ocean::FINE_SIZE as usize;
        let at = |x: f32, y: f32| {
            let c = ((Vec2::new(x, y) - origin) / ocean::FINE_CELL).floor();
            land[c.y as usize * n + c.x as usize]
        };
        assert!(at(10.5, 50.5), "the beach");
        assert!(!at(50.5, 50.5), "open water");
        assert!(at(100.2, 60.5) && at(99.6, 75.5), "the pier");
        assert!(!at(102.5, 60.5), "beside the pier");
        assert!(at(60.5, 30.5), "the rock at the water line");
        assert!(!at(-50.5, 50.5), "no region: open sea");
        // its distances
        let map = FineMap::build(origin.to_array(), &land).expect("map");
        let (d, _, w) = map.distance(Vec2::new(110.0, 60.0));
        assert!((d - 9.0).abs() < 1.0 && w == 1.0, "{d}");
        // a turned pier: its footprint follows its turn
        let turned = Footprint {
            rotation: Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
            ..pier
        };
        let land = rasterize(origin, &[], &[turned]);
        let at = |x: f32, y: f32| {
            let c = ((Vec2::new(x, y) - origin) / ocean::FINE_CELL).floor();
            land[c.y as usize * n + c.x as usize]
        };
        assert!(at(115.5, 60.5) && !at(100.5, 65.5));
    }

    #[test]
    fn physical_and_keyframed_objects_never_calm_the_water() {
        let lib = Arc::new(crate::scene::avatar::AvatarLibrary::load());
        let mut world = World::new(lib);
        for ev in crate::demo::events() {
            world.apply(ev);
        }
        let region = world.main_region.expect("demo region");
        // the ocean demo's fixtures: a physical raft, a still box, a
        // physical box in the air, a keyframed boat, a pier, two avatars
        let mut fixtures = Vec::new();
        crate::demo::ocean::objects("medium:ocean", 0.0, &mut fixtures);
        let mut index = std::collections::HashMap::new();
        for u in fixtures {
            let local = u.local_id;
            index.insert(local, world.objects.upsert(region, u));
        }
        let id = |local: u32| world.objects.get(index[&local]).expect("fixture").full_id;
        let offset = world.region_offset(region).expect("offset").truncate();
        let (lo, hi) = (offset + Vec2::new(-320.0, 60.0), offset + Vec2::new(-200.0, 220.0));
        let moving = |world: &World| -> Vec<uuid::Uuid> {
            world
                .objects
                .slots
                .iter()
                .filter_map(|o| o.as_ref().filter(|o| ocean_ride::moving_root(o)).map(|o| o.full_id))
                .collect()
        };
        assert_eq!(moving(&world), vec![id(99545)], "only the keyframed boat moves");
        let mut movers = Movers::default();
        movers.note(moving(&world), 1.0);
        let near = |prints: &[Footprint], local: u32| {
            let at = Scene::object_transform_unshifted(&world, index[&local], Instant::now())
                .expect("transform")
                .0;
            prints.iter().any(|f| f.center.distance(at) < 0.01)
        };
        let prints = footprints(&world, lo, hi, 1.0, &movers);
        assert!(
            near(&prints, 99541) && near(&prints, 99546),
            "the still box and the pier calm: {prints:?}"
        );
        assert!(
            !near(&prints, 99540) && !near(&prints, 99542) && !near(&prints, 99545),
            "{prints:?}"
        );
        assert_eq!(prints.len(), 2);
        // the boat stops: held as moving a while, then static
        let boat = index[&99545];
        world.objects.get_mut(boat).expect("boat").velocity = Vec3::ZERO;
        assert!(moving(&world).is_empty());
        assert_eq!(footprints(&world, lo, hi, 3.0, &movers).len(), 2);
        let later = 1.0 + ocean_ride::KFM_HOLD + 0.1;
        movers.note(Vec::new(), later);
        assert_eq!(footprints(&world, lo, hi, later, &movers).len(), 3);
    }

    #[test]
    fn the_map_moves_by_steps_and_keeps_the_camera_far_from_its_edges() {
        let size = ocean::FINE_SIZE as f32 * ocean::FINE_CELL;
        for eye in [Vec2::ZERO, Vec2::new(63.9, -0.1), Vec2::new(-700.0, 1300.0)] {
            let o = origin(eye);
            let margin = (eye - o).min(o + Vec2::splat(size) - eye).min_element();
            assert!(margin >= 0.5 * (size - STEP) - 1e-3, "{eye} {o} {margin}");
        }
    }

    #[test]
    fn nothing_is_built_while_the_calming_is_off() {
        let world = World::new(Arc::new(crate::scene::avatar::AvatarLibrary::load()));
        let mut builder = Builder::default();
        let off = WaterEffects {
            calm_edge: 1.0,
            ..Default::default()
        };
        assert!(builder.update(&world, Vec3::ZERO, 1.0, off).is_none());
        assert!(builder.pending.is_none() && builder.key.is_none());
        // on, but no region yet: nothing either
        assert!(builder.update(&world, Vec3::ZERO, 1.0, WaterEffects::default()).is_none());
        assert!(builder.pending.is_none());
    }
}
