//! Open-sea swell (local patch set, aurora-render ocean_swell.rs): what
//! rides it, on this screen only, while "Objects ride the swell" is on.
//! Physical linksets and moving keyframed ones (`moving_root`) whose root
//! floats at the surface, and avatars swimming there, are drawn higher or
//! lower by the displayed surface's height
//! (`World::ocean_ride`, read by `World::water_visual_dz`, which the drawn
//! transforms add: `Scene::object_transform`); what is linked to them, the
//! avatars sitting on them included, follows its root. The simulator's
//! positions, the physics and the network are untouched. Static objects
//! never move. Our own avatar keeps its heave (movement/water_float.rs,
//! while swimming or walking on water, the camera following its drawn
//! body) and is never listed here.
//!
//! A keyframed motion has no flag of its own in what the viewer receives.
//! What tells it apart: the simulator reports a velocity for the root of a
//! linkset it moves along a keyframed path, and the viewer moves such a
//! root by it between updates (LLViewerObject::interpolateLinearMotion,
//! indra/newview/llviewerobject.cpp; ObjectStore::motion_time here), while
//! a script that moves a non-physical object by steps (llSetPos and the
//! like) sends positions with no velocity. So a non-physical root that is
//! not an avatar and has a velocity above `KFM_SPEED` is keyframed, and
//! stays so for `KFM_HOLD` seconds after it stops (a motion's pause between
//! two frames), so that it neither drops off the swell nor starts calming
//! the water at each stop.
// aurora-local-patch: ocean-swell-ride-module

use super::Scene;
use crate::world::World;
use aurora_render::WaterEffects;
use aurora_render::water_effects::ocean;
use glam::{Quat, Vec3};
use rayon::prelude::*;
use std::time::Instant;
use uuid::Uuid;

/// FLAGS_USE_PHYSICS of the update flags.
const PHYSICS: u32 = 1;
/// Within this distance of the camera (m), inside the swell's full-detail
/// range (ocean::FAR_FROM).
const RANGE: f32 = 500.0;
/// The candidates are listed again this often (s), at most `MAX_RIDERS`.
const SCAN: f32 = 0.5;
const MAX_RIDERS: usize = 64;
/// How far (m) a physical root may stand off the surface and still float
/// on it, and the water a floater keeps under it (m): one resting on the
/// bed does not ride.
const AT_SURFACE: f32 = 0.3;
const OFF_BED: f32 = 0.3;
/// A swimmer's root is within half its height and this (m) of the surface,
/// as our own avatar's heave asks (water_float.rs at_surface).
const SWIM_BAND: f32 = 0.5;
/// A keyframed root moves at least this fast (m/s), and is held as moving
/// this long after it stops (s).
pub const KFM_SPEED: f32 = 0.02;
pub const KFM_HOLD: f32 = 5.0;

/// A non-physical root, not an avatar, that the simulator reports moving:
/// a keyframed motion (above).
pub fn moving_root(o: &crate::world::objects::Object) -> bool {
    o.parent_id == 0
        && !o.is_avatar()
        && o.update_flags & PHYSICS == 0
        && o.velocity.is_finite()
        && o.velocity.length_squared() > KFM_SPEED * KFM_SPEED
}

/// The keyframed roots seen moving lately, by id, with when.
#[derive(Default)]
pub struct Movers {
    seen: std::collections::HashMap<Uuid, f32>,
}

impl Movers {
    /// `ids` move at `time`; the others seen longer than `KFM_HOLD` ago are
    /// forgotten.
    pub fn note(&mut self, ids: impl IntoIterator<Item = Uuid>, time: f32) {
        for id in ids {
            self.seen.insert(id, time);
        }
        self.seen.retain(|_, at| (0.0..KFM_HOLD).contains(&(time - *at)));
    }

    /// Seen moving within `KFM_HOLD` of `time`.
    pub fn recent(&self, id: &Uuid, time: f32) -> bool {
        self.seen.get(id).is_some_and(|at| (0.0..KFM_HOLD).contains(&(time - *at)))
    }
}

/// A root that never moves: not physical, not keyframed (now or lately),
/// not an avatar. Only these calm the water (ocean_calm.rs).
pub fn is_static(root: &crate::world::objects::Object, time: f32, movers: &Movers) -> bool {
    !root.is_avatar() && root.update_flags & PHYSICS == 0 && !moving_root(root) && !movers.recent(&root.full_id, time)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kind {
    /// A physical linkset's root.
    Physical,
    /// A keyframed linkset's root, moving (or lately).
    Keyframed,
    /// Anything else that is not an avatar: it never rides.
    Fixed,
    /// An avatar standing on its own (not seated).
    Avatar,
    /// An avatar sitting on something: it follows its seat.
    Seated,
}

/// What the rule looks at: the vertical extent of the object as the
/// simulator places it, the flat water and the terrain under it.
#[derive(Debug, Clone, Copy)]
pub struct Probe {
    pub kind: Kind,
    pub bottom: f32,
    pub top: f32,
    pub level: f32,
    pub ground: f32,
}

/// Whether it rides the swell.
pub fn rides(p: Probe) -> bool {
    match p.kind {
        Kind::Physical | Kind::Keyframed => {
            p.bottom <= p.level + AT_SURFACE && p.top >= p.level - AT_SURFACE && p.bottom - p.ground > OFF_BED
        }
        Kind::Avatar => {
            let center = 0.5 * (p.bottom + p.top);
            (center - p.level).abs() <= 0.5 * (p.top - p.bottom) + SWIM_BAND && p.bottom - p.ground > OFF_BED
        }
        Kind::Fixed | Kind::Seated => false,
    }
}

/// One step of a drawn offset toward its target: the same response and
/// speed limit as our own heave (water_float.rs), so nothing pops when it
/// starts or stops riding.
pub fn follow(z: f32, target: f32, dt: f32, down: f32, up: f32) -> f32 {
    let (_, _, speed, settle) = crate::movement::water_float::heave_limits(down, up);
    z + ((target - z) * (1.0 - (-dt / settle).exp())).clamp(-dt * speed, dt * speed)
}

#[derive(Default)]
pub struct Riders {
    candidates: Vec<(usize, Uuid, Kind)>,
    scan_at: Option<f32>,
    last: Option<f32>,
    movers: Movers,
}

/// `keyframed`: a root seen moving lately (`Movers::recent`).
fn kind(o: &crate::world::objects::Object, agent: Uuid, keyframed: bool) -> Kind {
    if o.is_avatar() {
        if o.parent_id != 0 || o.full_id == agent {
            Kind::Seated
        } else {
            Kind::Avatar
        }
    } else if o.parent_id == 0 && o.update_flags & PHYSICS != 0 {
        Kind::Physical
    } else if o.parent_id == 0 && (moving_root(o) || keyframed) {
        Kind::Keyframed
    } else {
        Kind::Fixed
    }
}

/// Half the vertical extent of a root prim turned by `r` (an avatar: half
/// its height).
fn half_height(o: &crate::world::objects::Object, r: Quat) -> f32 {
    let half = o.scale.abs() * 0.5;
    if o.is_avatar() {
        half.z
    } else {
        (r * Vec3::X).z.abs() * half.x + (r * Vec3::Y).z.abs() * half.y + (r * Vec3::Z).z.abs() * half.z
    }
}

impl Scene {
    /// Each frame, after the water effects: the drawn offsets of what rides
    /// the swell (`on`: the switch), eased toward the surface under them;
    /// the linksets whose offset moved are placed again in the next sync.
    /// Off, the offsets ease back to 0 and are dropped.
    pub fn update_ocean_ride(&mut self, world: &mut World, eye: Vec3, time: f32, settings: WaterEffects, on: bool) {
        let active = on && self.lists.water_ground.ocean.active(settings).is_some();
        if !active && world.ocean_ride.is_empty() {
            if self.ocean_riders.last.is_some() {
                self.ocean_riders = Riders::default();
            }
            return;
        }
        let dt = self.ocean_riders.last.map_or(0.0, |last| (time - last).clamp(0.0, 0.25));
        self.ocean_riders.last = Some(time);
        let now = Instant::now();
        let due = self.ocean_riders.scan_at.is_none_or(|at| !(0.0..SCAN).contains(&(time - at)));
        if active && due {
            self.ocean_riders.scan_at = Some(time);
            let agent = world.agent_id;
            let world_ref: &World = world;
            let movers = &self.ocean_riders.movers;
            let mut near: Vec<(f32, usize, Uuid, Kind)> = world_ref
                .objects
                .slots
                .par_iter()
                .enumerate()
                .filter_map(|(idx, o)| {
                    let o = o.as_ref()?;
                    let kind = kind(o, agent, movers.recent(&o.full_id, time));
                    if !matches!(kind, Kind::Physical | Kind::Keyframed | Kind::Avatar) {
                        return None;
                    }
                    let (p, _, hud) = Scene::object_transform_unshifted(world_ref, idx, now)?;
                    let distance = p.truncate().distance(eye.truncate());
                    (!hud && distance < RANGE).then_some((distance, idx, o.full_id, kind))
                })
                .collect();
            near.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            let moving: Vec<Uuid> = near
                .iter()
                .filter(|(_, idx, _, kind)| *kind == Kind::Keyframed && world_ref.objects.get(*idx).is_some_and(moving_root))
                .map(|(_, _, id, _)| *id)
                .collect();
            self.ocean_riders.movers.note(moving, time);
            self.ocean_riders.candidates = near.into_iter().take(MAX_RIDERS).map(|(_, i, id, k)| (i, id, k)).collect();
        }
        if !active {
            self.ocean_riders.candidates.clear();
        }
        let (down, up) = (settings.ocean_depth, settings.ocean_up);
        let mut seen = Vec::with_capacity(self.ocean_riders.candidates.len());
        for &(idx, id, kind) in &self.ocean_riders.candidates {
            let Some(o) = world.objects.get(idx).filter(|o| o.full_id == id) else {
                continue;
            };
            let Some((p, r, _)) = Scene::object_transform_unshifted(world, idx, now) else {
                continue;
            };
            let level = super::water_effects::water_at(world, p).unwrap_or_else(|| world.main_water_height());
            let ground = world.ground_height(p).unwrap_or(level - ocean::VOID_DEPTH);
            let half = half_height(o, r);
            let probe = Probe {
                kind,
                bottom: p.z - half,
                top: p.z + half,
                level,
                ground,
            };
            let target = if rides(probe) {
                let bed = self.lists.water_ground.sample(p.truncate()).map_or(level - 20.0, |g| g[0]);
                self.lists.water_ground.surface(p.truncate(), time, settings, level, bed) - level
            } else {
                0.0
            };
            seen.push((idx, id, kind, target));
        }
        // what left the list eases back to 0
        for id in world.ocean_ride.keys() {
            if !seen.iter().any(|s| s.1 == *id)
                && let Some(idx) = world.objects.index_of_uuid(id)
            {
                seen.push((idx, *id, Kind::Fixed, 0.0));
            }
        }
        world.ocean_ride.retain(|id, _| seen.iter().any(|s| s.1 == *id));
        for (idx, id, kind, target) in seen {
            let z = world.ocean_ride.get(&id).copied().unwrap_or(0.0);
            let next = follow(z, target, dt, down, up);
            if target == 0.0 && next.abs() < 1e-4 {
                if world.ocean_ride.remove(&id).is_none() {
                    continue;
                }
            } else if next != z || !world.ocean_ride.contains_key(&id) {
                world.ocean_ride.insert(id, next);
            } else {
                continue;
            }
            // (an avatar is placed every frame anyway)
            if kind != Kind::Avatar {
                self.sets.dirty.insert(idx);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn probe(kind: Kind, bottom: f32, top: f32, ground: f32) -> Probe {
        Probe {
            kind,
            bottom,
            top,
            level: 20.0,
            ground,
        }
    }

    #[test]
    fn what_rides_the_swell() {
        // a physical floater at the surface, deep water under it
        assert!(rides(probe(Kind::Physical, 19.7, 20.3, -100.0)));
        assert!(rides(probe(Kind::Physical, 20.2, 21.0, -100.0)), "riding a little high");
        // in the air, sunk, or resting on the bed: no
        assert!(!rides(probe(Kind::Physical, 21.0, 22.0, -100.0)));
        assert!(!rides(probe(Kind::Physical, 15.0, 19.0, -100.0)));
        assert!(!rides(probe(Kind::Physical, 19.5, 20.5, 19.4)));
        // a keyframed boat at the surface rides like a physical one
        assert!(rides(probe(Kind::Keyframed, 19.7, 20.3, -100.0)));
        assert!(!rides(probe(Kind::Keyframed, 21.0, 22.0, -100.0)));
        // not physical, or sitting on something: never by itself
        assert!(!rides(probe(Kind::Fixed, 19.7, 20.3, -100.0)));
        assert!(!rides(probe(Kind::Seated, 19.1, 21.0, -100.0)));
        // a swimmer at the surface; one standing on a shallow bed, one deep under
        assert!(rides(probe(Kind::Avatar, 19.1, 21.0, -100.0)));
        assert!(!rides(probe(Kind::Avatar, 19.1, 21.0, 19.0)));
        assert!(!rides(probe(Kind::Avatar, 14.0, 15.9, -100.0)));
    }

    #[test]
    fn the_offset_eases_in_and_out_without_popping() {
        for fps in [30.0f32, 60.0, 144.0] {
            let dt = 1.0 / fps;
            let mut z = 0.0;
            let mut t = 0.0f32;
            while t < 6.0 {
                let target = -2.5 + 2.5 * (t * 0.7).sin();
                let next = follow(z, target, dt, 5.0, 0.0);
                assert!((next - z).abs() <= dt * 4.0 + 1e-6, "bounded speed");
                z = next;
                t += dt;
            }
            for _ in 0..(fps as usize * 4) {
                z = follow(z, 0.0, dt, 5.0, 0.0);
            }
            assert!(z.abs() < 1e-4, "settles back to the flat drawing");
        }
        assert_eq!(follow(0.3, 0.3, 0.016, 2.0, 0.0), 0.3);
        assert_eq!(follow(0.0, -1.0, 0.0, 2.0, 0.0), 0.0, "no time, no step");
        // a crest above the level: followed up as well
        let mut z = 0.0;
        for _ in 0..600 {
            z = follow(z, 4.0, 1.0 / 60.0, 0.0, 5.0);
        }
        assert!((z - 4.0).abs() < 1e-3, "{z}");
    }

    #[test]
    fn a_riding_linkset_carries_its_seated_avatar_without_moving_simulator_positions() {
        let lib = Arc::new(crate::scene::avatar::AvatarLibrary::load());
        let mut world = World::new(lib);
        for ev in crate::demo::events() {
            world.apply(ev);
        }
        let mut fixtures = Vec::new();
        crate::demo::ocean::objects("medium:ocean", 0.0, &mut fixtures);
        let region = world.main_region.expect("demo region");
        let mut raft = None;
        let mut rider = None;
        for u in fixtures {
            let (local, parent) = (u.local_id, u.parent_id);
            let idx = world.objects.upsert(region, u);
            if local == 99540 {
                raft = Some(idx);
            }
            if parent == 99540 {
                rider = Some(idx);
            }
        }
        let (raft, rider) = (raft.expect("raft"), rider.expect("rider"));
        let now = Instant::now();
        let positions = [raft, rider].map(|i| world.objects.get(i).expect("fixture").position);
        let before = [raft, rider].map(|i| Scene::object_transform(&world, i, now, 0).expect("transform").0);
        assert_eq!(kind(world.objects.get(raft).expect("raft"), world.agent_id, false), Kind::Physical);
        assert_eq!(kind(world.objects.get(rider).expect("rider"), world.agent_id, false), Kind::Seated);
        let raft_id = world.objects.get(raft).expect("raft").full_id;
        world.ocean_ride.insert(raft_id, -1.5);
        for (i, b) in [raft, rider].into_iter().zip(before) {
            let after = Scene::object_transform(&world, i, now, 0).expect("transform").0;
            assert!(after.abs_diff_eq(b - Vec3::Z * 1.5, 1e-4), "{after} vs {b}");
            assert_eq!(Scene::object_transform_unshifted(&world, raft, now).expect("raw").0, before[0]);
        }
        assert_eq!([raft, rider].map(|i| world.objects.get(i).expect("fixture").position), positions);
        // the switch off: nothing listed, nothing drawn off its place
        world.ocean_ride.clear();
        assert_eq!(world.water_visual_dz(&raft_id), 0.0);
        assert_eq!(Scene::object_transform(&world, raft, now, 0).expect("transform").0, before[0]);
    }

    #[test]
    fn a_keyframed_boat_rides_and_its_seated_avatar_follows_static_objects_never_move() {
        let lib = Arc::new(crate::scene::avatar::AvatarLibrary::load());
        let mut world = World::new(lib);
        for ev in crate::demo::events() {
            world.apply(ev);
        }
        let mut fixtures = Vec::new();
        crate::demo::ocean::objects("medium:ocean", 0.0, &mut fixtures);
        let region = world.main_region.expect("demo region");
        let mut index = std::collections::HashMap::new();
        for u in fixtures {
            let local = u.local_id;
            index.insert(local, world.objects.upsert(region, u));
        }
        fn get<'a>(world: &'a World, index: &std::collections::HashMap<u32, usize>, local: u32) -> &'a crate::world::objects::Object {
            world.objects.get(index[&local]).expect("fixture")
        }
        let agent = world.agent_id;
        // the criterion: a non-physical root with a velocity
        assert!(moving_root(get(&world, &index, 99545)));
        assert_eq!(kind(get(&world, &index, 99545), agent, false), Kind::Keyframed);
        for still in [99541, 99546] {
            assert!(!moving_root(get(&world, &index, still)));
            assert_eq!(kind(get(&world, &index, still), agent, false), Kind::Fixed, "static: never moved");
        }
        assert!(!moving_root(get(&world, &index, 99540)), "physical is its own kind");
        // a pause between two keyframes: still keyframed while held
        let mut movers = Movers::default();
        movers.note([get(&world, &index, 99545).full_id], 10.0);
        let boat = index[&99545];
        world.objects.get_mut(boat).expect("boat").velocity = Vec3::ZERO;
        assert!(!moving_root(get(&world, &index, 99545)));
        let recent = movers.recent(&get(&world, &index, 99545).full_id, 12.0);
        assert_eq!(kind(get(&world, &index, 99545), agent, recent), Kind::Keyframed);
        assert!(!is_static(get(&world, &index, 99545), 12.0, &movers));
        movers.note([], 10.0 + KFM_HOLD + 0.5);
        assert!(is_static(get(&world, &index, 99545), 10.0 + KFM_HOLD + 0.5, &movers));
        // drawn offset: the boat's root and an avatar seated on it follow
        // together; the simulator's positions stay
        let mut seated = crate::demo::action_avatar(true);
        seated.local_id = 99_560;
        seated.full_id = Uuid::from_u128(99_560);
        seated.parent_id = 99545;
        seated.position = Vec3::Z * 1.0;
        let rider = world.objects.upsert(region, seated);
        let now = Instant::now();
        let before = [boat, rider].map(|i| Scene::object_transform(&world, i, now, 0).expect("transform").0);
        let boat_id = get(&world, &index, 99545).full_id;
        world.ocean_ride.insert(boat_id, 1.25);
        for (i, b) in [boat, rider].into_iter().zip(before) {
            let after = Scene::object_transform(&world, i, now, 0).expect("transform").0;
            assert!(after.abs_diff_eq(b + Vec3::Z * 1.25, 1e-4), "{after} vs {b}");
        }
        // our own avatar is never listed: it keeps its own heave
        world.ocean_ride.insert(world.agent_id, 3.0);
        assert_eq!(world.water_visual_dz(&world.agent_id), world.water_float.offset());
    }
}
