// aurora-local-patch: region-crossing-origin-scene
//! A region crossing moves the render origin: what rests in the world
//! follows it on the GPU (aurora-render origin_shift.rs) instead of being
//! placed again and uploaded.

#[cfg(test)]
use super::ObjGpu;
use super::{Scene, hover};
use crate::world::World;
use aurora_render::Renderer;
use glam::Vec3;
use std::time::Instant;

impl Scene {
    /// The render origin moved from `before` to `self.last_origin`
    /// (`update_origin`: a crossing or a teleport). The objects that rest in
    /// the world (`ObjGpu::follows_origin`; `update_origin` did not mark
    /// them) move by the delta as a placement would have moved them: their
    /// faces' records on the GPU, their bounds, their object table entry,
    /// their pick entry; the detail levels' camera with them (the camera did
    /// not travel). One whose faces did not all move is placed again as
    /// before. Avatars, what they wear, HUDs, rigged and anchored objects
    /// were marked and are placed again, as before.
    /// `AURORA_ORIGIN_SHIFT=0`: nothing here, every object placed again and
    /// every record written again.
    pub(super) fn shift_records(&mut self, renderer: &mut Renderer, world: &mut World, before: Option<(u32, u32)>) {
        let (Some(old), Some(new)) = (before, self.last_origin) else {
            return;
        };
        if !aurora_render::origin_shift::enabled() {
            log::info!("render origin moved: every record written again (AURORA_ORIGIN_SHIFT=0)");
            return;
        }
        let started = Instant::now();
        let delta = [
            (f64::from(old.0) - f64::from(new.0)) as f32,
            (f64::from(old.1) - f64::from(new.1)) as f32,
        ];
        let d = Vec3::new(delta[0], delta[1], 0.0);
        // one walk: the faces to move, and what a placement would have moved
        // beside them (bounds, object table entry, pick entry)
        let mut ids = Vec::new();
        let mut followed = 0usize;
        for (idx, g) in self.gpu.iter_mut().enumerate() {
            if !g.follows_origin() {
                continue;
            }
            ids.extend(g.faces.iter().map(|f| f.record));
            g.center += d;
            renderer.cull.objects.set(idx, g.cull_object(idx));
            self.pick.set(idx, hover::PickEntry::of(g));
            followed += 1;
        }
        let (moved, refused) = renderer.records.shift_world(ids, delta);
        // (a face not placed by its model: its object placed again, as before)
        let mut placed = 0usize;
        for id in refused {
            let Some(idx) = renderer.records.faces.get(id as usize).map(|f| f.object as usize) else {
                continue;
            };
            if let Some(o) = world.objects.get_mut_untracked(idx) {
                placed += usize::from(!std::mem::replace(&mut o.render.needs_records, true));
                self.sets.dirty.insert(idx);
            }
        }
        self.sets.lod.shift(d);
        log::info!(
            "render origin moved by ({:.0}, {:.0}) m: {followed} objects and {moved} records followed on the GPU, {placed} placed again ({:.2} ms)",
            delta[0],
            delta[1],
            started.elapsed().as_secs_f64() * 1000.0
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_what_rests_in_the_world_follows_the_origin() {
        let g = ObjGpu::default();
        assert!(!g.follows_origin(), "nothing drawn: placed again");
        assert!(g.rests_in_world(), "a prim in the world");
        for change in [
            |g: &mut ObjGpu| g.is_avatar = true,
            |g: &mut ObjGpu| g.hud = true,
            |g: &mut ObjGpu| g.rigged = true,
            |g: &mut ObjGpu| g.owner_avatar = Some(3),
        ] {
            let mut h = ObjGpu::default();
            change(&mut h);
            assert!(!h.rests_in_world());
        }
    }
}
