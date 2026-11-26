// aurora-local-patch: scene-impostors-viewer-module
//! Scene impostors (Preferences › Graphics › Quality › Distance and detail;
//! local patch set): the rule that turns them on, the stamps of the GPU
//! object table, and the parameters each frame gives the renderer, which
//! draws the far field from a panorama (aurora-render/src/scene_panorama.rs).
//!
//! The stamps: each entry of the GPU object table carries the stamp of the
//! frame it last changed in (`CullObject::stamp`, written by
//! `ObjGpu::cull_object`; the table keeps the old stamp of an entry written
//! again unchanged, `scene_panorama::restamp`). A capture of the panorama
//! takes only the objects older than its first frame, and the panorama only
//! stands in for those: what moves, or arrived or changed since, is drawn
//! as usual until a later capture takes it. An object whose sync only
//! swapped its level of detail keeps its stamp ([`other_detail`],
//! `scene_panorama::detail_stamp`). While the switch is off every stamp
//! written is 0 and the table is written as without the patch.

use crate::scene::GeomKey;
use aurora_render::scene_panorama::PanoramaParams;
use std::sync::atomic::{AtomicU32, Ordering};

/// Whether the far field goes to the panorama: the switch on, and an
/// impostor distance short of the draw distance. Otherwise nothing would be
/// far enough, and the frame is drawn exactly as with the switch off.
pub fn active(enabled: bool, distance: f32, draw_distance: f32) -> bool {
    enabled && distance.is_finite() && distance > 0.0 && distance < draw_distance
}

/// Stamps given out so far (never 0 once on).
static COUNTER: AtomicU32 = AtomicU32::new(0);
/// The stamp of the frame being prepared: 0 while the impostors are off.
static STAMP: AtomicU32 = AtomicU32::new(0);

/// The stamp an entry of the GPU object table written now carries.
pub fn stamp() -> u32 {
    STAMP.load(Ordering::Relaxed)
}

/// The renderer's parameters of this frame, and the stamp of the entries
/// the next sync writes. `in_world`: the world screen (the renderer leaves
/// the photos, the headset and the camera under water out itself, and
/// keeps its panorama for after them: the stamps go on meanwhile).
/// `main_region`: the agent's region handle; `origin`: the render origin's
/// global position (`World::main_origin`).
pub fn frame(
    enabled: bool,
    distance: f32,
    draw_distance: f32,
    main_region: Option<u64>,
    origin: Option<(u32, u32)>,
    in_world: bool,
) -> PanoramaParams {
    let on = active(enabled, distance, draw_distance);
    let stamp = if on {
        COUNTER.fetch_add(1, Ordering::Relaxed).wrapping_add(1).max(1)
    } else {
        0
    };
    STAMP.store(stamp, Ordering::Relaxed);
    PanoramaParams {
        active: on && in_world,
        distance,
        draw_distance,
        epoch: main_region.map_or(0, |h| (h ^ (h >> 32)) as u32),
        origin: origin.map_or([0.0; 2], |(x, y)| [f64::from(x), f64::from(y)]),
        stamp,
    }
}

/// Whether the geometry `new` bound by a sync is another level of detail of
/// the shape `old` (`ObjGpu::lod_swap`): the same prim, sculpt, mesh or
/// tree. A first geometry, another shape, or a sculpt whose map arrived
/// is a change.
pub fn other_detail(old: Option<GeomKey>, new: Option<GeomKey>) -> bool {
    let (Some(old), Some(new)) = (old, new) else {
        return false;
    };
    match (old, new) {
        (GeomKey::Prim { hash: a, lod: x }, GeomKey::Prim { hash: b, lod: y })
        | (GeomKey::Sculpt { hash: a, lod: x }, GeomKey::Sculpt { hash: b, lod: y })
        | (GeomKey::SculptPlaceholder { hash: a, lod: x }, GeomKey::SculptPlaceholder { hash: b, lod: y }) => a == b && x != y,
        (GeomKey::Flexible { id: a, hash: h, lod: x }, GeomKey::Flexible { id: b, hash: k, lod: y }) => a == b && h == k && x != y,
        (GeomKey::Mesh { id: a, lod: x }, GeomKey::Mesh { id: b, lod: y }) => a == b && x != y,
        (GeomKey::Tree { species: a, lod: x }, GeomKey::Tree { species: b, lod: y }) => a == b && x != y,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn on_only_with_the_switch_and_a_distance_short_of_the_draw_distance() {
        assert!(active(true, 256.0, 1024.0));
        assert!(active(true, 64.0, 96.0));
        assert!(!active(false, 256.0, 1024.0));
        // as far as the draw distance or farther: nothing would be far enough
        assert!(!active(true, 256.0, 256.0));
        assert!(!active(true, 512.0, 256.0));
        assert!(!active(true, f32::NAN, 1024.0));
        assert!(!active(true, 0.0, 1024.0));
    }

    #[test]
    fn another_detail_of_the_same_shape_only() {
        let prim = |hash, lod| Some(GeomKey::Prim { hash, lod });
        assert!(other_detail(prim(7, 1), prim(7, 3)));
        assert!(!other_detail(prim(7, 1), prim(7, 1)));
        assert!(!other_detail(prim(7, 1), prim(8, 3)), "another shape");
        assert!(!other_detail(None, prim(7, 3)), "a first geometry");
        assert!(!other_detail(prim(7, 1), None));
        let id = uuid::Uuid::from_u128(5);
        assert!(other_detail(Some(GeomKey::Mesh { id, lod: 0 }), Some(GeomKey::Mesh { id, lod: 2 })));
        assert!(!other_detail(
            Some(GeomKey::Mesh { id, lod: 0 }),
            Some(GeomKey::Mesh {
                id: uuid::Uuid::from_u128(6),
                lod: 2
            })
        ));
        // a sculpt's map arriving changes its shape
        assert!(!other_detail(
            Some(GeomKey::SculptPlaceholder { hash: 7, lod: 1 }),
            Some(GeomKey::Sculpt { hash: 7, lod: 1 })
        ));
    }

    #[test]
    fn the_stamps_go_up_while_on_and_are_zero_off() {
        let a = frame(true, 256.0, 1024.0, Some((1 << 40) | 7), Some((256_000, 512_000)), true);
        assert!(a.active && a.stamp > 0 && stamp() == a.stamp);
        assert_eq!(a.origin, [256_000.0, 512_000.0]);
        let b = frame(true, 256.0, 1024.0, Some((1 << 40) | 7), None, true);
        assert!(b.stamp > a.stamp);
        assert_eq!(a.epoch, b.epoch);
        // the region is in the epoch
        assert_ne!(frame(true, 256.0, 1024.0, Some((2 << 40) | 7), None, true).epoch, a.epoch);
        // another screen: nothing drawn from a panorama, the stamps go on
        let c = frame(true, 256.0, 1024.0, None, None, false);
        assert!(!c.active && c.stamp > b.stamp && stamp() == c.stamp);
        // the switch off, or nothing far enough: 0
        for p in [
            frame(false, 256.0, 1024.0, None, None, true),
            frame(true, 1024.0, 1024.0, None, None, true),
        ] {
            assert!(!p.active);
            assert_eq!((p.stamp, stamp()), (0, 0));
        }
    }
}
