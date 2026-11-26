// aurora-local-patch: scene-impostors-demo-module
//! AURORA_DEMO_SCENE_IMPOSTORS=on|off[:walk] (local patch set): a far field
//! for the scene impostors. Twenty-four neighbour regions of rolling land
//! around the demo region, with hills among them, and about 2,400 boxes and
//! towers of all sizes and colours from 60 m out to about 1,000 m; the draw
//! distance at 1024 m, the camera held toward the north-east across them
//! until frame 700. `on` turns the impostors on from 120 m once the far
//! field is in, `off` draws the same scene as usual: the two captures of a
//! frame must look the same, the far boxes on their hills and the near
//! hills in front of the far boxes. `walk`: from frame 700 the camera goes
//! 5 cm a frame toward the far field (3 m a second at 60 fps: refreshes).

use super::HANDLE;
use crate::camera::Camera;
use crate::settings::Settings;
use crate::world::World;
use aurora_net::NetEvent;
use aurora_net::objects::ObjectUpdate;
use aurora_net::terrain::TerrainPatch;
use aurora_prim::params::{LL_PCODE_PATH_LINE, LL_PCODE_PROFILE_CIRCLE, LL_PCODE_PROFILE_SQUARE};
use glam::{Quat, Vec3};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use uuid::Uuid;

/// The scenario's value, when it runs.
pub fn mode() -> Option<&'static str> {
    static MODE: OnceLock<Option<String>> = OnceLock::new();
    MODE.get_or_init(|| std::env::var("AURORA_DEMO_SCENE_IMPOSTORS").ok()).as_deref()
}

/// The regions, in regions from the demo region on x and y.
const SPAN: std::ops::RangeInclusive<i32> = -1..=3;
/// Local ids of the boxes (far above the demo's own).
const FIRST_ID: u32 = 0x0700_0000;
/// Viewpoint, from the demo region's corner (m).
const EYE: Vec3 = Vec3::new(40.0, 30.0, 52.0);
const TARGET: Vec3 = Vec3::new(560.0, 520.0, 34.0);
const WATER: f32 = 20.0;
/// Frames of the world screen before the switch comes on (`on`): the far
/// field is synced by then, and the first panorama holds it.
const SWITCH_ON: u64 = 120;
/// The view is held until this frame (then the walk, or the user's camera).
const HOLD: u64 = 700;

/// A repeatable random number in [0, 1) for `seed` and `k`.
fn noise(seed: u32, k: u32) -> f32 {
    let mut h = u64::from(seed).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ u64::from(k).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 31;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 29;
    (h % 1_000_003) as f32 / 1_000_003.0
}

/// The ground at a place (m from the demo region's corner): the demo
/// region's own, with land and hills rising away from it (none on the
/// region itself, so its edges meet its neighbours).
fn ground(x: f32, y: f32) -> f32 {
    let dx = (-x).max(x - 256.0).max(0.0);
    let dy = (-y).max(y - 256.0).max(0.0);
    let out = (dx * dx + dy * dy).sqrt();
    let t = ((out - 16.0) / 80.0).clamp(0.0, 1.0);
    let fade = t * t * (3.0 - 2.0 * t);
    let hill = |cx: f32, cy: f32, h: f32, r: f32| h * (-((x - cx).powi(2) + (y - cy).powi(2)) / (2.0 * r * r)).exp();
    let rolling = 6.0 * (x / 83.0).sin() * (y / 71.0).cos() + 4.0 * ((x + y) / 131.0).sin();
    // one hill across the view at about 520 m, others farther and aside
    let hills =
        hill(420.0, 330.0, 46.0, 70.0) + hill(180.0, 640.0, 30.0, 90.0) + hill(760.0, 420.0, 38.0, 110.0) + hill(640.0, 860.0, 42.0, 95.0);
    super::height(x, y) + fade * (8.0 + rolling + hills)
}

/// The neighbours' handshakes and terrain.
fn regions(world: &mut World) -> usize {
    let Some(info) = world.regions.get(&HANDLE).and_then(|r| r.info.clone()) else {
        return 0;
    };
    let (ox, oy) = ((HANDLE >> 32) as i64, (HANDLE & 0xffff_ffff) as i64);
    let mut count = 0;
    for ix in SPAN {
        for iy in SPAN {
            if (ix, iy) == (0, 0) {
                continue;
            }
            let handle = (((ox + i64::from(ix) * 256) as u64) << 32) | (oy + i64::from(iy) * 256) as u64;
            let mut region = (*info).clone();
            region.handle = handle;
            region.is_main = false;
            region.name = format!("Aurora Demo Far {ix} {iy}");
            region.region_id = Uuid::from_u128(0x5ce0_0000_0000_0000_0000_0000_0000_0000 | (((ix + 8) as u128) << 8) | (iy + 8) as u128);
            world.apply(NetEvent::RegionHandshake(Arc::new(region)));
            let mut patches = Vec::with_capacity(256);
            for py in 0..16u32 {
                for px in 0..16u32 {
                    let mut heights = Vec::with_capacity(256);
                    for j in 0..16 {
                        for i in 0..16 {
                            let x = (ix * 256) as f32 + (px * 16 + i) as f32;
                            let y = (iy * 256) as f32 + (py * 16 + j) as f32;
                            heights.push(ground(x, y));
                        }
                    }
                    patches.push(TerrainPatch {
                        x: px,
                        y: py,
                        size: 16,
                        heights,
                    });
                }
            }
            world.apply(NetEvent::Terrain { handle, patches });
            count += 1;
        }
    }
    count
}

/// The boxes and towers, by region.
fn boxes() -> BTreeMap<(i32, i32), Vec<ObjectUpdate>> {
    let colours = [
        [0.78, 0.74, 0.68, 1.0],
        [0.62, 0.66, 0.72, 1.0],
        [0.72, 0.52, 0.42, 1.0],
        [0.45, 0.52, 0.60, 1.0],
        [0.86, 0.84, 0.80, 1.0],
        [0.55, 0.60, 0.48, 1.0],
        [0.68, 0.40, 0.36, 1.0],
        [0.40, 0.44, 0.52, 1.0],
    ];
    let square = super::shape(LL_PCODE_PATH_LINE, LL_PCODE_PROFILE_SQUARE, 100, 0, 0);
    let round = super::shape(LL_PCODE_PATH_LINE, LL_PCODE_PROFILE_CIRCLE, 100, 0, 0);
    let mut out: BTreeMap<(i32, i32), Vec<ObjectUpdate>> = BTreeMap::new();
    let mut id = FIRST_ID;
    let (lo, hi) = (*SPAN.start() * 256 + 16, (*SPAN.end() + 1) * 256 - 16);
    for gx in (lo..hi).step_by(25) {
        for gy in (lo..hi).step_by(25) {
            let k = id - FIRST_ID;
            let x = gx as f32 + (noise(k, 1) - 0.5) * 16.0;
            let y = gy as f32 + (noise(k, 2) - 0.5) * 16.0;
            id += 1;
            let here = Vec3::new(x, y, 0.0);
            // the demo's plaza and the camera's own surroundings stay clear
            if here.distance(Vec3::new(128.0, 128.0, 0.0)) < 80.0 || here.distance(EYE.with_z(0.0)) < 60.0 {
                continue;
            }
            let tower = noise(k, 3) < 0.08;
            let height = if tower {
                48.0 + 16.0 * noise(k, 4)
            } else {
                8.0 + 40.0 * noise(k, 4)
            };
            let scale = Vec3::new(6.0 + 18.0 * noise(k, 5), 6.0 + 18.0 * noise(k, 6), height);
            let base = ground(x, y).max(WATER);
            let mut c = colours[(noise(k, 7) * colours.len() as f32) as usize % colours.len()];
            let shade = 0.8 + 0.3 * noise(k, 8);
            for v in &mut c[..3] {
                *v = (*v * shade).min(1.0);
            }
            let (ix, iy) = (x.div_euclid(256.0) as i32, y.div_euclid(256.0) as i32);
            let local = Vec3::new(x - ix as f32 * 256.0, y - iy as f32 * 256.0, base + height * 0.5 - 1.0);
            let volume = if noise(k, 9) < 0.2 { round } else { square };
            out.entry((ix, iy)).or_default().push(super::prim(
                id,
                local,
                Quat::from_rotation_z(noise(k, 10) * std::f32::consts::TAU),
                scale,
                volume,
                super::te(c, 0, false, 0.0),
                Default::default(),
                "",
            ));
        }
    }
    out
}

/// Each frame of the world screen in the demo: returns whether the
/// settings changed. The steps count from its first frame (the loading
/// screen before it lasts a second and a half or more).
pub fn update(camera: &mut Camera, settings: &mut Settings, world: &mut World, panels: &mut crate::ui::Panels, frame: u64) -> bool {
    let Some(mode) = mode() else {
        return false;
    };
    static FIRST: AtomicU64 = AtomicU64::new(u64::MAX);
    let first = match FIRST.compare_exchange(u64::MAX, frame, Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => frame,
        Err(first) => first,
    };
    let k = frame.saturating_sub(first);
    let mut changed = false;
    if k == 0 {
        changed = true;
        panels.chat = false;
        panels.people = false;
        panels.perf = false;
        panels.minimap = false;
        panels.inventory = false;
        panels.appearance = false;
        panels.ao = false;
        settings.draw_distance = 1024.0;
        settings.draw_distance_stepping = false;
        settings.draw_stepping = None;
        settings.scene_impostors = false;
        settings.impostor_distance = 120.0;
        log::info!("scene impostors demo: world from frame {frame}, draw distance 1024 m, impostor distance 120 m");
    }
    if k == 2 {
        let n = regions(world);
        log::info!("scene impostors demo: {n} neighbour regions");
    }
    if k == 8 {
        let all = boxes();
        let count: usize = all.values().map(Vec::len).sum();
        let (ox, oy) = ((HANDLE >> 32) as i64, (HANDLE & 0xffff_ffff) as i64);
        for ((ix, iy), objects) in all {
            let handle = (((ox + i64::from(ix) * 256) as u64) << 32) | (oy + i64::from(iy) * 256) as u64;
            world.apply(NetEvent::ObjectUpdates { handle, objects });
        }
        log::info!("scene impostors demo: {count} boxes");
    }
    if k == SWITCH_ON && !mode.starts_with("off") {
        changed = true;
        settings.scene_impostors = true;
        log::info!("scene impostors demo: impostors on at frame {frame}");
    }
    // the view, held over the loading camera, and the walk
    let walk = mode.contains("walk") && frame > HOLD;
    if camera.free().is_none() && (frame <= HOLD || walk) {
        let along = (TARGET - EYE).with_z(0.0).normalize_or(Vec3::X) * (frame.saturating_sub(HOLD).min(4000) as f32 * 0.05);
        let (eye, target) = (EYE + along, TARGET + along);
        camera.set_view(eye, target, &mut world.agent, &settings.camera);
        camera.position = eye;
        camera.target = target;
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_far_field_reaches_a_kilometre_and_spares_the_plaza() {
        let all = boxes();
        let count: usize = all.values().map(Vec::len).sum();
        assert!((2000..=4000).contains(&count), "{count} boxes");
        let mut farthest = 0.0f32;
        for ((ix, iy), objects) in &all {
            assert!(SPAN.contains(ix) && SPAN.contains(iy));
            for o in objects {
                let global = Vec3::new(o.position.x + *ix as f32 * 256.0, o.position.y + *iy as f32 * 256.0, 0.0);
                assert!(global.distance(Vec3::new(128.0, 128.0, 0.0)) >= 80.0);
                assert!((0.0..256.0).contains(&o.position.x) && (0.0..256.0).contains(&o.position.y));
                farthest = farthest.max(global.distance(EYE.with_z(0.0)));
            }
        }
        assert!(farthest > 1000.0, "{farthest}");
        // the demo region keeps its own ground
        assert_eq!(ground(100.0, 200.0), super::super::height(100.0, 200.0));
        assert!(ground(420.0, 330.0) > ground(300.0, 200.0) + 20.0, "a hill across the view");
    }
}
