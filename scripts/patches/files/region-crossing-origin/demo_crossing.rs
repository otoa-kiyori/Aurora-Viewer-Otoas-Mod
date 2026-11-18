// aurora-local-patch: region-crossing-origin-demo-module
//! AURORA_DEMO_CROSSING=<prims>[,<frames>[,<first>]] (local patch set):
//! region crossings in a full scene. Eleven neighbour regions around the
//! demo region (one west, two east, a row north and south) and `prims`
//! boxes over the twelve (64,000 by default: about 384,000 face records,
//! 117 MB, as a busy ride across regions), a quarter of them two-prim linksets; the
//! draw distance at 400 m and the camera held across the east border. From
//! frame `first` (1,200) the main region goes from the demo region to its
//! east neighbour and back every `frames` frames (180), as an agent walking
//! to and fro over the border: `MainRegionChanged` then
//! `AgentMovementComplete`, the render origin moving 256 m each time while
//! nothing moves in the world. Read the `frame writes took` lines and, with
//! `AURORA_PROFILE=1`, the `max:` of the seconds with an `arrival:` line;
//! `AURORA_ORIGIN_SHIFT=0` writes every record again, as before. A capture
//! before a crossing and one after it show the same view.

use super::HANDLE;
use crate::camera::Camera;
use crate::settings::Settings;
use crate::world::World;
use aurora_net::NetEvent;
use aurora_net::objects::ObjectUpdate;
use aurora_prim::params::{LL_PCODE_PATH_LINE, LL_PCODE_PROFILE_SQUARE};
use glam::{Quat, Vec3};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use uuid::Uuid;

/// The scenario's parameters: prims, frames between crossings, first one.
pub fn mode() -> Option<(u32, u64, u64)> {
    static MODE: OnceLock<Option<(u32, u64, u64)>> = OnceLock::new();
    *MODE.get_or_init(|| std::env::var("AURORA_DEMO_CROSSING").ok().map(|v| parse(&v)))
}

fn parse(v: &str) -> (u32, u64, u64) {
    let mut it = v.split(',').map(str::trim);
    let prims = it.next().and_then(|s| s.parse().ok()).unwrap_or(64_000).clamp(1, 400_000);
    let frames = it.next().and_then(|s| s.parse().ok()).unwrap_or(180u64).max(2);
    let first = it.next().and_then(|s| s.parse().ok()).unwrap_or(1200u64);
    (prims, frames, first)
}

/// The regions, in regions from the demo region: x then y.
const SPAN_X: std::ops::RangeInclusive<i32> = -1..=2;
const SPAN_Y: std::ops::RangeInclusive<i32> = -1..=1;
/// Local ids of the boxes (far above the demo's own and the far field's).
const FIRST_ID: u32 = 0x0800_0000;
/// The camera, from the demo region's corner (m): near the east border,
/// looking across it.
const EYE: Vec3 = Vec3::new(214.0, 70.0, 44.0);
const TARGET: Vec3 = Vec3::new(330.0, 150.0, 22.0);

/// The demo region's east neighbour, where the crossings go.
fn east() -> u64 {
    HANDLE + (256u64 << 32)
}

fn handle_of(ix: i32, iy: i32) -> u64 {
    let (ox, oy) = ((HANDLE >> 32) as i64, (HANDLE & 0xffff_ffff) as i64);
    (((ox + i64::from(ix) * 256) as u64) << 32) | (oy + i64::from(iy) * 256) as u64
}

/// A repeatable random number in [0, 1) for `seed` and `k`.
fn noise(seed: u32, k: u32) -> f32 {
    let mut h = u64::from(seed).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ u64::from(k).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 31;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 29;
    (h % 1_000_003) as f32 / 1_000_003.0
}

/// The ground (m from the demo region's corner): the demo region's own,
/// gently rolling land around it (none on the region itself, so its edges
/// meet its neighbours).
fn ground(x: f32, y: f32) -> f32 {
    let dx = (-x).max(x - 256.0).max(0.0);
    let dy = (-y).max(y - 256.0).max(0.0);
    let out = (dx * dx + dy * dy).sqrt();
    let t = ((out - 8.0) / 60.0).clamp(0.0, 1.0);
    let fade = t * t * (3.0 - 2.0 * t);
    let rolling = 3.0 * (x / 61.0).sin() * (y / 53.0).cos() + 2.0 * ((x + y) / 97.0).sin();
    super::height(x, y) + fade * (4.0 + rolling)
}

/// The neighbours' handshakes and terrain.
fn regions(world: &mut World) -> usize {
    let Some(info) = world.regions.get(&HANDLE).and_then(|r| r.info.clone()) else {
        return 0;
    };
    let mut count = 0;
    for ix in SPAN_X {
        for iy in SPAN_Y {
            if (ix, iy) == (0, 0) {
                continue;
            }
            let handle = handle_of(ix, iy);
            let mut region = (*info).clone();
            region.handle = handle;
            region.is_main = false;
            region.name = format!("Aurora Demo Crossing {ix} {iy}");
            region.region_id = Uuid::from_u128(0xc205_0000_0000_0000_0000_0000_0000_0000 | (((ix + 8) as u128) << 8) | (iy + 8) as u128);
            world.apply(NetEvent::RegionHandshake(Arc::new(region)));
            let mut patches = Vec::with_capacity(256);
            for py in 0..16u32 {
                for px in 0..16u32 {
                    let mut heights = Vec::with_capacity(256);
                    for j in 0..16 {
                        for i in 0..16 {
                            heights.push(ground(
                                (ix * 256) as f32 + (px * 16 + i) as f32,
                                (iy * 256) as f32 + (py * 16 + j) as f32,
                            ));
                        }
                    }
                    patches.push(aurora_net::terrain::TerrainPatch {
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

/// The boxes, by region: `prims` of them on a jittered grid over the
/// twelve regions (the plaza and the camera's surroundings left clear),
/// one in four a root with a child box on top.
fn boxes(prims: u32) -> Vec<((i32, i32), Vec<ObjectUpdate>)> {
    let colours = [
        [0.78, 0.74, 0.68, 1.0],
        [0.62, 0.66, 0.72, 1.0],
        [0.72, 0.52, 0.42, 1.0],
        [0.45, 0.52, 0.60, 1.0],
        [0.86, 0.84, 0.80, 1.0],
        [0.55, 0.60, 0.48, 1.0],
    ];
    let square = super::shape(LL_PCODE_PATH_LINE, LL_PCODE_PROFILE_SQUARE, 100, 0, 0);
    let (x0, x1) = (*SPAN_X.start() as f32 * 256.0, (*SPAN_X.end() + 1) as f32 * 256.0);
    let (y0, y1) = (*SPAN_Y.start() as f32 * 256.0, (*SPAN_Y.end() + 1) as f32 * 256.0);
    // a linkset counts its two prims; a few more sites than needed (the
    // plaza and the camera's surroundings are left out)
    let sites = (prims as f32 / 1.25).ceil().max(1.0);
    let step = ((x1 - x0) * (y1 - y0) / sites).sqrt() * 0.97;
    let mut out: std::collections::BTreeMap<(i32, i32), Vec<ObjectUpdate>> = Default::default();
    let (mut id, mut made, mut k) = (FIRST_ID, 0u32, 0u32);
    let mut y = y0 + step * 0.5;
    'rows: while y < y1 {
        let mut x = x0 + step * 0.5;
        while x < x1 {
            if made >= prims {
                break 'rows;
            }
            k += 1;
            let (px, py) = (x + (noise(k, 1) - 0.5) * step * 0.8, y + (noise(k, 2) - 0.5) * step * 0.8);
            x += step;
            let here = Vec3::new(px, py, 0.0);
            if here.distance(Vec3::new(128.0, 128.0, 0.0)) < 40.0 || here.distance(EYE.with_z(0.0)) < 12.0 {
                continue;
            }
            let (ix, iy) = (px.div_euclid(256.0) as i32, py.div_euclid(256.0) as i32);
            if !SPAN_X.contains(&ix) || !SPAN_Y.contains(&iy) {
                continue;
            }
            let size = 0.8 + 2.4 * noise(k, 3);
            let height = 0.6 + 4.0 * noise(k, 4);
            let scale = Vec3::new(size, size * (0.6 + 0.8 * noise(k, 5)), height);
            let local = Vec3::new(px - ix as f32 * 256.0, py - iy as f32 * 256.0, ground(px, py) + height * 0.5 - 0.1);
            let c = colours[(noise(k, 6) * colours.len() as f32) as usize % colours.len()];
            let rot = Quat::from_rotation_z(noise(k, 7) * std::f32::consts::TAU);
            let list = out.entry((ix, iy)).or_default();
            let root = id;
            list.push(super::prim(
                id,
                local,
                rot,
                scale,
                square,
                super::te(c, 0, false, 0.0),
                Default::default(),
                "",
            ));
            id += 1;
            made += 1;
            if k.is_multiple_of(4) && made < prims {
                // a child box on top, placed through its root's matrix
                let mut child = super::prim(
                    id,
                    Vec3::new(0.0, 0.0, height * 0.5 + 0.3),
                    Quat::from_rotation_z(0.6),
                    scale * Vec3::new(0.5, 0.5, 0.3),
                    square,
                    super::te([0.40, 0.44, 0.52, 1.0], 0, false, 0.0),
                    Default::default(),
                    "",
                );
                child.parent_id = root;
                list.push(child);
                id += 1;
                made += 1;
            }
        }
        y += step;
    }
    out.into_iter().collect()
}

/// The crossings, to and fro over the east border, each frame of the demo
/// before the camera follows the origin (as the network's).
pub fn cross(world: &mut World, frame: u64) {
    let Some((_, every, first_crossing)) = mode() else {
        return;
    };
    if frame < first_crossing || !(frame - first_crossing).is_multiple_of(every) {
        return;
    }
    let n = (frame - first_crossing) / every;
    let handle = if n.is_multiple_of(2) { east() } else { HANDLE };
    world.apply(NetEvent::MainRegionChanged { handle });
    let (position, look_at) = (world.agent.position, world.agent.forward());
    world.apply(NetEvent::AgentMovementComplete { handle, position, look_at });
    log::info!(
        "crossing demo: crossing {} at frame {frame}, into the {}",
        n + 1,
        if handle == HANDLE { "demo region" } else { "east neighbour" }
    );
}

/// Each frame of the world screen in the demo: returns whether the
/// settings changed. The steps count from its first frame.
pub fn update(camera: &mut Camera, settings: &mut Settings, world: &mut World, panels: &mut crate::ui::Panels, frame: u64) -> bool {
    let Some((prims, _, _)) = mode() else {
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
        settings.draw_distance = 400.0;
        settings.draw_distance_stepping = false;
        settings.draw_stepping = None;
        log::info!("crossing demo: world from frame {frame}, draw distance 400 m");
    }
    if k == 2 {
        let n = regions(world);
        log::info!("crossing demo: {n} neighbour regions");
    }
    if k == 8 {
        let all = boxes(prims);
        let count: usize = all.iter().map(|(_, v)| v.len()).sum();
        for ((ix, iy), objects) in all {
            world.apply(NetEvent::ObjectUpdates {
                handle: handle_of(ix, iy),
                objects,
            });
        }
        log::info!("crossing demo: {count} prims");
    }
    // the view, held at the same place in the world whatever the origin
    if camera.free().is_none()
        && let Some(offset) = world.region_offset(HANDLE)
    {
        let (eye, target) = (EYE + offset, TARGET + offset);
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
    fn the_scene_has_the_prims_asked_for_over_twelve_regions() {
        let all = boxes(20_000);
        let count: usize = all.iter().map(|(_, v)| v.len()).sum();
        assert_eq!(count, 20_000);
        assert_eq!(all.len(), 12, "every region holds boxes");
        let children = all.iter().flat_map(|(_, v)| v).filter(|o| o.parent_id != 0).count();
        assert!((3_000..=5_000).contains(&children), "{children} linksets");
        for ((ix, iy), objects) in &all {
            for o in objects.iter().filter(|o| o.parent_id == 0) {
                assert!((0.0..256.0).contains(&o.position.x) && (0.0..256.0).contains(&o.position.y));
                let global = Vec3::new(o.position.x + *ix as f32 * 256.0, o.position.y + *iy as f32 * 256.0, 0.0);
                assert!(global.distance(Vec3::new(128.0, 128.0, 0.0)) >= 40.0, "the plaza stays clear");
            }
        }
    }

    #[test]
    fn the_parameters_default_and_clamp() {
        assert_eq!(parse(""), (64_000, 180, 1200));
        assert_eq!(parse("1000,60,900"), (1000, 60, 900));
        assert_eq!(parse("0,1"), (1, 2, 1200));
    }

    #[test]
    fn the_crossings_go_east_and_back() {
        assert_eq!(east() >> 32, (HANDLE >> 32) + 256);
        assert_eq!(east() & 0xffff_ffff, HANDLE & 0xffff_ffff);
        assert_eq!(handle_of(1, 0), east());
        assert_eq!(handle_of(0, 0), HANDLE);
    }
}
