//! Open-sea swell and calm water (local patch set,
//! scripts/patches/109-ocean-swell.toml): long rolling waves on open water
//! and calmer water along the shore and the static objects in it, a
//! viewer-local look. Nothing is sent and the environment asset is
//! untouched; the Legacy Viewer has no such effect (its water is flat), so
//! nothing here is ported.
//!
//! - **Model.** Four long Gerstner waves of 54 to 131 m (`WAVES`), within
//!   about 25 degrees of the wind, slowly modulated into sets that travel at
//!   the group speed (`SET_*`). `s` is their height normalised to [-1, 1].
//!   The wind is the way the environment's big water wave (EEP wave1)
//!   scrolls, the only value the swell takes from the environment.
//!   "Ocean Swell Down" (`down`) and "Ocean Swell Up" (`up`) map it
//!   linearly: the surface moves by `open * ((up - down) / 2 + (up + down)
//!   / 2 * s)`, so the troughs reach `down` below the water's level and the
//!   crests `up` above it (up 0: the crests stay at the level, as before).
//!   The existing swell, the wakes and the micro waves ride on top.
//! - **Calm water** ("Calm distance" `reach`, "Calm at the edge" `edge`):
//!   at the line where the water meets land or a static object, every
//!   vertical movement of the surface (this swell, the existing geometric
//!   waves) is `edge` times its full height, rising smoothly to full at
//!   `reach` metres ([`calm`]); the horizontal motion is scaled the same
//!   way, so no crest folds. Edge 1 or reach 0: no calming.
//! - **Openness** (0..1) of the swell: the calm factor at the point, and
//!   never so deep that a trough comes within `KEEL` of the bed.
//! - **Distances.** Two maps around the camera, both on the thread pool: the
//!   coarse field (`FIELD_SIZE`² cells of `FIELD_CELL`, terrain only, while
//!   the swell is on: the far sea) and the fine map (`FINE_SIZE`² cells of
//!   `FINE_CELL`, terrain and the static objects' footprints at the water
//!   line, while the calming is on). Within the fine map its distance
//!   holds, blended into the coarse one over its outer `FINE_BLEND` metres.
//!   Both travel to the GPU in the rows of the bathymetry texture under its
//!   own 64 (`FIELD_ROW`, `FINE_ROW`): no new binding.
//! - **Environment changes** (patch 998, the viewer's world/eep_fade.rs):
//!   while the water cross-fades from one EEP water to another whose big
//!   wave points another way, the old swell and the new one are each
//!   evaluated in their own direction and mixed by the share the water
//!   normal maps take (`Ocean::next_wind`, `Ocean::share`, [`Long::mix`]):
//!   the old crests fade out where they are while the new ones fade in, none
//!   turns or slides. Every term of the swell (height, gradient, horizontal
//!   motion, fold) is linear in its waves, so the mix is the mixed surface's
//!   own: the displacement, the normals, the crest foam's compression, the
//!   CPU's inversion and everything that rides it follow one surface. The
//!   openness and the calming do not depend on the direction. Outside a
//!   change, or with the same direction on both sides, one swell is
//!   evaluated, as before.
//! - **GPU.** The same functions are `ocean_long`, `ocean_long_mix`,
//!   `ocean_bilinear`, `ocean_guard` and `ocean_calm` in
//!   water_spectrum.wgsl, read through the shared water uniform's five vec4
//!   after the caustics' extension (`UNIFORM_OFFSET`). Tests keep the
//!   constants and the results equal.
//! - **Down and Up 0, edge 1** (the defaults have edge 0): nothing here
//!   runs, every caller takes its original path, the uniform's tail stays
//!   zero.
// aurora-local-patch: ocean-swell-module

use super::{WaterEffects, WaterGround};
use glam::{UVec2, Vec2, Vec3};
use std::sync::Arc;

/// The sliders' ranges (m): how far the troughs go below the water's level
/// ("Ocean Swell Down", stored as `ocean_depth`) and how far the crests go
/// above it ("Ocean Swell Up").
pub const DEPTH_MAX: f32 = 10.0;
pub const DEPTH_RANGE: std::ops::RangeInclusive<f32> = 0.0..=DEPTH_MAX;
pub const UP_MAX: f32 = 10.0;
pub const UP_RANGE: std::ops::RangeInclusive<f32> = 0.0..=UP_MAX;
/// "Calm distance" (m) and "Calm at the edge" (share of the full height),
/// their ranges and defaults.
pub const CALM_REACH_MAX: f32 = 100.0;
pub const CALM_REACH_RANGE: std::ops::RangeInclusive<f32> = 0.0..=CALM_REACH_MAX;
pub const CALM_REACH_DEFAULT: f32 = 100.0;
pub const CALM_EDGE_RANGE: std::ops::RangeInclusive<f32> = 0.0..=1.0;
pub const CALM_EDGE_DEFAULT: f32 = 0.0;

/// A saved or typed "Ocean Swell Down" within its range (not a number: 0).
pub fn depth(value: f32) -> f32 {
    if value.is_finite() { value.clamp(0.0, DEPTH_MAX) } else { 0.0 }
}

/// A saved or typed "Ocean Swell Up" within its range (not a number: 0).
pub fn up(value: f32) -> f32 {
    if value.is_finite() { value.clamp(0.0, UP_MAX) } else { 0.0 }
}

/// A saved or typed "Calm distance" within its range (not a number: the
/// default).
pub fn calm_reach(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, CALM_REACH_MAX)
    } else {
        CALM_REACH_DEFAULT
    }
}

/// A saved or typed "Calm at the edge" within its range (not a number: the
/// default).
pub fn calm_edge(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        CALM_EDGE_DEFAULT
    }
}

/// The swell is on: enhanced water with its geometric waves, and a depth or
/// a height above 0.
pub fn swell_on(settings: WaterEffects) -> bool {
    settings.enabled && settings.waves && settings.ocean_depth + settings.ocean_up > 0.0
}

/// The calming is on: enhanced water with its geometric waves, a reach
/// above 0 and an edge below 1.
pub fn calm_on(settings: WaterEffects) -> bool {
    settings.enabled && settings.waves && settings.calm_reach > 0.0 && settings.calm_edge < 1.0
}

/// The long waves: wavelength (m), angle from the wind (rad), share of the
/// swell's height (the shares sum to 1, so |s| <= 1), phase (rad).
/// Incommensurate wavelengths: no visible repeat; the two longest also beat
/// into kilometre-long groups. OCEAN_WAVES in water_spectrum.wgsl.
pub const WAVES: [[f32; 4]; 4] = [
    [131.0, 0.0, 0.34, 0.0],
    [117.0, 0.21, 0.26, 2.1],
    [71.3, -0.31, 0.22, 4.4],
    [53.9, 0.42, 0.18, 1.3],
];
/// Horizontal share of the Gerstner motion: rounder troughs, sharper crests,
/// far from folding (at 10 m from crest to trough the surface compresses by
/// at most about 20 %).
pub const STEEPNESS: f32 = 0.6;
/// Sets: their length along the wind (m), their speed (m/s, the group speed
/// of the longest wave, half its phase speed) and how much lower the swell
/// is between two sets.
pub const SET_LENGTH: f32 = 860.0;
pub const SET_SPEED: f32 = 7.15;
pub const SET_DEPTH: f32 = 0.25;
/// The water kept between a trough and the bed (m).
pub const KEEL: f32 = 1.0;
/// The swell's height the troughs' shade and the crests' light are scaled
/// by (m): the look of the earlier 0 to 5 m slider.
pub const SHADE_DEPTH: f32 = 5.0;
const GRAVITY: f32 = 9.81;
const TAU: f32 = std::f32::consts::TAU;

/// The coarse land-distance field: cells a side, their size (m), first row
/// in the bathymetry texture (the bathymetry keeps its 64 x 64 texels).
pub const FIELD_SIZE: u32 = 128;
pub const FIELD_CELL: f32 = 16.0;
pub const FIELD_ROW: u32 = super::WATER_GROUND_SIZE;
/// The fine calm-distance map: cells a side, their size (m), first row in
/// the bathymetry texture, and the width of its border (m) over which it
/// hands over to the coarse field.
pub const FINE_SIZE: u32 = 512;
pub const FINE_CELL: f32 = 1.0;
pub const FINE_ROW: u32 = FIELD_ROW + FIELD_SIZE;
pub const FINE_BLEND: f32 = 32.0;
/// The bathymetry texture with both maps under it.
pub const GROUND_WIDTH: u32 = FINE_SIZE;
pub const GROUND_HEIGHT: u32 = FINE_ROW + FINE_SIZE;
/// Terrain this close under the water (m) counts as land.
pub const LAND_MARGIN: f32 = 0.3;
/// Depth given to water with no region under it (m), and the distance of a
/// cell when its map holds no land at all (m).
pub const VOID_DEPTH: f32 = 1000.0;
pub const NO_LAND: f32 = 10_000.0;

/// Distances from the camera (m): the displaced swell fades to its mean
/// level between `FAR_FROM` and `FAR_TO` (the far water is drawn coarser);
/// the viewer draws 64 m patches out to `SUBDIVIDE`, on an 8 x 8 grid
/// (`FAR_GRID_CELLS`) out to `FAR_GRID`, and the normals keep the swell
/// out to `NORMAL_TO` (env.wgsl / common.wgsl constants of the same names).
pub const FAR_FROM: f32 = 600.0;
pub const FAR_TO: f32 = 1000.0;
pub const NORMAL_TO: f32 = 4000.0;
pub const SUBDIVIDE: f32 = 1100.0;
pub const FAR_GRID: f32 = 768.0;
pub const FAR_GRID_CELLS: usize = 8;

/// Where the ocean's five vec4 start in the shared water uniform: after its
/// own 64 bytes and the caustics' 80 (water_caustic_texture.rs).
pub(super) const UNIFORM_OFFSET: u64 = 144;
/// Their size (bytes).
pub(super) const TAIL_BYTES: u64 = 80;
/// down + up, wind xy, far openness; field origin xy, cell, valid; down,
/// up, calm reach, calm edge; fine origin xy, cell, valid; while the
/// environment cross-fades two swell directions, the new one xy and its
/// share (0: one swell), 0.
type Tail = [f32; 20];

/// The shared water uniform's contents at creation, with the ocean's tail
/// (zero: off).
pub(super) fn initial_uniform(mut bytes: Vec<u8>) -> Vec<u8> {
    bytes.extend_from_slice(bytemuck::bytes_of(&[0.0f32; 20]));
    bytes
}

/// The shared water uniform: `contents` (its own 64 bytes, then the
/// caustics' 80 of patch 91, which this patch needs) and the ocean's tail.
pub(super) fn uniform_buffer(device: &wgpu::Device, label: &str, contents: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    debug_assert_eq!(contents.len() as u64, UNIFORM_OFFSET);
    crate::gpu_alloc::buffer_init(device, label, &initial_uniform(contents.to_vec()), usage)
}

/// The direction the swell travels: the way the environment's big water
/// wave scrolls (EEP wave1 moves the normal map's features along
/// -wave1), or a fixed one when the environment gives none.
pub fn wind(wave1: Vec2) -> Vec2 {
    let toward = -wave1;
    if toward.is_finite() && toward.length() > 1e-3 {
        toward.normalize()
    } else {
        Vec2::new(-0.928_477_7, 0.371_391_1)
    }
}

/// The long swell at `p` (material position, m) and `time` (s).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Long {
    /// Height in [-1, 1].
    pub s: f32,
    /// Its gradient (per m).
    pub ds: Vec2,
    /// Horizontal motion per metre of amplitude.
    pub shift: Vec2,
    /// xx, xy, yy of the horizontal Jacobian's decrease per metre of
    /// amplitude (J = I - amplitude * fold).
    pub fold: Vec3,
}

impl Long {
    /// `ocean_long_mix` of water_spectrum.wgsl: this swell and `next` mixed
    /// by `share` (0: this one). Each term is linear in the waves, so the
    /// result is the mixed surface's own.
    pub fn mix(self, next: Long, share: f32) -> Long {
        let keep = 1.0 - share;
        Long {
            s: self.s * keep + next.s * share,
            ds: self.ds * keep + next.ds * share,
            shift: self.shift * keep + next.shift * share,
            fold: self.fold * keep + next.fold * share,
        }
    }
}

/// Two directions closer than this are the same swell (at 1 km the crests
/// would move by a millimetre).
const SAME_DIRECTION: f32 = 1e-6;

/// `ocean_long` of water_spectrum.wgsl.
pub fn long_swell(p: Vec2, time: f32, wind: Vec2) -> Long {
    let ke = TAU / SET_LENGTH;
    let psi = ke * (p.dot(wind) - SET_SPEED * time);
    let env = 1.0 - SET_DEPTH * (0.5 + 0.5 * psi.sin());
    let denv = wind * (-SET_DEPTH * 0.5 * psi.cos() * ke);
    let mut sum = 0.0f32;
    let mut dsum = Vec2::ZERO;
    let mut shift = Vec2::ZERO;
    let mut fold = Vec3::ZERO;
    for [length, angle, share, phase0] in WAVES {
        let (sa, ca) = angle.sin_cos();
        let d = Vec2::new(wind.x * ca - wind.y * sa, wind.x * sa + wind.y * ca);
        let k = TAU / length;
        let phase = k * p.dot(d) - (GRAVITY * k).sqrt() * time + phase0;
        let (s, c) = phase.sin_cos();
        sum += share * s;
        dsum += d * (share * k * c);
        shift += d * (share * STEEPNESS * c);
        fold += Vec3::new(d.x * d.x, d.x * d.y, d.y * d.y) * (share * STEEPNESS * k * s);
    }
    Long {
        s: env * sum,
        ds: dsum * env + denv * sum,
        shift: shift * env,
        fold: fold * env,
    }
}

/// `ocean_bilinear` of water_spectrum.wgsl: the value and the gradient of
/// four texels `a b / c d` (row by row) at bilinear weights `f`, `cell` the
/// texel size (m).
pub fn bilinear(a: f32, b: f32, c: f32, d: f32, f: Vec2, cell: f32) -> (f32, Vec2) {
    let lerp = |x: f32, y: f32, t: f32| x + (y - x) * t;
    let v = lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
    let gx = lerp(b - a, d - c, f.y) / cell;
    let gy = (lerp(c, d, f.x) - lerp(a, b, f.x)) / cell;
    (v, Vec2::new(gx, gy))
}

/// `ocean_guard` of water_spectrum.wgsl: how much of the swell's trough
/// the water depth under a point allows (troughs never come within `KEEL`
/// of the bed), and its gradient, from four texels of water depth (m);
/// `down` the slider (m).
pub fn guard(a: f32, b: f32, c: f32, d: f32, f: Vec2, cell: f32, down: f32) -> (f32, Vec2) {
    let (v, g) = bilinear(a, b, c, d, f, cell);
    let raw = (v - KEEL) / down.max(0.001);
    let dguard = if raw > 0.0 && raw < 1.0 { 1.0 / down.max(0.001) } else { 0.0 };
    (raw.clamp(0.0, 1.0), g * dguard)
}

/// `ocean_calm` of water_spectrum.wgsl: the share of the full vertical
/// movement at `distance` (m) from the nearest land or static object, and
/// its derivative: `edge` at the contact line, rising smoothly to 1 at
/// `reach`, 1 beyond it. Reach 0: always 1.
pub fn calm(distance: f32, reach: f32, edge: f32) -> (f32, f32) {
    if reach <= 0.0 {
        return (1.0, 0.0);
    }
    let x = (distance / reach).clamp(0.0, 1.0);
    let s = x * x * (3.0 - 2.0 * x);
    let ds = if x > 0.0 && x < 1.0 { 6.0 * x * (1.0 - x) / reach } else { 0.0 };
    (edge + (1.0 - edge) * s, (1.0 - edge) * ds)
}

/// What the viewer samples for one cell of the coarse field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Cell {
    /// Terrain at or above the water, or a region without its terrain.
    Land,
    /// Region water this deep (m).
    Water(f32),
    /// Water with no region under it: open sea.
    Void,
}

/// Each cell's distance (m) to the nearest land cell of an `n` x `n` grid
/// of `cell` metres, row by row: to that cell's centre less half a cell, so
/// the water next to the shore is about half a cell from it; `NO_LAND`
/// when there is none. A Euclidean distance transform (Felzenszwalb and
/// Huttenlocher), rows then columns, in f64. None when `land` is not `n`².
pub fn distance_field(n: usize, cell: f32, land: &[bool]) -> Option<Vec<f32>> {
    if land.len() != n * n || n == 0 {
        return None;
    }
    const NONE: f64 = 1e12;
    let mut squared: Vec<f64> = land.iter().map(|l| if *l { 0.0 } else { NONE }).collect();
    let mut line = vec![0.0f64; n];
    let mut out = vec![0.0f64; n];
    let mut hulls = vec![0usize; n];
    let mut bounds = vec![0.0f64; n + 1];
    for y in 0..n {
        line.copy_from_slice(&squared[y * n..(y + 1) * n]);
        distance_1d(&line, &mut out, &mut hulls, &mut bounds);
        squared[y * n..(y + 1) * n].copy_from_slice(&out);
    }
    for x in 0..n {
        for y in 0..n {
            line[y] = squared[y * n + x];
        }
        distance_1d(&line, &mut out, &mut hulls, &mut bounds);
        for y in 0..n {
            squared[y * n + x] = out[y];
        }
    }
    Some(
        squared
            .iter()
            .map(|sq| {
                if *sq >= NONE * 0.5 {
                    NO_LAND
                } else {
                    ((sq.sqrt() as f32) * cell - cell * 0.5).max(0.0)
                }
            })
            .collect(),
    )
}

/// One line of the squared distance transform: `f` the squared distances
/// so far (0 on land), `d` the result; `v` and `z` the lower envelope's
/// parabolas and their bounds (scratch, `f.len()` and `f.len() + 1`).
fn distance_1d(f: &[f64], d: &mut [f64], v: &mut [usize], z: &mut [f64]) {
    let n = f.len();
    let mut k = 0usize;
    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in 1..n {
        let qf = q as f64;
        // (z[0] is minus infinity: the walk back stops at the first parabola)
        let s = loop {
            let p = v[k] as f64;
            let s = ((f[q] + qf * qf) - (f[v[k]] + p * p)) / (2.0 * qf - 2.0 * p);
            if s <= z[k] && k > 0 {
                k -= 1;
            } else {
                break s;
            }
        };
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
    }
    k = 0;
    for (q, out) in d.iter_mut().enumerate().take(n) {
        let qf = q as f64;
        while z[k + 1] < qf {
            k += 1;
        }
        let p = v[k] as f64;
        *out = (qf - p) * (qf - p) + f[v[k]];
    }
}

/// The texels around `xy` of an `n` x `n` map of `cell` metres from
/// `origin`: the four texel indices, the bilinear weights, and whether
/// `xy` is inside the texel centres (outside, the edge holds).
fn texels_at(xy: Vec2, origin: [f32; 2], cell: f32, n: u32) -> ([usize; 4], Vec2, bool) {
    let nf = n as f32;
    let at = (xy - Vec2::from_array(origin)) / cell - Vec2::splat(0.5);
    let pos = at.clamp(Vec2::ZERO, Vec2::splat(nf - 1.0));
    let i = pos.floor().as_uvec2();
    let j = (i + UVec2::ONE).min(UVec2::splat(n - 1));
    let index = |x: u32, y: u32| (y * n + x) as usize;
    let inside = at.cmpge(Vec2::ZERO).all() && at.cmple(Vec2::splat(nf - 1.0)).all();
    (
        [index(i.x, i.y), index(j.x, i.y), index(i.x, j.y), index(j.x, j.y)],
        pos - pos.floor(),
        inside,
    )
}

/// The coarse land-distance field around the camera: per cell, the
/// distance to the nearest land (m) and the water's depth (m), row by row
/// from `origin`.
#[derive(Debug)]
pub struct Field {
    pub origin: [f32; 2],
    pub texels: Box<[[f32; 2]]>,
}

impl Field {
    /// The field of `cells` (`FIELD_SIZE`², row by row from `origin`); None
    /// when they are not that many.
    pub fn build(origin: [f32; 2], cells: &[Cell]) -> Option<Field> {
        let land: Vec<bool> = cells.iter().map(|c| *c == Cell::Land).collect();
        let distances = distance_field(FIELD_SIZE as usize, FIELD_CELL, &land)?;
        let texels = cells
            .iter()
            .zip(distances)
            .map(|(cell, distance)| {
                let depth = match cell {
                    Cell::Land => 0.0,
                    Cell::Water(depth) => depth.max(0.0),
                    Cell::Void => VOID_DEPTH,
                };
                [distance, depth]
            })
            .collect();
        Some(Field { origin, texels })
    }

    fn valid(&self) -> bool {
        self.texels.len() == (FIELD_SIZE * FIELD_SIZE) as usize
    }

    #[cfg(test)]
    fn texel(&self, x: u32, y: u32) -> [f32; 2] {
        self.texels[(y * FIELD_SIZE + x) as usize]
    }

    /// Channel `k` (0 distance, 1 depth) at `xy` and its gradient, as
    /// `ocean_open` reads it on the GPU (the edge texels hold outside the
    /// field, without gradient).
    fn channel(&self, xy: Vec2, k: usize) -> (f32, Vec2) {
        let ([a, b, c, d], f, inside) = texels_at(xy, self.origin, FIELD_CELL, FIELD_SIZE);
        let t = |i: usize| self.texels[i][k];
        let (v, g) = bilinear(t(a), t(b), t(c), t(d), f, FIELD_CELL);
        (v, if inside { g } else { Vec2::ZERO })
    }

    /// How much of the trough the bed allows at `xy`, and its gradient.
    pub fn guard(&self, xy: Vec2, down: f32) -> (f32, Vec2) {
        if !xy.is_finite() || !self.valid() {
            return (0.0, Vec2::ZERO);
        }
        let ([a, b, c, d], f, inside) = texels_at(xy, self.origin, FIELD_CELL, FIELD_SIZE);
        let t = |i: usize| self.texels[i][1];
        let (v, g) = guard(t(a), t(b), t(c), t(d), f, FIELD_CELL, down);
        (v, if inside { g } else { Vec2::ZERO })
    }
}

/// The fine calm-distance map around the camera: per cell (`FINE_CELL`),
/// the distance to the nearest land or static object at the water line
/// (m), row by row from `origin`; the second channel is unused (the
/// texture's format).
#[derive(Debug)]
pub struct FineMap {
    pub origin: [f32; 2],
    pub texels: Box<[[f32; 2]]>,
}

impl FineMap {
    /// The map of `land` (`FINE_SIZE`², row by row from `origin`); None when
    /// they are not that many.
    pub fn build(origin: [f32; 2], land: &[bool]) -> Option<FineMap> {
        let distances = distance_field(FINE_SIZE as usize, FINE_CELL, land)?;
        Some(FineMap {
            origin,
            texels: distances.into_iter().map(|d| [d, 0.0]).collect(),
        })
    }

    /// The distance at `xy`, its gradient and the map's weight there (1
    /// inside, falling to 0 over its outer `FINE_BLEND` metres), as
    /// `ocean_shore` reads it on the GPU.
    pub fn distance(&self, xy: Vec2) -> (f32, Vec2, f32) {
        if !xy.is_finite() || self.texels.len() != (FINE_SIZE * FINE_SIZE) as usize {
            return (NO_LAND, Vec2::ZERO, 0.0);
        }
        let size = FINE_SIZE as f32 * FINE_CELL;
        let local = xy - Vec2::from_array(self.origin);
        let border = local.min(Vec2::splat(size) - local).min_element();
        let x = (border / FINE_BLEND).clamp(0.0, 1.0);
        let weight = x * x * (3.0 - 2.0 * x);
        let ([a, b, c, d], f, inside) = texels_at(xy, self.origin, FINE_CELL, FINE_SIZE);
        let t = |i: usize| self.texels[i][0];
        let (v, g) = bilinear(t(a), t(b), t(c), t(d), f, FINE_CELL);
        (v, if inside { g } else { Vec2::ZERO }, weight)
    }
}

/// The ocean state the viewer hands to the renderer with the bathymetry:
/// the swell's direction, both maps, and two values at the camera.
#[derive(Debug, Clone, Default)]
pub struct Ocean {
    /// Unit direction the swell travels ([`wind`]).
    pub wind: [f32; 2],
    /// While an environment change cross-fades the water to a big wave of
    /// another direction (patch 998): the new swell's direction and its
    /// share, the water normal maps' ([`Ocean::with_next`]); share 0: the
    /// one swell of `wind`.
    pub next_wind: [f32; 2],
    pub share: f32,
    /// The coarse field, while the swell is on.
    pub field: Option<Arc<Field>>,
    /// The fine calm-distance map, while the calming is on.
    pub fine: Option<Arc<FineMap>>,
    /// The swell's movement at the camera (m, -down to +up).
    pub eye_dz: f32,
    /// Its mean level there, (up - down) / 2 * openness (m): the plane of
    /// the planar reflection and of the underwater window's capture.
    pub eye_mean: f32,
}

impl Ocean {
    /// The coarse field, while the swell is on.
    pub fn active(&self, settings: WaterEffects) -> Option<&Field> {
        if swell_on(settings) { self.field.as_deref() } else { None }
    }

    /// The fine map, while the calming is on.
    pub fn calming(&self, settings: WaterEffects) -> Option<&FineMap> {
        if calm_on(settings) { self.fine.as_deref() } else { None }
    }

    fn wind_vec(&self) -> Vec2 {
        let w = Vec2::from_array(self.wind);
        if w.is_finite() && w.length() > 0.5 { w } else { wind(Vec2::ZERO) }
    }

    /// The same state with the swell an environment change fades to:
    /// `next`, the new side's big wave (EEP wave1) and its share. No change,
    /// no share yet or the same direction: the one swell of `wind`; the
    /// whole share: the new swell alone.
    pub fn with_next(mut self, next: Option<(Vec2, f32)>) -> Self {
        self.next_wind = [0.0; 2];
        self.share = 0.0;
        let Some((wave1, share)) = next.filter(|(_, share)| share.is_finite() && *share > 0.0) else {
            return self;
        };
        let to = wind(wave1);
        if share >= 1.0 {
            self.wind = to.to_array();
        } else if to.distance(self.wind_vec()) > SAME_DIRECTION {
            self.next_wind = to.to_array();
            self.share = share;
        }
        self
    }

    /// The long swell at `p`: the one swell, or during a change between two
    /// directions both, each in its own, mixed by the new one's share.
    pub fn swell(&self, p: Vec2, time: f32) -> Long {
        let first = long_swell(p, time, self.wind_vec());
        let next = Vec2::from_array(self.next_wind);
        if self.share > 0.0 && next.is_finite() && next.length() > 0.5 {
            first.mix(long_swell(p, time, next), self.share.min(1.0))
        } else {
            first
        }
    }

    /// The fifth vec4 of the uniform's tail: the new swell's direction and
    /// its share while two are mixed, zero otherwise (`ocean_long_at`).
    fn next_tail(&self) -> [f32; 4] {
        let next = Vec2::from_array(self.next_wind);
        if self.share > 0.0 && next.is_finite() && next.length() > 0.5 {
            [next.x, next.y, self.share.min(1.0), 0.0]
        } else {
            [0.0; 4]
        }
    }

    /// The distance (m) to the nearest land or static object at `xy` and
    /// its gradient: the fine map within it, the coarse field beyond (only
    /// terrain), `NO_LAND` without either.
    pub fn shore(&self, xy: Vec2, settings: WaterEffects) -> (f32, Vec2) {
        let coarse = self
            .active(settings)
            .filter(|f| f.valid() && xy.is_finite())
            .map_or((NO_LAND, Vec2::ZERO), |f| f.channel(xy, 0));
        match self.calming(settings) {
            Some(fine) => {
                let (d, g, w) = fine.distance(xy);
                if w <= 0.0 {
                    coarse
                } else {
                    (coarse.0 + (d - coarse.0) * w, coarse.1.lerp(g, w))
                }
            }
            None => coarse,
        }
    }

    /// The calm factor at `xy` and its gradient (1 and 0 while the calming
    /// is off).
    pub fn calm_at(&self, xy: Vec2, settings: WaterEffects) -> (f32, Vec2) {
        if !calm_on(settings) {
            return (1.0, Vec2::ZERO);
        }
        let (d, g) = self.shore(xy, settings);
        let (c, dc) = calm(d, settings.calm_reach, settings.calm_edge);
        (c, g * dc)
    }

    /// The swell's openness at `xy` and its gradient: the calm factor and
    /// the bed's guard.
    pub fn openness(&self, field: &Field, xy: Vec2, settings: WaterEffects) -> (f32, Vec2) {
        let (g, dg) = field.guard(xy, settings.ocean_depth);
        let (c, dc) = self.calm_at(xy, settings);
        (g * c, dg * c + dc * g)
    }

    /// The mean openness of the coarse field's border: the level the far
    /// water (beyond `FAR_TO`, drawn coarse or as whole tiles) settles at.
    pub fn far_openness(&self, settings: WaterEffects) -> f32 {
        let Some(field) = self.active(settings).filter(|f| f.valid()) else {
            return 0.0;
        };
        let last = FIELD_SIZE - 1;
        let mut sum = 0.0;
        let mut count = 0.0;
        for k in 0..FIELD_SIZE {
            for (x, y) in [(k, 0), (k, last), (0, k), (last, k)] {
                let [distance, depth] = field.texels[(y * FIELD_SIZE + x) as usize];
                let g = guard(depth, depth, depth, depth, Vec2::ZERO, FIELD_CELL, settings.ocean_depth).0;
                let c = if calm_on(settings) {
                    calm(distance, settings.calm_reach, settings.calm_edge).0
                } else {
                    1.0
                };
                sum += g * c;
                count += 1.0;
            }
        }
        sum / count
    }

    /// The same state with its values at the camera `eye`.
    pub fn at_eye(mut self, eye: Vec2, time: f32, settings: WaterEffects) -> Self {
        let (dz, mean) = match self.active(settings) {
            Some(field) => (
                long_drop(&self, field, eye, time, settings),
                0.5 * (settings.ocean_up - settings.ocean_depth) * self.openness(field, eye, settings).0,
            ),
            None => (0.0, 0.0),
        };
        self.eye_dz = dz;
        self.eye_mean = mean;
        self
    }

    /// The plane the planar reflection and the window's capture use: the
    /// region's flat water, or the swell's mean level at the camera.
    pub fn reflection_level(&self, water_height: f32) -> f32 {
        if self.eye_mean != 0.0 {
            water_height + self.eye_mean
        } else {
            water_height
        }
    }
}

/// The long swell's displacement at material position `p`: the slider's
/// mean level and amplitude times the openness there (two swells mixed
/// during a change of direction, [`Ocean::swell`]).
fn long_offset(ocean: &Ocean, field: &Field, p: Vec2, time: f32, settings: WaterEffects) -> Vec3 {
    let open = ocean.openness(field, p, settings).0;
    let a = 0.5 * (settings.ocean_depth + settings.ocean_up) * open;
    let mean = 0.5 * (settings.ocean_up - settings.ocean_depth) * open;
    let l = ocean.swell(p, time);
    (l.shift * a).extend(mean + a * l.s)
}

/// Fixed-point steps that invert the horizontal motion: at 10 m down and
/// 10 m up the long swell contracts a step's error to about 0.42 of it,
/// so eight leave a few millimetres.
const INVERT_STEPS: usize = 8;

/// The long swell's movement under world `xy`, its horizontal motion
/// inverted.
fn long_drop(ocean: &Ocean, field: &Field, xy: Vec2, time: f32, settings: WaterEffects) -> f32 {
    let mut source = xy;
    for _ in 0..INVERT_STEPS {
        source = xy - long_offset(ocean, field, source, time, settings).truncate();
    }
    long_offset(ocean, field, source, time, settings).z
}

/// The displayed surface's height under world `xy` with the swell or the
/// calming on: the existing swell (`super::surface_level`'s, shore-damped
/// the same way, calmed) and the long one together, their horizontal
/// motion inverted (`INVERT_STEPS`).
pub fn surface_level(xy: Vec2, time: f32, settings: WaterEffects, level: f32, ground: f32, ocean: &Ocean) -> f32 {
    if !settings.enabled || !settings.waves {
        return level;
    }
    let t = ((level - ground - 0.08) / 2.32).clamp(0.0, 1.0);
    let shore = t * t * (3.0 - 2.0 * t);
    let field = ocean.active(settings);
    // (water_effects.rs surface_level's waves, water_spectrum.wgsl water_swell)
    let small = |p: Vec2| {
        let mut offset = Vec3::ZERO;
        for (i, (direction, k, amplitude)) in [
            (Vec2::new(0.88, 0.475), 0.19f32, 0.24),
            (Vec2::new(0.62, 0.785), 0.32, 0.115),
            (Vec2::new(-0.22, 0.975), 0.48, 0.055),
            (Vec2::new(0.97, -0.243), 0.71, 0.025),
        ]
        .into_iter()
        .take((settings.quality.min(3) + 2).min(4) as usize)
        .enumerate()
        {
            let phase = p.dot(direction) * k - time * (9.81 * k).sqrt() + i as f32 * 1.73;
            offset += (direction * phase.cos() * amplitude * 0.85).extend(phase.sin() * amplitude);
        }
        offset * shore * ocean.calm_at(p, settings).0
    };
    let sample = |p: Vec2| small(p) + field.map_or(Vec3::ZERO, |f| long_offset(ocean, f, p, time, settings));
    let mut source = xy;
    for _ in 0..INVERT_STEPS {
        source = xy - sample(source).truncate();
    }
    level + sample(source).z
}

impl WaterGround {
    /// The displayed surface's height under `xy`: `surface_level` as
    /// upstream has it, or with the open-sea swell and the calming while
    /// they are on.
    pub fn surface(&self, xy: Vec2, time: f32, settings: WaterEffects, level: f32, ground: f32) -> f32 {
        if self.ocean.active(settings).is_some() || self.ocean.calming(settings).is_some() {
            surface_level(xy, time, settings, level, ground, &self.ocean)
        } else {
            super::surface_level(xy, time, settings, level, ground)
        }
    }

    /// The swell's own movement under `xy` (-down to +up); None while it is
    /// off.
    pub fn ocean_dz(&self, xy: Vec2, time: f32, settings: WaterEffects) -> Option<f32> {
        let field = self.ocean.active(settings)?;
        xy.is_finite().then(|| long_drop(&self.ocean, field, xy, time, settings))
    }
}

/// The renderer's side: what was last written to the uniform's tail and
/// which maps the texture holds.
#[derive(Default)]
pub(super) struct Gpu {
    written: Tail,
    field: Option<Arc<Field>>,
    fine: Option<Arc<FineMap>>,
}

fn upload(queue: &wgpu::Queue, texture: &wgpu::Texture, row: u32, size: u32, texels: &[[f32; 2]]) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: row, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(texels),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size * 8),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
    );
}

impl super::WaterGpu {
    /// Each frame after the shared uniform: each map into the bathymetry
    /// texture's lower rows when it is new, and the uniform's tail when it
    /// changed (it stays zero, as created, while both are off).
    pub(super) fn prepare_ocean(&mut self, queue: &wgpu::Queue, settings: WaterEffects, ground: &WaterGround, valid: bool) {
        let usable = valid && settings.flags() & 1 != 0;
        let field = ground
            .ocean
            .field
            .as_ref()
            .filter(|_| usable && ground.ocean.active(settings).is_some());
        let fine = ground
            .ocean
            .fine
            .as_ref()
            .filter(|_| usable && ground.ocean.calming(settings).is_some());
        let mut wanted: Tail = [0.0; 20];
        if let Some(field) = field {
            if !self.ocean.field.as_ref().is_some_and(|old| Arc::ptr_eq(old, field)) {
                self.ocean.field = Some(field.clone());
                upload(queue, &self.ground, FIELD_ROW, FIELD_SIZE, &field.texels);
            }
            let w = ground.ocean.wind_vec();
            wanted[..8].copy_from_slice(&[
                settings.ocean_depth + settings.ocean_up,
                w.x,
                w.y,
                ground.ocean.far_openness(settings),
                field.origin[0],
                field.origin[1],
                FIELD_CELL,
                1.0,
            ]);
            // the swell an environment change fades to, while it has a share
            wanted[16..].copy_from_slice(&ground.ocean.next_tail());
        }
        if let Some(fine) = fine {
            if !self.ocean.fine.as_ref().is_some_and(|old| Arc::ptr_eq(old, fine)) {
                self.ocean.fine = Some(fine.clone());
                upload(queue, &self.ground, FINE_ROW, FINE_SIZE, &fine.texels);
            }
            wanted[12..16].copy_from_slice(&[fine.origin[0], fine.origin[1], FINE_CELL, 1.0]);
        }
        if field.is_some() || fine.is_some() {
            // the calming's values whenever it is on, as the CPU's
            // `Ocean::calm_at` takes them (the coarse field alone until the
            // fine map is in)
            let calming = calm_on(settings);
            wanted[8..12].copy_from_slice(&[
                settings.ocean_depth,
                settings.ocean_up,
                if calming { settings.calm_reach } else { 0.0 },
                if calming { settings.calm_edge } else { 1.0 },
            ]);
        }
        if wanted != self.ocean.written {
            queue.write_buffer(&self.uniform, UNIFORM_OFFSET, bytemuck::cast_slice(&wanted));
            self.ocean.written = wanted;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open_field() -> Field {
        let n = (FIELD_SIZE * FIELD_SIZE) as usize;
        Field::build([-1024.0, -1024.0], &vec![Cell::Void; n]).expect("field")
    }

    fn fine_size() -> usize {
        (FINE_SIZE * FINE_SIZE) as usize
    }

    /// Down, up; no calming.
    fn on(down: f32, up: f32, quality: u8) -> WaterEffects {
        WaterEffects {
            quality,
            ocean_depth: down,
            ocean_up: up,
            calm_edge: 1.0,
            ..Default::default()
        }
    }

    fn calmed(down: f32, up: f32, reach: f32, edge: f32) -> WaterEffects {
        WaterEffects {
            quality: 3,
            ocean_depth: down,
            ocean_up: up,
            calm_reach: reach,
            calm_edge: edge,
            ..Default::default()
        }
    }

    fn ground(field: Option<Field>, fine: Option<FineMap>) -> WaterGround {
        WaterGround {
            origin: [-128.0; 2],
            extent: 256.0,
            heights: vec![[0.0, 20.0]; 4096].into(),
            ocean: Ocean {
                wind: wind(Vec2::new(1.05, -0.42)).to_array(),
                field: field.map(Arc::new),
                fine: fine.map(Arc::new),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// A fine map around 0 with a 4 m pier along x = 0 (static land).
    fn pier() -> FineMap {
        let n = FINE_SIZE as usize;
        let origin = [-256.0, -256.0];
        let land: Vec<bool> = (0..n * n)
            .map(|i| {
                let x = origin[0] + (i % n) as f32 + 0.5;
                x.abs() < 2.0
            })
            .collect();
        FineMap::build(origin, &land).expect("fine")
    }

    #[test]
    fn the_swell_reaches_down_and_up_and_crests_stay_at_the_level_without_up() {
        let g = ground(Some(open_field()), None);
        // the existing swell's reach: the sum of its amplitudes
        let reach = 0.24 + 0.115 + 0.055 + 0.025 + 1e-4;
        for (down, up) in [
            (0.5, 0.0),
            (5.0, 0.0),
            (10.0, 0.0),
            (0.0, 4.0),
            (0.0, 10.0),
            (3.0, 6.0),
            (10.0, 10.0),
        ] {
            let settings = on(down, up, 3);
            let (mut low, mut high) = (f32::MAX, f32::MIN);
            for i in 0..40 {
                for j in 0..40 {
                    let xy = Vec2::new(i as f32 * 15.3 - 300.0, j as f32 * 14.1 - 280.0);
                    let time = (i * 40 + j) as f32 * 0.37;
                    let level = g.surface(xy, time, settings, 20.0, 0.0) - 20.0;
                    let dz = g.ocean_dz(xy, time, settings).expect("on");
                    assert!(dz <= up + 1e-4 && dz >= -down - 1e-4, "{dz} at {down} / {up}");
                    assert!(level <= up + reach && level >= -down - reach, "{level} at {down} / {up}");
                    low = low.min(dz);
                    high = high.max(dz);
                }
            }
            let span = down + up;
            assert!(low < -down + 0.4 * span, "troughs reach toward -down: {low} at {down} / {up}");
            assert!(high > up - 0.4 * span, "crests reach toward +up: {high} at {down} / {up}");
            if up == 0.0 {
                assert!(high <= 1e-5, "without up the crests stay at the level: {high}");
            }
            if down == 0.0 {
                assert!(low >= -1e-5, "without down the troughs stay at the level: {low}");
            }
        }
    }

    #[test]
    fn the_bed_rule_keeps_troughs_off_the_bed_whatever_the_crests() {
        for water in [0.5, 1.0, 2.0, 3.5, 6.0, 40.0] {
            for down in [0.5, 2.0, 5.0, 10.0] {
                for up in [0.0, 5.0, 10.0] {
                    let (g, _) = guard(water, water, water, water, Vec2::ZERO, FIELD_CELL, down);
                    // the trough: mean - amplitude = -down * openness
                    let a = 0.5 * (down + up) * g;
                    let trough = 0.5 * (up - down) * g - a;
                    assert!(-trough <= (water - KEEL).max(0.0) + 1e-5, "{water} {down} {up} {g}");
                }
            }
        }
        // up only: no trough, the guard lets the deep water rise in full
        assert_eq!(guard(30.0, 30.0, 30.0, 30.0, Vec2::ZERO, FIELD_CELL, 0.0).0, 1.0);
        // the analytic gradient matches the slope inside a texel
        let f = Vec2::new(0.3, 0.6);
        let h = 1e-3;
        let (_, grad) = guard(4.0, 4.6, 5.5, 3.0, f, FIELD_CELL, 5.0);
        let dx = (guard(4.0, 4.6, 5.5, 3.0, f + Vec2::X * h, FIELD_CELL, 5.0).0
            - guard(4.0, 4.6, 5.5, 3.0, f - Vec2::X * h, FIELD_CELL, 5.0).0)
            / (2.0 * h * FIELD_CELL);
        assert!((grad.x - dx).abs() < 1e-3, "{grad} vs {dx}");
    }

    #[test]
    fn calming_curve_at_the_edge_in_the_middle_and_beyond_the_distance() {
        for edge in [0.0f32, 0.5, 1.0] {
            let reach = 100.0;
            assert_eq!(calm(0.0, reach, edge).0, edge, "the contact line");
            assert_eq!(calm(-3.0, reach, edge).0, edge);
            let middle = calm(50.0, reach, edge).0;
            assert!((middle - (edge + (1.0 - edge) * 0.5)).abs() < 1e-6, "{middle}");
            assert_eq!(calm(reach, reach, edge).0, 1.0, "at the distance");
            assert_eq!(calm(5000.0, reach, edge).0, 1.0, "beyond it");
            // smooth and rising
            let mut last = edge;
            for i in 1..=100 {
                let (c, dc) = calm(i as f32, reach, edge);
                assert!(c >= last - 1e-6);
                let h = 1e-2;
                let numeric = (calm(i as f32 - 0.5 + h, reach, edge).0 - calm(i as f32 - 0.5 - h, reach, edge).0) / (2.0 * h);
                assert!((calm(i as f32 - 0.5, reach, edge).1 - numeric).abs() < 1e-3);
                last = c;
                if edge == 1.0 {
                    assert_eq!((c, dc), (1.0, 0.0), "edge 1: no calming");
                }
            }
        }
        // reach 0: never calmed
        assert_eq!(calm(0.0, 0.0, 0.0), (1.0, 0.0));
    }

    #[test]
    fn the_calming_scales_every_vertical_movement_near_a_static_pier() {
        let settings = calmed(4.0, 2.0, 100.0, 0.0);
        let g = ground(Some(open_field()), Some(pier()));
        // the fine map's distance: about the distance from the pier's side
        let d = g.ocean.shore(Vec2::new(30.0, 10.0), settings).0;
        assert!((d - 28.0).abs() < 1.0, "{d}");
        // at the pier the water does not move, at 100 m and beyond in full
        let still = (0..50).all(|i| {
            let t = i as f32 * 0.9;
            (g.surface(Vec2::new(2.2, 10.0), t, settings, 20.0, 0.0) - 20.0).abs() < 0.02
        });
        assert!(still, "calm at the contact line");
        let mut far_range = 0.0f32;
        for i in 0..80 {
            let t = i as f32 * 0.9;
            let dz = g.ocean_dz(Vec2::new(150.0, 10.0), t, settings).expect("on");
            far_range = far_range.max(dz.abs());
        }
        assert!(far_range > 1.0, "full beyond the distance: {far_range}");
        // the edge at 0.5 halves the movement at the contact line
        let half = calmed(4.0, 2.0, 100.0, 0.5);
        let (c, _) = g.ocean.calm_at(Vec2::new(2.0, 10.0), half);
        assert!((c - 0.5).abs() < 0.02, "{c}");
        // edge 1: exactly no calming
        assert_eq!(
            g.ocean.calm_at(Vec2::new(2.0, 10.0), calmed(4.0, 2.0, 100.0, 1.0)),
            (1.0, Vec2::ZERO)
        );
        // the upstream waves alone (no swell) are calmed too
        let waves_only = calmed(0.0, 0.0, 100.0, 0.0);
        let g2 = ground(None, Some(pier()));
        let mut near = 0.0f32;
        let mut far = 0.0f32;
        for i in 0..60 {
            let t = i as f32 * 0.37;
            near = near.max((g2.surface(Vec2::new(2.1, 10.0), t, waves_only, 20.0, 0.0) - 20.0).abs());
            far = far.max((g2.surface(Vec2::new(120.0, 10.0), t, waves_only, 20.0, 0.0) - 20.0).abs());
        }
        assert!(near < 0.01 && far > 0.2, "{near} {far}");
    }

    #[test]
    fn off_switches_keep_upstreams_surface_exactly() {
        let g = ground(Some(open_field()), Some(pier()));
        for quality in 0..4 {
            for (x, y, t) in [(43.0, 118.0, 7.3), (-12.0, 3.5, 0.0), (300.0, -77.0, 1234.5)] {
                let xy = Vec2::new(x, y);
                let up = super::super::surface_level(xy, t, on(0.0, 0.0, quality), 20.0, 13.0);
                assert_eq!(g.surface(xy, t, on(0.0, 0.0, quality), 20.0, 13.0).to_bits(), up.to_bits());
                assert_eq!(g.ocean_dz(xy, t, on(0.0, 0.0, quality)), None);
                for off in [
                    WaterEffects {
                        enabled: false,
                        ..calmed(3.0, 3.0, 100.0, 0.0)
                    },
                    WaterEffects {
                        waves: false,
                        ..calmed(3.0, 3.0, 100.0, 0.0)
                    },
                ] {
                    assert_eq!(
                        g.surface(xy, t, off, 20.0, 13.0),
                        super::super::surface_level(xy, t, off, 20.0, 13.0)
                    );
                    assert_eq!(g.ocean_dz(xy, t, off), None);
                }
            }
        }
        // no maps yet: upstream's
        let empty = ground(None, None);
        let xy = Vec2::new(5.0, 9.0);
        assert_eq!(
            empty.surface(xy, 3.0, calmed(4.0, 1.0, 100.0, 0.0), 20.0, 0.0),
            super::super::surface_level(xy, 3.0, calmed(4.0, 1.0, 100.0, 0.0), 20.0, 0.0)
        );
        let o = Ocean::default().at_eye(xy, 3.0, on(4.0, 2.0, 1));
        assert_eq!((o.eye_dz, o.eye_mean, o.reflection_level(20.0)), (0.0, 0.0, 20.0));
        // the uniform's tail starts zero, after the caustics' extension
        let bytes = initial_uniform(super::super::caustic_texture::initial_uniform());
        assert_eq!(bytes.len() as u64, UNIFORM_OFFSET + TAIL_BYTES);
        assert_eq!(TAIL_BYTES as usize, std::mem::size_of::<Tail>());
        assert!(bytes[UNIFORM_OFFSET as usize..].iter().all(|b| *b == 0));
    }

    #[test]
    fn a_closed_field_gives_upstreams_swell_and_the_inversion_lands_on_its_point() {
        // all land under the coarse field, no calming: openness 0 by the
        // bed's guard, the existing swell alone
        let n = (FIELD_SIZE * FIELD_SIZE) as usize;
        let land = Field::build([-1024.0; 2], &vec![Cell::Land; n]).expect("field");
        let g = ground(Some(land), None);
        for quality in 0..4 {
            let settings = on(5.0, 3.0, quality);
            for i in 0..30 {
                let xy = Vec2::new(i as f32 * 7.7 - 90.0, i as f32 * 3.1);
                let t = i as f32 * 0.9;
                let ours = g.surface(xy, t, settings, 20.0, 0.0);
                let up = super::super::surface_level(xy, t, settings, 20.0, 0.0);
                assert!((ours - up).abs() < 5e-4, "{ours} vs {up}");
            }
        }
        // open: the source found by the inversion moves back onto xy
        let mut g = ground(Some(open_field()), None);
        g.ocean.wind = wind(Vec2::new(0.3, 1.0)).to_array();
        let field = g.ocean.field.as_deref().expect("field");
        let settings = on(10.0, 10.0, 3);
        for i in 0..50 {
            let xy = Vec2::new(i as f32 * 13.0 - 300.0, 40.0 - i as f32 * 5.0);
            let t = i as f32 * 1.7;
            let mut source = xy;
            for _ in 0..INVERT_STEPS {
                source = xy - long_offset(&g.ocean, field, source, t, settings).truncate();
            }
            let back = source + long_offset(&g.ocean, field, source, t, settings).truncate();
            assert!(back.distance(xy) < 0.02, "{back} vs {xy}");
        }
    }

    #[test]
    fn the_surface_never_folds_at_full_height() {
        let g = ground(Some(open_field()), None);
        let field = g.ocean.field.as_deref().expect("field");
        let settings = on(DEPTH_MAX, UP_MAX, 3);
        let w = wind(Vec2::ZERO);
        for i in 0..400 {
            let p = Vec2::new(i as f32 * 3.7, i as f32 * -2.3);
            let l = long_swell(p, i as f32 * 0.11, w);
            assert!(l.s.abs() <= 1.0 + 1e-6);
            let a = 0.5 * (DEPTH_MAX + UP_MAX) * g.ocean.openness(field, p, settings).0;
            let j = Vec3::new(1.0, 0.0, 1.0) - l.fold * a;
            let det = j.x * j.z - j.y * j.y;
            assert!(det > 0.25, "{det}");
        }
    }

    /// An open field while the environment changes from the default big
    /// wave to one of another direction, the new swell's `share`.
    fn changing(share: f32) -> WaterGround {
        let mut g = ground(Some(open_field()), None);
        g.ocean = g.ocean.with_next(Some((Vec2::new(-0.2, 1.3), share)));
        g
    }

    #[test]
    fn a_change_of_direction_mixes_two_swells_each_in_its_own_direction() {
        let first = wind(Vec2::new(1.05, -0.42));
        let next = wind(Vec2::new(-0.2, 1.3));
        for share in [0.1f32, 0.5, 0.9] {
            let g = changing(share);
            assert_eq!(
                (g.ocean.wind, g.ocean.next_wind, g.ocean.share),
                (first.to_array(), next.to_array(), share)
            );
            for i in 0..40 {
                let p = Vec2::new(i as f32 * 23.0 - 400.0, 300.0 - i as f32 * 17.0);
                let t = i as f32 * 0.7;
                let (a, b) = (long_swell(p, t, first), long_swell(p, t, next));
                let m = g.ocean.swell(p, t);
                let keep = 1.0 - share;
                assert!((m.s - (a.s * keep + b.s * share)).abs() < 1e-6);
                assert!(m.ds.distance(a.ds * keep + b.ds * share) < 1e-6);
                assert!(m.shift.distance(a.shift * keep + b.shift * share) < 1e-6);
                assert!(m.fold.distance(a.fold * keep + b.fold * share) < 1e-6);
            }
        }
        // linear in the share: halfway between a quarter and three quarters
        let p = Vec2::new(37.0, -81.0);
        let s = |share: f32| changing(share).ocean.swell(p, 4.0).s;
        assert!((s(0.5) - 0.5 * (s(0.25) + s(0.75))).abs() < 1e-5);
        // the GPU reads the new direction and the share (ocean_long_at)
        assert_eq!(changing(0.5).ocean.next_tail(), [next.x, next.y, 0.5, 0.0]);
    }

    #[test]
    fn the_cpu_sample_is_the_mixed_field() {
        let settings = on(4.0, 2.0, 3);
        let share = 0.4;
        let g = changing(share);
        let field = g.ocean.field.as_deref().expect("field");
        let (first, next) = (wind(Vec2::new(1.05, -0.42)), wind(Vec2::new(-0.2, 1.3)));
        // the mixed field written out: each swell's displacement at the
        // same openness, mixed by the share
        let offset = |p: Vec2, t: f32| {
            let open = g.ocean.openness(field, p, settings).0;
            let a = 0.5 * (settings.ocean_depth + settings.ocean_up) * open;
            let mean = 0.5 * (settings.ocean_up - settings.ocean_depth) * open;
            let one = |w: Vec2| {
                let l = long_swell(p, t, w);
                (l.shift * a).extend(mean + a * l.s)
            };
            one(first) * (1.0 - share) + one(next) * share
        };
        for i in 0..30 {
            let xy = Vec2::new(i as f32 * 19.0 - 280.0, 60.0 - i as f32 * 7.0);
            let t = 3.0 + i as f32 * 1.3;
            let mut source = xy;
            for _ in 0..INVERT_STEPS {
                source = xy - offset(source, t).truncate();
            }
            assert!((source + offset(source, t).truncate()).distance(xy) < 0.02);
            let expected = offset(source, t).z;
            let dz = g.ocean_dz(xy, t, settings).expect("on");
            assert!((dz - expected).abs() < 1e-4, "{dz} vs {expected}");
            // the camera's values (the ride, the reflection plane) take the same
            let eye = g.ocean.clone().at_eye(xy, t, settings);
            assert!((eye.eye_dz - dz).abs() < 1e-6);
        }
    }

    #[test]
    fn a_change_starts_and_ends_on_one_swell_without_a_step() {
        let settings = on(4.0, 2.0, 3);
        let next = wind(Vec2::new(-0.2, 1.3));
        let before = ground(Some(open_field()), None);
        let mut after = ground(Some(open_field()), None);
        after.ocean.wind = next.to_array();
        let (start, end) = (changing(0.0), changing(1.0));
        assert_eq!(
            (start.ocean.wind, start.ocean.share),
            (before.ocean.wind, 0.0),
            "no share yet: the old swell alone"
        );
        assert_eq!(
            (end.ocean.wind, end.ocean.share),
            (next.to_array(), 0.0),
            "the whole share: the new one alone"
        );
        assert_eq!((start.ocean.next_tail(), end.ocean.next_tail()), ([0.0; 4], [0.0; 4]));
        let (just_begun, almost_done) = (changing(1e-4), changing(1.0 - 1e-4));
        for i in 0..30 {
            let xy = Vec2::new(i as f32 * 31.0 - 450.0, i as f32 * 11.0 - 90.0);
            let t = 1.0 + i as f32 * 2.1;
            let dz = |g: &WaterGround| g.ocean_dz(xy, t, settings).expect("on");
            assert_eq!(dz(&start), dz(&before));
            assert_eq!(dz(&end), dz(&after));
            assert!((dz(&just_begun) - dz(&before)).abs() < 2e-3);
            assert!((dz(&almost_done) - dz(&after)).abs() < 2e-3);
        }
    }

    #[test]
    fn the_same_direction_on_both_sides_is_one_swell() {
        let g = ground(Some(open_field()), None);
        // a longer big wave the same way: the same swell, no fade
        let same = g.ocean.clone().with_next(Some((Vec2::new(1.05, -0.42) * 3.0, 0.5)));
        assert_eq!((same.wind, same.next_wind, same.share), (g.ocean.wind, [0.0; 2], 0.0));
        assert_eq!(same.next_tail(), [0.0; 4]);
        let p = Vec2::new(12.0, 34.0);
        assert_eq!(same.swell(p, 5.0), long_swell(p, 5.0, g.ocean.wind_vec()));
        // no change, or a share that is not a number: one swell
        assert_eq!(g.ocean.clone().with_next(None).share, 0.0);
        assert_eq!(g.ocean.clone().with_next(Some((Vec2::new(-0.2, 1.3), f32::NAN))).share, 0.0);
        assert_eq!(g.ocean.swell(p, 5.0), long_swell(p, 5.0, g.ocean.wind_vec()));
    }

    fn cells(f: impl Fn(usize, usize) -> Cell) -> Vec<Cell> {
        let n = FIELD_SIZE as usize;
        (0..n * n).map(|i| f(i % n, i / n)).collect()
    }

    #[test]
    fn distance_field_on_an_island_a_coast_a_missing_patch_and_the_void() {
        // an island: a 5-cell square in the middle
        let island = Field::build(
            [0.0; 2],
            &cells(|x, y| {
                if (62..67).contains(&x) && (62..67).contains(&y) {
                    Cell::Land
                } else {
                    Cell::Water(30.0)
                }
            }),
        )
        .expect("field");
        assert_eq!(island.texel(64, 64), [0.0, 0.0]);
        let along = island.texel(76, 64)[0]; // 10 cells east of its edge
        assert!((along - (10.0 * FIELD_CELL - FIELD_CELL * 0.5)).abs() < 1e-3, "{along}");
        let diagonal = island.texel(76, 76)[0];
        assert!(
            (diagonal - ((200.0f32).sqrt() * FIELD_CELL - FIELD_CELL * 0.5)).abs() < 1e-2,
            "{diagonal}"
        );
        assert_eq!(island.texel(10, 10)[1], 30.0);
        // a straight coast: land west of x = 40
        let coast = Field::build([0.0; 2], &cells(|x, _| if x < 40 { Cell::Land } else { Cell::Void })).expect("field");
        for x in 40..128u32 {
            let expected = (x - 39) as f32 * FIELD_CELL - FIELD_CELL * 0.5;
            assert!((coast.texel(x, 77)[0] - expected).abs() < 1e-3);
            assert_eq!(coast.texel(x, 77)[1], VOID_DEPTH);
        }
        // a missing terrain patch counts as land like the island
        let patch = Field::build(
            [0.0; 2],
            &cells(|x, y| if x / 16 == 2 && y / 16 == 5 { Cell::Land } else { Cell::Void }),
        )
        .expect("field");
        assert_eq!(patch.texel(40, 88)[0], 0.0);
        assert!(patch.texel(40, 100)[0] > 0.0);
        // no land at all: open everywhere
        let void = open_field();
        assert!(void.texels.iter().all(|t| *t == [NO_LAND, VOID_DEPTH]));
        let g = ground(Some(open_field()), None);
        assert_eq!(g.ocean.far_openness(on(5.0, 0.0, 3)), 1.0);
        // a wrong cell count is refused
        assert!(Field::build([0.0; 2], &[Cell::Void; 3]).is_none());
        assert!(FineMap::build([0.0; 2], &[false; 3]).is_none());
    }

    #[test]
    fn the_fine_map_hands_over_to_the_coarse_field_at_its_border() {
        // the coarse field: a coast west of x = 0 (cells of 16 m from -1024)
        let coast = Field::build(
            [-1024.0, -1024.0],
            &cells(|x, _| if x < 64 { Cell::Land } else { Cell::Water(30.0) }),
        )
        .expect("field");
        let empty = FineMap::build([-256.0, -256.0], &vec![false; fine_size()]).expect("fine");
        let g = ground(Some(coast), Some(empty));
        let settings = calmed(3.0, 0.0, 100.0, 0.0);
        // inside the fine map (no land in it): its NO_LAND
        assert!(g.ocean.shore(Vec2::new(0.0, 0.0), settings).0 > NO_LAND - 1.0);
        // beyond it: the coarse coast's distance
        let beyond = g.ocean.shore(Vec2::new(300.0, 0.0), settings).0;
        assert!((beyond - 300.0).abs() < 16.0, "{beyond}");
        // within the border band: between the two
        let band = g.ocean.shore(Vec2::new(240.0, 0.0), settings).0;
        assert!(band > 240.0 && band < NO_LAND, "{band}");
    }

    #[test]
    fn the_wind_follows_the_big_wave_and_the_settings_are_bounded() {
        assert!((wind(Vec2::new(2.0, 0.0)) - Vec2::new(-1.0, 0.0)).length() < 1e-6);
        assert!((wind(Vec2::ZERO) - wind(Vec2::new(1.05, -0.42))).length() < 1e-5);
        assert_eq!(wind(Vec2::new(f32::NAN, 1.0)), wind(Vec2::ZERO));
        assert_eq!(depth(f32::NAN), 0.0);
        assert_eq!(depth(19.0), DEPTH_MAX);
        assert_eq!(depth(-1.0), 0.0);
        assert_eq!(up(f32::NAN), 0.0);
        assert_eq!(up(12.0), UP_MAX);
        assert_eq!(calm_reach(f32::NAN), CALM_REACH_DEFAULT);
        assert_eq!(calm_reach(250.0), 100.0);
        assert_eq!(calm_reach(-1.0), 0.0);
        assert_eq!(calm_edge(f32::NAN), CALM_EDGE_DEFAULT);
        assert_eq!(calm_edge(2.0), 1.0);
        assert!(!swell_on(on(0.0, 0.0, 3)));
        assert!(swell_on(on(0.0, 0.1, 3)));
        assert!(!calm_on(on(1.0, 0.0, 3)), "edge 1: no calming");
        assert!(!calm_on(calmed(1.0, 0.0, 0.0, 0.0)), "reach 0: no calming");
        assert!(calm_on(WaterEffects::default()), "on by default with the waves");
    }

    #[test]
    fn the_shader_constants_are_these() {
        let spectrum = include_str!("shaders/water_spectrum.wgsl");
        let waves: Vec<String> = WAVES
            .iter()
            .map(|w| format!("vec4<f32>({:?},{:?},{:?},{:?})", w[0], w[1], w[2], w[3]))
            .collect();
        assert!(
            spectrum.contains(&format!("const OCEAN_WAVES=array<vec4<f32>,4>({});", waves.join(","))),
            "OCEAN_WAVES"
        );
        for (name, value) in [
            ("OCEAN_STEEPNESS", STEEPNESS),
            ("OCEAN_SET_LENGTH", SET_LENGTH),
            ("OCEAN_SET_SPEED", SET_SPEED),
            ("OCEAN_SET_DEPTH", SET_DEPTH),
            ("OCEAN_KEEL", KEEL),
        ] {
            assert!(spectrum.contains(&format!("const {name}:f32={value:?};")), "{name}");
        }
        let common = include_str!("shaders/common.wgsl");
        for (name, value) in [
            ("OCEAN_FAR_FROM", FAR_FROM),
            ("OCEAN_FAR_TO", FAR_TO),
            ("OCEAN_NORMAL_TO", NORMAL_TO),
            ("OCEAN_SHADE_DEPTH", SHADE_DEPTH),
            ("OCEAN_NO_LAND", NO_LAND),
            ("OCEAN_FINE_BLEND", FINE_BLEND),
        ] {
            assert!(common.contains(&format!("const {name}:f32={value:?};")), "{name}");
        }
        assert!(common.contains(&format!("const OCEAN_FIELD_SIZE:i32={FIELD_SIZE};")));
        assert!(common.contains(&format!("const OCEAN_FIELD_ROW:i32={FIELD_ROW};")));
        assert!(common.contains(&format!("const OCEAN_FINE_SIZE:i32={FINE_SIZE};")));
        assert!(common.contains(&format!("const OCEAN_FINE_ROW:i32={FINE_ROW};")));
        // the uniform: the caustics' five vec4 after the shared four, then ours
        assert!(common.contains(
            "counts: vec4<u32>, caustic: vec4<f32>, caustic_rest: array<vec4<f32>,4>, ocean: vec4<f32>, ocean_area: vec4<f32>, ocean_shape: vec4<f32>, ocean_fine: vec4<f32>, ocean_next: vec4<f32> };"
        ));
    }

    #[test]
    fn cpu_and_gpu_long_swell_guard_and_calm_agree() {
        use crate::pipeline_cache::CachedPipelines;
        let _gpu = crate::test_gpu::lock();
        let instance = wgpu::Instance::default();
        let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
            return;
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).expect("GPU device");
        let source = format!(
            "{}\n{}",
            include_str!("shaders/water_spectrum.wgsl"),
            r#"
@group(0) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
@compute @workgroup_size(1)
fn probe(@builtin(global_invocation_id) id:vec3<u32>) {
    let q=f32(id.x);
    let p=vec2<f32>(q*37.3-410.0,55.0-q*11.9);
    let time=12.5+q*0.7;
    let wind=normalize(vec2<f32>(cos(q*0.9),sin(q*0.9)));
    let l=ocean_long(p,time,wind);
    let m=ocean_long_mix(l,ocean_long(p,time,vec2<f32>(-wind.y,wind.x)),fract(q*0.29));
    let a=0.5+q*0.3;
    let f=fract(vec2<f32>(q*0.37,q*0.61));
    let g=ocean_guard(a,a+0.8,a+1.5,a-0.2,f,16.0,2.5);
    let b=ocean_bilinear(80.0+q*6.0,92.0+q*6.0,71.0+q*6.0,85.0+q*6.0,f,1.0);
    let c=ocean_calm(q*4.5,60.0,0.25);
    result[id.x*5u]=vec4<f32>(l.s,l.ds,g.x);
    result[id.x*5u+1u]=vec4<f32>(l.shift,g.yz);
    result[id.x*5u+2u]=vec4<f32>(l.fold,c.x);
    result[id.x*5u+3u]=vec4<f32>(b,c.y);
    result[id.x*5u+4u]=vec4<f32>(m.s,m.shift,m.fold.y);
}
"#
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CPU / rendered open-sea swell agreement"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline_cached(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });
        const QUERIES: u64 = 24;
        let bytes = QUERIES * 5 * 16;
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: crate::readback::readback_size(bytes),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: output.as_entire_binding(),
            }],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(QUERIES as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, bytes);
        queue.submit([encoder.finish()]);
        readback.map_async(wgpu::MapMode::Read, ..bytes, |r| r.expect("ocean readback"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("GPU completion");
        let mapped = readback.get_mapped_range(..bytes).expect("mapped ocean");
        let samples = crate::readback::f32x4s(&mapped);
        for query in 0..QUERIES as usize {
            let q = query as f32;
            let p = Vec2::new(q * 37.3 - 410.0, 55.0 - q * 11.9);
            let time = 12.5 + q * 0.7;
            let w = Vec2::new((q * 0.9).cos(), (q * 0.9).sin()).normalize();
            let l = long_swell(p, time, w);
            let share = (q * 0.29).fract();
            let m = l.mix(long_swell(p, time, Vec2::new(-w.y, w.x)), share);
            let a = 0.5 + q * 0.3;
            let f = Vec2::new(q * 0.37, q * 0.61);
            let f = f - f.floor();
            let (g, dg) = guard(a, a + 0.8, a + 1.5, a - 0.2, f, 16.0, 2.5);
            let (b, db) = bilinear(80.0 + q * 6.0, 92.0 + q * 6.0, 71.0 + q * 6.0, 85.0 + q * 6.0, f, 1.0);
            let (c, dc) = calm(q * 4.5, 60.0, 0.25);
            let cpu = [
                [l.s, l.ds.x, l.ds.y, g],
                [l.shift.x, l.shift.y, dg.x, dg.y],
                [l.fold.x, l.fold.y, l.fold.z, c],
                [b, db.x, db.y, dc],
                [m.s, m.shift.x, m.shift.y, m.fold.y],
            ];
            for (row, expected) in cpu.iter().enumerate() {
                let gpu = samples[query * 5 + row];
                for (e, v) in expected.iter().zip(gpu) {
                    assert!(
                        (e - v).abs() < 2e-3 * e.abs().max(1.0),
                        "query {query} row {row}: CPU {expected:?} GPU {gpu:?}"
                    );
                }
            }
        }
        drop(mapped);
        readback.unmap();
    }
}
