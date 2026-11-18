//! Open-sea swell (local patch set, scripts/patches/109-ocean-swell.toml):
//! long rolling waves on open water, a viewer-local look. Nothing is sent
//! and the environment asset is untouched; the Legacy Viewer has no such
//! effect (its water is flat), so nothing here is ported.
//!
//! - **Model.** Four long Gerstner waves of 54 to 131 m (`WAVES`), within
//!   about 25 degrees of the wind, slowly modulated into sets that travel at
//!   the group speed (`SET_*`). `s` is their height normalised to [-1, 1].
//!   The surface drops by `depth / 2 * openness * (s - 1)`: crests stay at
//!   the water's own level (where the existing swell already reaches), the
//!   troughs reach `depth` below it. The existing swell, the wakes and the
//!   micro waves ride on top unchanged.
//! - **Openness** (0..1): nothing within `OPEN_FROM` of land, full strength
//!   from `OPEN_FULL`, and never so deep that a trough comes within `KEEL`
//!   of the bed. Land is terrain at or above the water (`LAND_MARGIN`),
//!   a missing terrain patch, never a prim or a mesh; water with no region
//!   at all is open sea. The viewer samples it around the camera
//!   (`FIELD_SIZE`² cells of `FIELD_CELL`), the distance transform runs off
//!   the frame thread (`Field::build`), and the field travels to the GPU in
//!   the rows of the bathymetry texture under its own 64 (`FIELD_ROW`): no
//!   new binding.
//! - **GPU.** The same functions are `ocean_long` and `ocean_openness` in
//!   water_spectrum.wgsl, read through the shared water uniform's last two
//!   vec4 (`UNIFORM_OFFSET`, after the caustics' extension). Tests keep the
//!   constants and the results equal.
//! - **Depth 0** (the default): nothing here runs, every caller takes its
//!   original path, the uniform's tail stays zero.
// aurora-local-patch: ocean-swell-module

use super::{WaterEffects, WaterGround};
use glam::{UVec2, Vec2, Vec3};
use std::sync::Arc;

/// The slider's range (m): how far the troughs go below the water's level.
pub const DEPTH_MAX: f32 = 5.0;
pub const DEPTH_RANGE: std::ops::RangeInclusive<f32> = 0.0..=DEPTH_MAX;

/// A saved or typed depth within the slider's range (not a number: 0).
pub fn depth(value: f32) -> f32 {
    if value.is_finite() { value.clamp(0.0, DEPTH_MAX) } else { 0.0 }
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
/// far from folding (at 5 m the surface compresses by at most about 20 %).
pub const STEEPNESS: f32 = 0.6;
/// Sets: their length along the wind (m), their speed (m/s, the group speed
/// of the longest wave, half its phase speed) and how much lower the swell
/// is between two sets.
pub const SET_LENGTH: f32 = 860.0;
pub const SET_SPEED: f32 = 7.15;
pub const SET_DEPTH: f32 = 0.25;
/// Openness: distances to land (m) where it starts and where it is full,
/// and the water kept between a trough and the bed (m).
pub const OPEN_FROM: f32 = 100.0;
pub const OPEN_FULL: f32 = 180.0;
pub const KEEL: f32 = 1.0;
const GRAVITY: f32 = 9.81;
const TAU: f32 = std::f32::consts::TAU;

/// The land-distance field: cells a side, their size (m), first row in the
/// bathymetry texture (the bathymetry keeps its 64 x 64 texels), and that
/// texture's size.
pub const FIELD_SIZE: u32 = 128;
pub const FIELD_CELL: f32 = 16.0;
pub const FIELD_ROW: u32 = super::WATER_GROUND_SIZE;
pub const GROUND_WIDTH: u32 = FIELD_SIZE;
pub const GROUND_HEIGHT: u32 = FIELD_ROW + FIELD_SIZE;
/// Terrain this close under the water (m) counts as land.
pub const LAND_MARGIN: f32 = 0.3;
/// Depth given to water with no region under it (m), and the distance of a
/// cell when the field holds no land at all (m).
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

/// Where the ocean's two vec4 start in the shared water uniform: after its
/// own 64 bytes and the caustics' 80 (water_caustic_texture.rs).
pub(super) const UNIFORM_OFFSET: u64 = 144;
/// depth, wind xy, far openness; field origin xy, cell, valid.
type Tail = [f32; 8];

/// The shared water uniform's contents at creation, with the ocean's tail
/// (zero: off).
pub(super) fn initial_uniform(mut bytes: Vec<u8>) -> Vec<u8> {
    bytes.extend_from_slice(bytemuck::bytes_of(&[0.0f32; 8]));
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

/// `ocean_openness` of water_spectrum.wgsl: the openness and its gradient
/// from the four texels around a point (distance to land m, water depth m),
/// `f` its bilinear weights, `cell` the texel size (m), `depth` the slider.
pub fn openness(a: [f32; 2], b: [f32; 2], c: [f32; 2], d: [f32; 2], f: Vec2, cell: f32, depth: f32) -> (f32, Vec2) {
    let [a, b, c, d] = [a, b, c, d].map(Vec2::from_array);
    let v = a.lerp(b, f.x).lerp(c.lerp(d, f.x), f.y);
    let gx = (b - a).lerp(d - c, f.y) / cell;
    let gy = (c.lerp(d, f.x) - a.lerp(b, f.x)) / cell;
    let x = ((v.x - OPEN_FROM) / (OPEN_FULL - OPEN_FROM)).clamp(0.0, 1.0);
    let far = x * x * (3.0 - 2.0 * x);
    let dfar = 6.0 * x * (1.0 - x) / (OPEN_FULL - OPEN_FROM);
    let raw = (v.y - KEEL) / depth.max(0.001);
    let guard = raw.clamp(0.0, 1.0);
    let dguard = if raw > 0.0 && raw < 1.0 { 1.0 / depth.max(0.001) } else { 0.0 };
    let open = far * guard;
    let grad = Vec2::new(gx.x, gy.x) * (dfar * guard) + Vec2::new(gx.y, gy.y) * (dguard * far);
    (open, grad)
}

/// What the viewer samples for one cell of the field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Cell {
    /// Terrain at or above the water, or a region without its terrain.
    Land,
    /// Region water this deep (m).
    Water(f32),
    /// Water with no region under it: open sea.
    Void,
}

/// The land-distance field around the camera: per cell, the distance to the
/// nearest land (m) and the water's depth (m), row by row from `origin`.
#[derive(Debug)]
pub struct Field {
    pub origin: [f32; 2],
    pub texels: Box<[[f32; 2]]>,
}

impl Field {
    /// The field of `cells` (`FIELD_SIZE`², row by row from `origin`); None
    /// when they are not that many. A Euclidean distance transform
    /// (Felzenszwalb and Huttenlocher), rows then columns, in f64; a cell's
    /// distance is to the nearest land cell's centre less half a cell, so
    /// the water next to the shore is about half a cell from it.
    pub fn build(origin: [f32; 2], cells: &[Cell]) -> Option<Field> {
        let n = FIELD_SIZE as usize;
        if cells.len() != n * n {
            return None;
        }
        const NONE: f64 = 1e12;
        let mut squared: Vec<f64> = cells.iter().map(|c| if *c == Cell::Land { 0.0 } else { NONE }).collect();
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
        let texels = cells
            .iter()
            .zip(&squared)
            .map(|(cell, sq)| {
                let distance = if *sq >= NONE * 0.5 {
                    NO_LAND
                } else {
                    ((sq.sqrt() as f32) * FIELD_CELL - FIELD_CELL * 0.5).max(0.0)
                };
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

    fn texel(&self, x: u32, y: u32) -> [f32; 2] {
        self.texels[(y * FIELD_SIZE + x) as usize]
    }

    /// The openness at `xy` and its gradient, as `ocean_open` reads it on
    /// the GPU (the edge texels hold outside the field, without gradient).
    pub fn openness(&self, xy: Vec2, depth: f32) -> (f32, Vec2) {
        if !xy.is_finite() || self.texels.len() != (FIELD_SIZE * FIELD_SIZE) as usize {
            return (0.0, Vec2::ZERO);
        }
        let n = FIELD_SIZE as f32;
        let cell = (xy - Vec2::from_array(self.origin)) / FIELD_CELL - Vec2::splat(0.5);
        let pos = cell.clamp(Vec2::ZERO, Vec2::splat(n - 1.0));
        let i = pos.floor().as_uvec2();
        let j = (i + UVec2::ONE).min(UVec2::splat(FIELD_SIZE - 1));
        let f = pos - pos.floor();
        let (open, grad) = openness(
            self.texel(i.x, i.y),
            self.texel(j.x, i.y),
            self.texel(i.x, j.y),
            self.texel(j.x, j.y),
            f,
            FIELD_CELL,
            depth,
        );
        let inside = Vec2::select(cell.cmpge(Vec2::ZERO) & cell.cmple(Vec2::splat(n - 1.0)), Vec2::ONE, Vec2::ZERO);
        (open, grad * inside)
    }

    /// The mean openness of the field's border: the level the far water
    /// (beyond `FAR_TO`, drawn coarse or as whole tiles) settles at.
    pub fn far_openness(&self, depth: f32) -> f32 {
        if self.texels.len() != (FIELD_SIZE * FIELD_SIZE) as usize {
            return 0.0;
        }
        let last = FIELD_SIZE - 1;
        let mut sum = 0.0;
        let mut count = 0.0;
        for k in 0..FIELD_SIZE {
            for (x, y) in [(k, 0), (k, last), (0, k), (last, k)] {
                let t = self.texel(x, y);
                sum += openness(t, t, t, t, Vec2::ZERO, FIELD_CELL, depth).0;
                count += 1.0;
            }
        }
        sum / count
    }
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

/// The ocean state the viewer hands to the renderer with the bathymetry:
/// the swell's direction, the field, and two values at the camera.
#[derive(Debug, Clone, Default)]
pub struct Ocean {
    /// Unit direction the swell travels ([`wind`]).
    pub wind: [f32; 2],
    pub field: Option<Arc<Field>>,
    /// The swell's drop at the camera (m, 0 or below).
    pub eye_dz: f32,
    /// Its mean level there, -depth / 2 * openness (m): the plane of the
    /// planar reflection and of the underwater window's capture.
    pub eye_mean: f32,
}

impl Ocean {
    /// The field, while the swell is on: enhanced water with its geometric
    /// waves and a depth above 0.
    pub fn active(&self, settings: WaterEffects) -> Option<&Field> {
        if settings.enabled && settings.waves && settings.ocean_depth > 0.0 {
            self.field.as_deref()
        } else {
            None
        }
    }

    fn wind_vec(&self) -> Vec2 {
        let w = Vec2::from_array(self.wind);
        if w.is_finite() && w.length() > 0.5 { w } else { wind(Vec2::ZERO) }
    }

    /// The same state with its values at the camera `eye`.
    pub fn at_eye(mut self, eye: Vec2, time: f32, settings: WaterEffects) -> Self {
        let (dz, mean) = match self.active(settings) {
            Some(field) => (
                long_drop(field, eye, time, self.wind_vec(), settings.ocean_depth),
                -0.5 * settings.ocean_depth * field.openness(eye, settings.ocean_depth).0,
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

/// The long swell's displacement at material position `p`.
fn long_offset(field: &Field, p: Vec2, time: f32, wind: Vec2, depth: f32) -> Vec3 {
    let a = 0.5 * depth * field.openness(p, depth).0;
    let l = long_swell(p, time, wind);
    (l.shift * a).extend(a * (l.s - 1.0))
}

/// The long swell's drop under world `xy`, its horizontal motion inverted.
fn long_drop(field: &Field, xy: Vec2, time: f32, wind: Vec2, depth: f32) -> f32 {
    let mut source = xy;
    for _ in 0..4 {
        source = xy - long_offset(field, source, time, wind, depth).truncate();
    }
    long_offset(field, source, time, wind, depth).z
}

/// The displayed surface's height under world `xy` with the swell on: the
/// existing swell (`super::surface_level`'s, shore-damped the same way) and
/// the long one together, their horizontal motion inverted (four steps: the
/// two together move a point by at most about a fifth of a step's error).
pub fn surface_level(xy: Vec2, time: f32, settings: WaterEffects, level: f32, ground: f32, field: &Field, wind: Vec2) -> f32 {
    if !settings.enabled || !settings.waves {
        return level;
    }
    let t = ((level - ground - 0.08) / 2.32).clamp(0.0, 1.0);
    let shore = t * t * (3.0 - 2.0 * t);
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
        offset * shore
    };
    let sample = |p: Vec2| small(p) + long_offset(field, p, time, wind, settings.ocean_depth);
    let mut source = xy;
    for _ in 0..4 {
        source = xy - sample(source).truncate();
    }
    level + sample(source).z
}

impl WaterGround {
    /// The displayed surface's height under `xy`: `surface_level` as
    /// upstream has it, or with the open-sea swell while it is on.
    pub fn surface(&self, xy: Vec2, time: f32, settings: WaterEffects, level: f32, ground: f32) -> f32 {
        match self.ocean.active(settings) {
            Some(field) => surface_level(xy, time, settings, level, ground, field, self.ocean.wind_vec()),
            None => super::surface_level(xy, time, settings, level, ground),
        }
    }

    /// The swell's own drop under `xy` (0 to -depth); None while it is off.
    pub fn ocean_dz(&self, xy: Vec2, time: f32, settings: WaterEffects) -> Option<f32> {
        let field = self.ocean.active(settings)?;
        xy.is_finite()
            .then(|| long_drop(field, xy, time, self.ocean.wind_vec(), settings.ocean_depth))
    }
}

/// The renderer's side: what was last written to the uniform's tail and
/// which field the texture holds.
#[derive(Default)]
pub(super) struct Gpu {
    written: Tail,
    field: Option<Arc<Field>>,
}

impl super::WaterGpu {
    /// Each frame after the shared uniform: the field into the bathymetry
    /// texture's lower rows when it is new, and the uniform's tail when it
    /// changed (it stays zero, as created, while the swell is off).
    pub(super) fn prepare_ocean(&mut self, queue: &wgpu::Queue, settings: WaterEffects, ground: &WaterGround, valid: bool) {
        let field = ground
            .ocean
            .field
            .as_ref()
            .filter(|_| valid && settings.flags() & 1 != 0 && ground.ocean.active(settings).is_some());
        let wanted: Tail = match field {
            Some(field) => {
                if !self.ocean.field.as_ref().is_some_and(|old| Arc::ptr_eq(old, field)) {
                    self.ocean.field = Some(field.clone());
                    queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &self.ground,
                            mip_level: 0,
                            origin: wgpu::Origin3d { x: 0, y: FIELD_ROW, z: 0 },
                            aspect: wgpu::TextureAspect::All,
                        },
                        bytemuck::cast_slice(field.texels.as_ref()),
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(FIELD_SIZE * 8),
                            rows_per_image: None,
                        },
                        wgpu::Extent3d {
                            width: FIELD_SIZE,
                            height: FIELD_SIZE,
                            depth_or_array_layers: 1,
                        },
                    );
                }
                let w = ground.ocean.wind_vec();
                [
                    settings.ocean_depth,
                    w.x,
                    w.y,
                    field.far_openness(settings.ocean_depth),
                    field.origin[0],
                    field.origin[1],
                    FIELD_CELL,
                    1.0,
                ]
            }
            None => [0.0; 8],
        };
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

    fn on(depth: f32, quality: u8) -> WaterEffects {
        WaterEffects {
            quality,
            ocean_depth: depth,
            ..Default::default()
        }
    }

    fn ground(field: Option<Field>) -> WaterGround {
        WaterGround {
            origin: [-128.0; 2],
            extent: 256.0,
            heights: vec![[0.0, 20.0]; 4096].into(),
            ocean: Ocean {
                wind: wind(Vec2::new(1.05, -0.42)).to_array(),
                field: field.map(Arc::new),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn crests_stay_at_todays_reach_and_troughs_stop_at_the_depth() {
        let g = ground(Some(open_field()));
        // the existing swell's reach: the sum of its amplitudes
        let reach = 0.24 + 0.115 + 0.055 + 0.025 + 1e-4;
        for depth in [0.5, 2.5, 5.0] {
            let settings = on(depth, 3);
            let (mut low, mut high) = (f32::MAX, f32::MIN);
            for i in 0..40 {
                for j in 0..40 {
                    let xy = Vec2::new(i as f32 * 15.3 - 300.0, j as f32 * 14.1 - 280.0);
                    let time = (i * 40 + j) as f32 * 0.37;
                    let level = g.surface(xy, time, settings, 20.0, 0.0) - 20.0;
                    let drop = g.ocean_dz(xy, time, settings).expect("on");
                    assert!(drop <= 1e-5 && drop >= -depth - 1e-4, "{drop} at depth {depth}");
                    assert!(level <= reach && level >= -depth - reach, "{level} at depth {depth}");
                    low = low.min(drop);
                    high = high.max(drop);
                }
            }
            assert!(low < -0.6 * depth, "troughs reach toward the depth: {low}");
            assert!(high > -0.4 * depth, "crests come near the level: {high}");
        }
    }

    #[test]
    fn depth_zero_and_off_switches_keep_upstreams_surface_exactly() {
        let g = ground(Some(open_field()));
        for quality in 0..4 {
            for (x, y, t) in [(43.0, 118.0, 7.3), (-12.0, 3.5, 0.0), (300.0, -77.0, 1234.5)] {
                let xy = Vec2::new(x, y);
                let up = super::super::surface_level(xy, t, on(0.0, quality), 20.0, 13.0);
                assert_eq!(g.surface(xy, t, on(0.0, quality), 20.0, 13.0).to_bits(), up.to_bits());
                assert_eq!(g.ocean_dz(xy, t, on(0.0, quality)), None);
                for off in [
                    WaterEffects {
                        enabled: false,
                        ..on(3.0, quality)
                    },
                    WaterEffects {
                        waves: false,
                        ..on(3.0, quality)
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
        // no field yet: upstream's
        let empty = ground(None);
        let xy = Vec2::new(5.0, 9.0);
        assert_eq!(
            empty.surface(xy, 3.0, on(4.0, 1), 20.0, 0.0),
            super::super::surface_level(xy, 3.0, on(4.0, 1), 20.0, 0.0)
        );
        let o = Ocean::default().at_eye(xy, 3.0, on(4.0, 1));
        assert_eq!((o.eye_dz, o.eye_mean, o.reflection_level(20.0)), (0.0, 0.0, 20.0));
        // the uniform's tail starts zero, after the caustics' extension
        let bytes = initial_uniform(super::super::caustic_texture::initial_uniform());
        assert_eq!(bytes.len() as u64, UNIFORM_OFFSET + 32);
        assert!(bytes[UNIFORM_OFFSET as usize..].iter().all(|b| *b == 0));
    }

    #[test]
    fn a_closed_field_gives_upstreams_swell_and_the_inversion_lands_on_its_point() {
        // all land: openness 0, the existing swell alone (four inversion
        // steps instead of three: equal within a tenth of a millimetre)
        let n = (FIELD_SIZE * FIELD_SIZE) as usize;
        let land = Field::build([-1024.0; 2], &vec![Cell::Land; n]).expect("field");
        for quality in 0..4 {
            let settings = on(5.0, quality);
            for i in 0..30 {
                let xy = Vec2::new(i as f32 * 7.7 - 90.0, i as f32 * 3.1);
                let t = i as f32 * 0.9;
                let ours = surface_level(xy, t, settings, 20.0, 0.0, &land, wind(Vec2::ZERO));
                let up = super::super::surface_level(xy, t, settings, 20.0, 0.0);
                assert!((ours - up).abs() < 5e-4, "{ours} vs {up}");
            }
        }
        // open: the source found by the inversion moves back onto xy
        let field = open_field();
        let w = wind(Vec2::new(0.3, 1.0));
        for i in 0..50 {
            let xy = Vec2::new(i as f32 * 13.0 - 300.0, 40.0 - i as f32 * 5.0);
            let t = i as f32 * 1.7;
            let mut source = xy;
            for _ in 0..4 {
                source = xy - long_offset(&field, source, t, w, 5.0).truncate();
            }
            let back = source + long_offset(&field, source, t, w, 5.0).truncate();
            assert!(back.distance(xy) < 0.01, "{back} vs {xy}");
        }
    }

    #[test]
    fn the_surface_never_folds_at_full_depth() {
        let field = open_field();
        let w = wind(Vec2::ZERO);
        for i in 0..400 {
            let p = Vec2::new(i as f32 * 3.7, i as f32 * -2.3);
            let l = long_swell(p, i as f32 * 0.11, w);
            assert!(l.s.abs() <= 1.0 + 1e-6);
            let a = 0.5 * DEPTH_MAX * field.openness(p, DEPTH_MAX).0;
            let j = Vec3::new(1.0, 0.0, 1.0) - l.fold * a;
            let det = j.x * j.z - j.y * j.y;
            assert!(det > 0.7, "{det}");
        }
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
        // no land at all: open everywhere, full openness
        let void = open_field();
        assert!(void.texels.iter().all(|t| *t == [NO_LAND, VOID_DEPTH]));
        assert_eq!(void.openness(Vec2::new(5.0, -7.0), 5.0), (1.0, Vec2::ZERO));
        assert_eq!(void.far_openness(5.0), 1.0);
        // a wrong cell count is refused
        assert!(Field::build([0.0; 2], &[Cell::Void; 3]).is_none());
    }

    #[test]
    fn openness_fades_in_from_land_and_keeps_the_troughs_off_the_bed() {
        let at = |distance: f32, water: f32, depth: f32| {
            let t = [distance, water];
            openness(t, t, t, t, Vec2::ZERO, FIELD_CELL, depth).0
        };
        assert_eq!(at(0.0, 50.0, 5.0), 0.0);
        assert_eq!(at(OPEN_FROM, 50.0, 5.0), 0.0);
        assert!((at(0.5 * (OPEN_FROM + OPEN_FULL), 50.0, 5.0) - 0.5).abs() < 1e-6);
        assert_eq!(at(OPEN_FULL, 50.0, 5.0), 1.0);
        assert_eq!(at(5000.0, 50.0, 5.0), 1.0);
        // the guard: a trough of depth x openness never comes within KEEL of the bed
        for water in [0.5, 1.0, 2.0, 3.5, 6.0, 40.0] {
            for depth in [0.5, 2.0, 5.0] {
                let o = at(500.0, water, depth);
                assert!(depth * o <= (water - KEEL).max(0.0) + 1e-5, "{water} {depth} {o}");
            }
        }
        // the analytic gradient matches the field's slope inside a texel
        let a = [120.0, 4.0];
        let b = [150.0, 4.6];
        let c = [110.0, 5.5];
        let d = [160.0, 3.0];
        let f = Vec2::new(0.3, 0.6);
        let h = 1e-3;
        let (_, grad) = openness(a, b, c, d, f, FIELD_CELL, 5.0);
        let dx = (openness(a, b, c, d, f + Vec2::X * h, FIELD_CELL, 5.0).0 - openness(a, b, c, d, f - Vec2::X * h, FIELD_CELL, 5.0).0)
            / (2.0 * h * FIELD_CELL);
        let dy = (openness(a, b, c, d, f + Vec2::Y * h, FIELD_CELL, 5.0).0 - openness(a, b, c, d, f - Vec2::Y * h, FIELD_CELL, 5.0).0)
            / (2.0 * h * FIELD_CELL);
        assert!((grad.x - dx).abs() < 1e-3 && (grad.y - dy).abs() < 1e-3, "{grad} vs {dx} {dy}");
        // near land the whole surface is upstream's: nothing extra within OPEN_FROM
        let coast = Field::build([0.0; 2], &cells(|x, _| if x < 40 { Cell::Land } else { Cell::Void })).expect("field");
        let shore = Vec2::new(40.0 * FIELD_CELL + 90.0, 900.0);
        assert_eq!(coast.openness(shore, 5.0).0, 0.0);
        assert!(coast.openness(shore + Vec2::X * 200.0, 5.0).0 > 0.99);
    }

    #[test]
    fn the_wind_follows_the_big_wave_and_falls_back() {
        assert!((wind(Vec2::new(2.0, 0.0)) - Vec2::new(-1.0, 0.0)).length() < 1e-6);
        assert!((wind(Vec2::ZERO) - wind(Vec2::new(1.05, -0.42))).length() < 1e-5);
        assert_eq!(wind(Vec2::new(f32::NAN, 1.0)), wind(Vec2::ZERO));
        assert_eq!(depth(f32::NAN), 0.0);
        assert_eq!(depth(9.0), DEPTH_MAX);
        assert_eq!(depth(-1.0), 0.0);
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
            ("OCEAN_OPEN_FROM", OPEN_FROM),
            ("OCEAN_OPEN_FULL", OPEN_FULL),
            ("OCEAN_KEEL", KEEL),
        ] {
            assert!(spectrum.contains(&format!("const {name}:f32={value:?};")), "{name}");
        }
        let common = include_str!("shaders/common.wgsl");
        for (name, value) in [
            ("OCEAN_FAR_FROM", FAR_FROM),
            ("OCEAN_FAR_TO", FAR_TO),
            ("OCEAN_NORMAL_TO", NORMAL_TO),
            ("OCEAN_DEPTH_MAX", DEPTH_MAX),
        ] {
            assert!(common.contains(&format!("const {name}:f32={value:?};")), "{name}");
        }
        assert!(common.contains(&format!("const OCEAN_FIELD_SIZE:i32={FIELD_SIZE};")));
        assert!(common.contains(&format!("const OCEAN_FIELD_ROW:i32={FIELD_ROW};")));
        // the uniform: the caustics' five vec4 after the shared four, then ours
        assert!(common.contains(
            "counts: vec4<u32>, caustic: vec4<f32>, caustic_rest: array<vec4<f32>,4>, ocean: vec4<f32>, ocean_area: vec4<f32> };"
        ));
    }

    #[test]
    fn cpu_and_gpu_long_swell_and_openness_agree() {
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
    let a=vec2<f32>(80.0+q*6.0,0.5+q*0.3);
    let o=ocean_openness(a,a+vec2<f32>(12.0,0.8),a+vec2<f32>(-9.0,1.5),a+vec2<f32>(5.0,-0.2),fract(vec2<f32>(q*0.37,q*0.61)),16.0,2.5);
    result[id.x*3u]=vec4<f32>(l.s,l.ds,o.x);
    result[id.x*3u+1u]=vec4<f32>(l.shift,o.yz);
    result[id.x*3u+2u]=vec4<f32>(l.fold,0.0);
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
        let bytes = QUERIES * 3 * 16;
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
            let a = [80.0 + q * 6.0, 0.5 + q * 0.3];
            let f = Vec2::new(q * 0.37, q * 0.61);
            let (o, g) = openness(
                a,
                [a[0] + 12.0, a[1] + 0.8],
                [a[0] - 9.0, a[1] + 1.5],
                [a[0] + 5.0, a[1] - 0.2],
                f - f.floor(),
                16.0,
                2.5,
            );
            let cpu = [
                [l.s, l.ds.x, l.ds.y, o],
                [l.shift.x, l.shift.y, g.x, g.y],
                [l.fold.x, l.fold.y, l.fold.z, 0.0],
            ];
            for (row, expected) in cpu.iter().enumerate() {
                let gpu = samples[query * 3 + row];
                for (e, v) in expected.iter().zip(gpu) {
                    assert!((e - v).abs() < 2e-3, "query {query} row {row}: CPU {expected:?} GPU {gpu:?}");
                }
            }
        }
        drop(mapped);
        readback.unmap();
    }
}
