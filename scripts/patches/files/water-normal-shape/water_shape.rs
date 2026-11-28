//! Wave shape from the water normal map (local patch set,
//! scripts/patches/120-water-normal-shape.toml): the environment's water
//! normal map, the texture the water's shading reads, also shapes the small
//! waves' rise and fall, a viewer-local look. Nothing is sent and the
//! environment asset is untouched; the Legacy Viewer's water is flat, so
//! nothing here is ported.
//!
//! - **Height map.** A normal map holds slopes, not heights. When the
//!   region's map changes, its decoded texels (kept on the CPU as a sculpt
//!   map's are) go to the thread pool: their slopes (`-x / z`, `-y / z`)
//!   averaged onto `SIZE`² texels, integrated by a Fourier (Poisson) solve,
//!   H = F⁻¹[−i(kx·Fx + ky·Fy) / (kx² + ky²)], mean 0, and filtered into
//!   `LEVELS` levels by 2 x 2 averages. Heights are in pattern units: a
//!   slope of 1 over the whole pattern gives a height of 1 pattern length.
//!   The levels sit side by side in `ROWS` rows of the bathymetry texture
//!   under the open-sea swell's maps (`ROW`, x of level l: `2 SIZE - (2
//!   SIZE >> l)`): no new binding.
//! - **Sampling.** The same three layers the shading reads
//!   (env.wgsl water_pixel: the big wave and two little ones, their
//!   scales, wave1 / wave2 directions, scroll speeds, the slow x warp and
//!   their weights 0.5, 0.2, 0.3), each `Scale` times larger: a layer of
//!   pattern length L metres is `Scale` · L long and `Scale` · L · weight ·
//!   H high, its slopes those of the map. Each layer reads the level whose
//!   texel is at least `FILTER` vertex spacings of the water's dense mesh
//!   (`spacing`, by quality), trilinear; a layer finer than the last level
//!   is skipped.
//! - **Ratio.** The short waves' vertical movement is mix(geometric waves,
//!   shape, Ratio); their horizontal motion and foam compression scale by
//!   1 - Ratio. The open-sea swell stays on top; the calming and the
//!   shallow-water damping scale the result as they scale the geometric
//!   waves. The shading takes the shape's analytic gradient (the same
//!   texels, the warp's Jacobian included) in place of the waves' gradient
//!   by the same mix.
//! - **GPU.** `water_shape` and `shape_level` of water_spectrum.wgsl, read
//!   through the shape's vec4 of the shared water uniform
//!   (`UNIFORM_OFFSET`, water_uniform_tail.rs): Ratio, Scale, spacing, 1
//!   once the map is in.
//!   Tests keep the constants and the results equal.
//! - **Ratio 0** (the default), waves off or no map yet: nothing here runs,
//!   every caller takes its earlier path, the vec4 stays zero.
//! - **Cross-fade** (patch 998, the viewer's world/eep_fade.rs): while an
//!   environment change fades from one water normal map to another (never
//!   between a day cycle's keyframes), the second map's height map sits in
//!   `ROWS` rows of its own under the first's (`NEXT_ROW`), the two vec4
//!   after the shape's hold its own wave directions and its share, and both
//!   shapes are read and mixed by it (`water_shape_at`, [`Shape::at`]): two
//!   maps that scroll along their own waves have no single mixed map. No
//!   fade: the share is 0, the second shape is not read.
// aurora-local-patch: water-normal-shape-module

use super::WaterEffects;
use glam::{Vec2, Vec3};
use std::sync::Arc;

/// "Wave shape from the normal map": the share of the short waves' height
/// the shape takes.
pub const RATIO_RANGE: std::ops::RangeInclusive<f32> = 0.0..=1.0;
pub const RATIO_DEFAULT: f32 = 0.0;
/// "Normal map scale": how many times the pattern is enlarged.
pub const SCALE_RANGE: std::ops::RangeInclusive<f32> = 1.0..=100.0;
pub const SCALE_DEFAULT: f32 = 1.0;

/// The Ratio within its range (not a number: the default, off).
pub fn ratio(value: f32) -> f32 {
    if value.is_finite() { value.clamp(0.0, 1.0) } else { RATIO_DEFAULT }
}

/// The Scale within its range (not a number: the default).
pub fn scale(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(1.0, 100.0)
    } else {
        SCALE_DEFAULT
    }
}

/// The shape takes part: enhanced water with its geometric waves and a
/// Ratio above 0.
pub fn on(settings: WaterEffects) -> bool {
    settings.enabled && settings.waves && ratio(settings.shape_ratio) > 0.0
}

/// The height map's side (texels) and its levels (the last one a single
/// texel, the mean: 0).
pub const SIZE: u32 = 256;
pub const LEVELS: u32 = 9;
/// The last level, as the shaders compare a level of detail with it.
pub const TOP: f32 = (LEVELS - 1) as f32;
/// A level's texel is at least this many vertex spacings: the pattern
/// moving over the mesh does not shimmer.
pub const FILTER: f32 = 2.0;
/// Where the levels sit in the bathymetry texture: under the open-sea
/// swell's fine map, `ROWS` rows, levels side by side.
pub const ROW: u32 = super::ocean::GROUND_HEIGHT;
pub const ROWS: u32 = SIZE;
/// A cross-fade's second height map (patch 998): under the first one's.
pub const NEXT_ROW: u32 = ROW + ROWS;
pub const BLOCK_WIDTH: u32 = 2 * SIZE;
/// The bathymetry texture's height with the shape's rows.
pub const GROUND_HEIGHT: u32 = NEXT_ROW + ROWS;

/// The layers the shading reads (env.wgsl water_pixel): uv per metre (x,
/// y), scroll speed (uv per second of its half time), weight. The first
/// and the last scroll along wave1, the middle one along wave2.
pub const LAYERS: [[f32; 4]; 3] = [[0.04, 0.04, 0.055, 0.5], [0.45, 0.9, 0.13, 0.2], [0.1, 0.2, 0.1, 0.3]];

/// The dense water grid's cells per 64 m patch by quality (scene/water.rs
/// `grids`): the vertex spacing the levels are chosen by.
const GRID_CELLS: [f32; 4] = [32.0, 48.0, 64.0, 96.0];

/// The dense mesh's vertex spacing (m) at a quality.
pub fn spacing(quality: u8) -> f32 {
    64.0 / GRID_CELLS[quality.min(3) as usize]
}

/// The level of detail a layer of `uv` per metre reads at `spacing` and
/// `scale`: its texel at least `FILTER` vertex spacings, from 0 (finest) to
/// `TOP` (nothing left).
pub fn lod(spacing: f32, uv: Vec2, scale: f32) -> f32 {
    (FILTER * spacing * uv.max_element() * SIZE as f32 / scale).log2().clamp(0.0, TOP)
}

/// Where the shape's vec4 and the cross-fade's two sit in the shared water
/// uniform: their region (water_uniform_tail.rs), zero (off) at creation.
pub(super) const UNIFORM_OFFSET: u64 = super::uniform_tail::SHAPE_OFFSET;

// ------------------------------------------------------------ integration

/// In-place radix-2 FFT of `re` + i `im` (length a power of two); the
/// inverse is not divided by the length.
fn fft(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let angle = sign * std::f64::consts::TAU / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (s, c) = (angle * k as f64).sin_cos();
                let (a, b) = (start + k, start + k + len / 2);
                let (xr, xi) = (re[b] * c - im[b] * s, re[b] * s + im[b] * c);
                re[b] = re[a] - xr;
                im[b] = im[a] - xi;
                re[a] += xr;
                im[a] += xi;
            }
        }
        len <<= 1;
    }
}

/// 2D FFT of an n x n grid (rows of n), rows then columns.
fn fft2(re: &mut [f64], im: &mut [f64], n: usize, inverse: bool) {
    for row in 0..n {
        let r = row * n..(row + 1) * n;
        fft(&mut re[r.clone()], &mut im[r], inverse);
    }
    let (mut cr, mut ci) = (vec![0.0; n], vec![0.0; n]);
    for col in 0..n {
        for (row, (r, i)) in cr.iter_mut().zip(ci.iter_mut()).enumerate() {
            (*r, *i) = (re[row * n + col], im[row * n + col]);
        }
        fft(&mut cr, &mut ci, inverse);
        for (row, (r, i)) in cr.iter().zip(&ci).enumerate() {
            (re[row * n + col], im[row * n + col]) = (*r, *i);
        }
    }
}

/// The tileable height (pattern units, mean 0) whose gradient best fits
/// `slopes` (dh/du, dh/dv per texel, rows of `n`, u along a row): the
/// Poisson solve in Fourier space. The Nyquist terms carry no slope and
/// are dropped.
pub fn integrate(n: usize, slopes: &[[f32; 2]]) -> Option<Vec<f32>> {
    if !n.is_power_of_two() || n < 2 || slopes.len() != n * n || slopes.iter().any(|s| !s[0].is_finite() || !s[1].is_finite()) {
        return None;
    }
    let (mut xr, mut xi) = (slopes.iter().map(|s| s[0] as f64).collect::<Vec<_>>(), vec![0.0; n * n]);
    let (mut yr, mut yi) = (slopes.iter().map(|s| s[1] as f64).collect::<Vec<_>>(), vec![0.0; n * n]);
    fft2(&mut xr, &mut xi, n, false);
    fft2(&mut yr, &mut yi, n, false);
    let signed = |m: usize| if m > n / 2 { m as f64 - n as f64 } else { m as f64 };
    let (mut hr, mut hi) = (vec![0.0; n * n], vec![0.0; n * n]);
    for my in 0..n {
        for mx in 0..n {
            if (mx == 0 && my == 0) || mx == n / 2 || my == n / 2 {
                continue;
            }
            let kx = std::f64::consts::TAU * signed(mx);
            let ky = std::f64::consts::TAU * signed(my);
            let k2 = kx * kx + ky * ky;
            let i = my * n + mx;
            // -i (kx Fx + ky Fy) / k²
            hr[i] = (kx * xi[i] + ky * yi[i]) / k2;
            hi[i] = -(kx * xr[i] + ky * yr[i]) / k2;
        }
    }
    fft2(&mut hr, &mut hi, n, true);
    let norm = (n * n) as f64;
    Some(hr.iter().map(|h| (h / norm) as f32).collect())
}

/// The slope a normal map's texel gives (dh/du, dh/dv): its normal's
/// `-x / z` and `-y / z`, z kept above 0.2 (a slope of 5 at most).
pub fn texel_slope(texel: &[u8]) -> [f32; 2] {
    let n = |c: u8| c as f32 / 127.5 - 1.0;
    let z = n(texel[2]).max(0.2);
    [-n(texel[0]) / z, -n(texel[1]) / z]
}

/// The height map and its levels.
#[derive(Clone, PartialEq)]
pub struct HeightMap {
    levels: Vec<Vec<f32>>,
}

impl std::fmt::Debug for HeightMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HeightMap")
            .field("size", &SIZE)
            .field("levels", &self.levels.len())
            .finish()
    }
}

impl HeightMap {
    /// From `SIZE`² heights (rows): the mean taken out, the levels by 2 x 2
    /// averages.
    pub fn from_heights(mut heights: Vec<f32>) -> Option<HeightMap> {
        let n = SIZE as usize;
        if heights.len() != n * n || heights.iter().any(|h| !h.is_finite()) {
            return None;
        }
        let mean = (heights.iter().map(|h| *h as f64).sum::<f64>() / (n * n) as f64) as f32;
        heights.iter_mut().for_each(|h| *h -= mean);
        let mut levels = vec![heights];
        for level in 1..LEVELS as usize {
            let size = n >> level;
            let above = &levels[level - 1];
            let wide = size * 2;
            let next = (0..size * size)
                .map(|i| {
                    let (x, y) = (i % size * 2, i / size * 2);
                    0.25 * (above[y * wide + x] + above[y * wide + x + 1] + above[(y + 1) * wide + x] + above[(y + 1) * wide + x + 1])
                })
                .collect();
            levels.push(next);
        }
        Some(HeightMap { levels })
    }

    /// From `SIZE`² slopes (rows).
    pub fn from_slopes(slopes: &[[f32; 2]]) -> Option<HeightMap> {
        HeightMap::from_heights(integrate(SIZE as usize, slopes)?)
    }

    /// From a decoded normal map (`components` bytes a texel, rows from the
    /// top, which the water reads at v 0): its slopes averaged onto `SIZE`²
    /// texels when its sides are whole multiples of it, sampled bilinearly
    /// (wrapping) otherwise. None for a map without three channels.
    pub fn from_normal_map(width: u32, height: u32, components: u8, pixels: &[u8]) -> Option<HeightMap> {
        let (w, h, c) = (width as usize, height as usize, components as usize);
        if w == 0 || h == 0 || c < 3 || pixels.len() < w * h * c {
            return None;
        }
        let source: Vec<[f32; 2]> = pixels.chunks_exact(c).take(w * h).map(texel_slope).collect();
        let n = SIZE as usize;
        let slopes: Vec<[f32; 2]> = if w % n == 0 && h % n == 0 {
            let (bx, by) = (w / n, h / n);
            let count = (bx * by) as f32;
            (0..n * n)
                .map(|i| {
                    let (x0, y0) = (i % n * bx, i / n * by);
                    let mut sum = [0.0f32; 2];
                    for y in y0..y0 + by {
                        for s in &source[y * w + x0..y * w + x0 + bx] {
                            sum[0] += s[0];
                            sum[1] += s[1];
                        }
                    }
                    [sum[0] / count, sum[1] / count]
                })
                .collect()
        } else {
            (0..n * n)
                .map(|i| {
                    let x = ((i % n) as f32 + 0.5) * w as f32 / n as f32 - 0.5;
                    let y = ((i / n) as f32 + 0.5) * h as f32 / n as f32 - 0.5;
                    let (fx, fy) = (x - x.floor(), y - y.floor());
                    let wrap = |v: f32, m: usize| (v as i64).rem_euclid(m as i64) as usize;
                    let (x0, y0) = (wrap(x.floor(), w), wrap(y.floor(), h));
                    let (x1, y1) = ((x0 + 1) % w, (y0 + 1) % h);
                    let at = |x: usize, y: usize, k: usize| source[y * w + x][k];
                    let mix = |k| {
                        let top = at(x0, y0, k) + (at(x1, y0, k) - at(x0, y0, k)) * fx;
                        let bottom = at(x0, y1, k) + (at(x1, y1, k) - at(x0, y1, k)) * fx;
                        top + (bottom - top) * fy
                    };
                    [mix(0), mix(1)]
                })
                .collect()
        };
        HeightMap::from_slopes(&slopes)
    }

    /// A level's texel (wrapping).
    pub fn texel(&self, level: usize, x: i32, y: i32) -> f32 {
        let n = (SIZE >> level) as i32;
        self.levels[level][(y.rem_euclid(n) * n + x.rem_euclid(n)) as usize]
    }

    /// `shape_level` of water_spectrum.wgsl: a level's value at `uv`
    /// (bilinear, wrapping) and its gradient per uv.
    pub fn level(&self, level: usize, uv: Vec2) -> Vec3 {
        let n = (SIZE >> level) as f32;
        let pos = (uv - uv.floor()) * n - Vec2::splat(0.5);
        let base = pos.floor();
        let f = pos - base;
        let (i, j) = (base.x as i32, base.y as i32);
        let t = |x, y| self.texel(level, x, y);
        let (v, g) = super::ocean::bilinear(t(i, j), t(i + 1, j), t(i, j + 1), t(i + 1, j + 1), f, 1.0 / n);
        Vec3::new(v, g.x, g.y)
    }

    /// `water_shape` of water_spectrum.wgsl: the shape's height (x, m) and
    /// gradient (yz) at material position `p` and `time` (s), the normal
    /// map's wave directions `waves` (wave1 xy, wave2 zw), `scale` and the
    /// mesh's vertex `spacing` (m).
    pub fn shape(&self, p: Vec2, time: f32, waves: [f32; 4], scale: f32, spacing: f32) -> Vec3 {
        let t = time * 0.5;
        let q = Vec2::new(p.x + ((p.x * 0.08).cos() + (p.y * 0.02).sin()) * 6.0, p.y);
        let j11 = 1.0 - 0.48 * (p.x * 0.08).sin();
        let j12 = 0.12 * (p.y * 0.02).cos();
        let mut out = Vec3::ZERO;
        for (k, layer) in LAYERS.iter().enumerate() {
            let a = Vec2::new(layer[0], layer[1]);
            let lod = lod(spacing, a, scale);
            if lod < TOP {
                let dir = if k == 1 {
                    Vec2::new(waves[2], waves[3])
                } else {
                    Vec2::new(waves[0], waves[1])
                };
                let h = self.layer(k, q, dir * t, scale, lod);
                out += Vec3::new(h.x, h.y * a.x * j11 / scale, (h.y * a.x * j12 + h.z * a.y) / scale);
            }
        }
        out
    }

    /// One layer `k` at warped position `q` (m), scrolled by `travel` (its
    /// direction times the half time, s): its height (x, m) and its
    /// gradient per uv (yz) times its amplitude, read at level of detail
    /// `lod` (trilinear).
    pub fn layer(&self, k: usize, q: Vec2, travel: Vec2, scale: f32, lod: f32) -> Vec3 {
        let layer = LAYERS[k];
        let a = Vec2::new(layer[0], layer[1]);
        let uv = (a * q + travel * layer[2]) / scale;
        let l0 = lod.floor();
        let h0 = self.level(l0 as usize, uv);
        let h = if lod > l0 {
            h0.lerp(self.level(l0 as usize + 1, uv), lod - l0)
        } else {
            h0
        };
        h * (layer[3] * scale / (a.x * a.y).sqrt())
    }

    /// The texels the bathymetry texture's shape rows take: `BLOCK_WIDTH` x
    /// `ROWS`, level l at x `2 SIZE - (2 SIZE >> l)`, the height in x.
    fn block(&self) -> Vec<[f32; 2]> {
        let width = BLOCK_WIDTH as usize;
        let mut texels = vec![[0.0f32; 2]; width * ROWS as usize];
        for (level, values) in self.levels.iter().enumerate() {
            let n = (SIZE >> level) as usize;
            let x0 = width - (width >> level);
            for (i, v) in values.iter().enumerate() {
                texels[(i / n) * width + x0 + i % n][0] = *v;
            }
        }
        texels
    }
}

/// The shape state the viewer hands to the renderer with the bathymetry
/// (`Ocean::shape`): the height map once built and the normal map's wave
/// directions (wave1 xy, wave2 zw).
#[derive(Debug, Clone, Default)]
pub struct Shape {
    pub map: Option<Arc<HeightMap>>,
    pub waves: [f32; 4],
    /// While the environment cross-fades two normal maps (patch 998): the
    /// second one's height map once built, its own wave directions and its
    /// share (0: the first shape alone).
    pub next: Option<Arc<HeightMap>>,
    pub next_waves: [f32; 4],
    pub blend: f32,
}

impl Shape {
    /// The height map, while the shape takes part.
    pub fn active(&self, settings: WaterEffects) -> Option<&HeightMap> {
        if on(settings) { self.map.as_deref() } else { None }
    }

    /// The cross-fade's second height map, while it has a share.
    pub fn fading(&self) -> Option<&HeightMap> {
        self.next.as_deref().filter(|_| self.blend > 0.0)
    }

    /// The shape's height and gradient at `p` (zero while it is off), a
    /// cross-fade's second shape mixed in by its share (`water_shape_at`).
    pub fn at(&self, p: Vec2, time: f32, settings: WaterEffects) -> Vec3 {
        self.active(settings).map_or(Vec3::ZERO, |map| {
            let (scale, spacing) = (scale(settings.shape_scale), spacing(settings.quality));
            let first = map.shape(p, time, self.waves, scale, spacing);
            match self.fading() {
                Some(next) => first.lerp(next.shape(p, time, self.next_waves, scale, spacing), self.blend),
                None => first,
            }
        })
    }

    /// The short waves' displacement `geometric` at `p` with the shape
    /// mixed in by the Ratio (`water_fx_wave_shaped`): unchanged while it is
    /// off.
    pub fn blend(&self, p: Vec2, time: f32, settings: WaterEffects, geometric: Vec3) -> Vec3 {
        if self.active(settings).is_none() {
            return geometric;
        }
        let r = ratio(settings.shape_ratio);
        let h = self.at(p, time, settings).x;
        Vec3::new(geometric.x * (1.0 - r), geometric.y * (1.0 - r), geometric.z * (1.0 - r) + h * r)
    }
}

/// The renderer's side: the three vec4 last written and the maps the
/// texture holds.
#[derive(Default)]
pub(super) struct Gpu {
    written: [f32; 12],
    map: Option<Arc<HeightMap>>,
    next: Option<Arc<HeightMap>>,
}

/// A height map into the bathymetry texture's `ROWS` rows from `row`.
fn write_block(queue: &wgpu::Queue, ground: &wgpu::Texture, row: u32, map: &HeightMap) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: ground,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: row, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&map.block()),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(BLOCK_WIDTH * 8),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width: BLOCK_WIDTH,
            height: ROWS,
            depth_or_array_layers: 1,
        },
    );
}

impl super::WaterGpu {
    /// Each frame after the swell's tail: the height map into the
    /// bathymetry texture's shape rows when it is new, and the vec4 when it
    /// changed (zero, as created, while the shape is off); a cross-fade's
    /// second height map into its own rows, its waves and share into the
    /// next two (zero without a fade).
    pub(super) fn prepare_shape(&mut self, queue: &wgpu::Queue, settings: WaterEffects, ground: &super::WaterGround) {
        let shape = &ground.ocean.shape;
        let map = shape.map.as_ref().filter(|_| on(settings) && settings.flags() & 1 != 0);
        let mut wanted = [0.0f32; 12];
        if let Some(map) = map {
            if !self.shape.map.as_ref().is_some_and(|old| Arc::ptr_eq(old, map)) {
                self.shape.map = Some(map.clone());
                write_block(queue, &self.ground, ROW, map);
            }
            wanted[..4].copy_from_slice(&[
                ratio(settings.shape_ratio),
                scale(settings.shape_scale),
                spacing(settings.quality),
                1.0,
            ]);
            if let Some(next) = shape.next.as_ref().filter(|_| shape.blend > 0.0) {
                if !self.shape.next.as_ref().is_some_and(|old| Arc::ptr_eq(old, next)) {
                    self.shape.next = Some(next.clone());
                    write_block(queue, &self.ground, NEXT_ROW, next);
                }
                wanted[4..8].copy_from_slice(&shape.next_waves);
                wanted[8] = shape.blend.min(1.0);
            }
        }
        if wanted != self.shape.written {
            queue.write_buffer(&self.uniform, UNIFORM_OFFSET, bytemuck::cast_slice(&wanted));
            self.shape.written = wanted;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3Swizzles;

    /// A tileable field of a few sines (pattern units) and its slopes, at
    /// texel centres of an n x n grid.
    const WAVES: [(f32, f32, f32, f32); 4] = [
        (1.0, 0.0, 0.020, 0.3),
        (2.0, 3.0, 0.008, 1.1),
        (-3.0, 1.0, 0.006, 2.0),
        (5.0, -4.0, 0.002, 0.7),
    ];
    fn field(u: f32, v: f32) -> (f32, [f32; 2]) {
        let mut h = 0.0;
        let mut s = [0.0; 2];
        for (a, b, amp, phase) in WAVES {
            let x = std::f32::consts::TAU * (a * u + b * v) + phase;
            h += amp * x.sin();
            s[0] += amp * std::f32::consts::TAU * a * x.cos();
            s[1] += amp * std::f32::consts::TAU * b * x.cos();
        }
        (h, s)
    }
    fn grid(n: usize) -> Vec<(f32, [f32; 2])> {
        (0..n * n)
            .map(|i| field(((i % n) as f32 + 0.5) / n as f32, ((i / n) as f32 + 0.5) / n as f32))
            .collect()
    }
    /// The field as an 8-bit RGB normal map of n x n.
    fn normal_map(n: usize) -> Vec<u8> {
        grid(n)
            .iter()
            .flat_map(|(_, s)| {
                let normal = Vec3::new(-s[0], -s[1], 1.0).normalize();
                normal.to_array().map(|c| ((c + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8)
            })
            .collect()
    }
    fn analytic_map() -> HeightMap {
        HeightMap::from_slopes(&grid(SIZE as usize).iter().map(|(_, s)| *s).collect::<Vec<_>>()).expect("map")
    }
    fn on_settings(ratio: f32, scale: f32) -> WaterEffects {
        WaterEffects {
            quality: 1,
            shape_ratio: ratio,
            shape_scale: scale,
            ..Default::default()
        }
    }

    #[test]
    fn the_integration_gives_back_a_known_height_up_to_its_mean() {
        let n = SIZE as usize;
        let expected = grid(n);
        let mean = expected.iter().map(|(h, _)| h).sum::<f32>() / (n * n) as f32;
        // from exact slopes: the same field
        let map = analytic_map();
        let worst = (0..n * n)
            .map(|i| (map.levels[0][i] - (expected[i].0 - mean)).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 2e-6, "exact slopes: {worst}");
        // from an 8-bit normal map at 512 (averaged down) and at 200
        // (resampled): within a few per cent of the field's 0.036 reach
        for side in [512usize, 200] {
            let map = HeightMap::from_normal_map(side as u32, side as u32, 3, &normal_map(side)).expect("map");
            let worst = (0..n * n)
                .map(|i| (map.levels[0][i] - (expected[i].0 - mean)).abs())
                .fold(0.0f32, f32::max);
            assert!(worst < 1.5e-3, "{side}: {worst}");
        }
        // the mean is 0 and the last level is it
        assert!(map.levels[0].iter().sum::<f32>().abs() < 1e-4);
        assert_eq!(map.levels.len(), LEVELS as usize);
        assert!(map.levels[LEVELS as usize - 1][0].abs() < 1e-7);
        // not a normal map, a wrong length, a bad slope: none
        assert!(HeightMap::from_normal_map(4, 4, 1, &[0; 16]).is_none());
        assert!(HeightMap::from_normal_map(4, 4, 3, &[0; 12]).is_none());
        assert!(integrate(4, &[[f32::NAN, 0.0]; 16]).is_none());
        assert!(integrate(6, &[[0.0; 2]; 36]).is_none());
    }

    #[test]
    fn the_map_tiles_without_a_seam_and_its_gradient_is_the_slope() {
        let map = analytic_map();
        for level in 0..LEVELS as usize - 1 {
            for v in [0.0f32, 0.13, 0.5, 0.77] {
                let at_edge = map.level(level, Vec2::new(0.0, v));
                let across = map.level(level, Vec2::new(1.0, v));
                assert!((at_edge - across).abs().max_element() < 1e-5, "level {level}");
                let before = map.level(level, Vec2::new(-1e-4, v)).x;
                assert!((before - at_edge.x).abs() < 1e-3, "level {level}: {before} {}", at_edge.x);
            }
        }
        // the finest level's gradient follows the field's slope
        for (u, v) in [(0.31f32, 0.62f32), (0.05, 0.9), (0.73, 0.18)] {
            let g = map.level(0, Vec2::new(u, v));
            let (_, s) = field(u, v);
            assert!((g.y - s[0]).abs() < 0.05 && (g.z - s[1]).abs() < 0.05, "{g} {s:?}");
        }
    }

    #[test]
    fn scale_enlarges_the_footprint_and_the_height_alike() {
        let map = analytic_map();
        // each layer at scale S, S times farther and S times later, is the
        // same pattern S times taller with the same slope (gradient per uv
        // over S: the chain rule of `shape`)
        for (k, layer) in LAYERS.iter().enumerate() {
            for (q, travel) in [
                (Vec2::new(3.1, 7.4), Vec2::new(0.3, -0.2)),
                (Vec2::new(-12.0, 0.6), Vec2::new(-1.5, 2.5)),
            ] {
                for lod in [0.0, 2.5] {
                    let one = map.layer(k, q, travel, 1.0, lod);
                    for scale in [1.0f32, 7.5, 100.0] {
                        let big = map.layer(k, q * scale, travel * scale, scale, lod);
                        assert!(
                            (big.x - scale * one.x).abs() < 1e-3 * (scale * one.x).abs().max(1e-3),
                            "height {k} {scale}"
                        );
                        assert!((big.yz() / scale - one.yz()).abs().max_element() < 1e-3 * one.yz().abs().max_element().max(1e-3));
                    }
                }
            }
            // and the pattern repeats every scale / uv metres
            let a = Vec2::new(layer[0], layer[1]);
            for scale in [1.0f32, 100.0] {
                let q = Vec2::new(4.0, 9.0);
                let period = Vec2::splat(scale) / a;
                let here = map.layer(k, q, Vec2::ZERO, scale, 0.0);
                let there = map.layer(k, q + period, Vec2::ZERO, scale, 0.0);
                assert!(
                    (here - there).abs().max_element() < 1e-3 * here.abs().max_element().max(1e-3),
                    "{k} {scale}"
                );
            }
        }
        // the whole shape: about 100 times taller at 100 times the scale
        let waves = [0.6, -0.8, -0.3, 0.95];
        let peak = |scale: f32| {
            (0..400)
                .map(|i| {
                    map.shape(
                        Vec2::new(i as f32 * 0.37 * scale, (i % 20) as f32 * 1.3 * scale),
                        0.0,
                        waves,
                        scale,
                        1e-3,
                    )
                    .x
                    .abs()
                })
                .fold(0.0f32, f32::max)
        };
        let (h1, h100) = (peak(1.0), peak(100.0));
        assert!(h1 > 0.0 && h100 > 50.0 * h1 && h100 < 200.0 * h1, "{h1} {h100}");
    }

    #[test]
    fn the_level_follows_the_vertex_spacing() {
        let big = Vec2::new(0.04, 0.04);
        // a texel of FILTER vertex spacings
        let s = spacing(1);
        assert_eq!(s, 64.0 / 48.0);
        let full = FILTER * s * 0.04 * SIZE as f32;
        assert!(lod(s, big, full).abs() < 1e-5, "level 0 from scale {full}");
        assert!((lod(s, big, full / 4.0) - 2.0).abs() < 1e-5);
        assert!(
            (lod(2.0 * s, big, full / 4.0) - 3.0).abs() < 1e-5,
            "twice the spacing, one level up"
        );
        assert_eq!(lod(s, big, 1000.0), 0.0);
        assert_eq!(lod(1000.0, big, 1.0), TOP);
        // the little waves at scale 1 are finer than any level: skipped
        assert_eq!(lod(spacing(1), Vec2::new(0.45, 0.9), 1.0), TOP);
        assert_eq!((spacing(0), spacing(3)), (2.0, 64.0 / 96.0));
        // a layer past the last level adds nothing
        let map = analytic_map();
        assert_eq!(map.shape(Vec2::new(3.0, 4.0), 1.0, [1.0, 0.0, 0.0, 1.0], 1.0, 1000.0), Vec3::ZERO);
    }

    #[test]
    fn ratio_zero_keeps_the_waves_and_one_takes_the_shape_only() {
        let shape = Shape {
            map: Some(Arc::new(analytic_map())),
            waves: [0.6, -0.8, -0.3, 0.95],
            ..Default::default()
        };
        let geometric = Vec3::new(0.12, -0.07, 0.21);
        let p = Vec2::new(31.0, -12.5);
        let zero = on_settings(0.0, 20.0);
        assert!(!on(zero));
        assert_eq!(shape.blend(p, 4.0, zero, geometric), geometric);
        assert_eq!(shape.at(p, 4.0, zero), Vec3::ZERO);
        let one = on_settings(1.0, 20.0);
        let h = shape.at(p, 4.0, one);
        assert!(h.x != 0.0);
        assert_eq!(shape.blend(p, 4.0, one, geometric), Vec3::new(0.0, 0.0, h.x));
        let half = on_settings(0.5, 20.0);
        let mixed = shape.blend(p, 4.0, half, geometric);
        assert!((mixed - Vec3::new(0.06, -0.035, 0.5 * (0.21 + h.x))).abs().max_element() < 1e-6);
        // the waves off, or no map yet: unchanged
        assert_eq!(shape.blend(p, 4.0, WaterEffects { waves: false, ..one }, geometric), geometric);
        assert_eq!(Shape::default().blend(p, 4.0, one, geometric), geometric);
        // the gradient is the height's: a numeric check at a level-0 scale
        let s = on_settings(1.0, 100.0);
        let e = 1e-2;
        let g = shape.at(p, 2.0, s);
        let dx = (shape.at(p + Vec2::X * e, 2.0, s).x - shape.at(p - Vec2::X * e, 2.0, s).x) / (2.0 * e);
        let dy = (shape.at(p + Vec2::Y * e, 2.0, s).x - shape.at(p - Vec2::Y * e, 2.0, s).x) / (2.0 * e);
        assert!(
            (g.y - dx).abs() < 0.1 * dx.abs().max(0.1) && (g.z - dy).abs() < 0.1 * dy.abs().max(0.1),
            "{g} {dx} {dy}"
        );
    }

    #[test]
    fn the_settings_are_bounded() {
        assert_eq!((ratio(-1.0), ratio(2.0), ratio(f32::NAN), ratio(0.25)), (0.0, 1.0, 0.0, 0.25));
        assert_eq!((scale(0.0), scale(500.0), scale(f32::NAN), scale(20.0)), (1.0, 100.0, 1.0, 20.0));
        let d = WaterEffects::default();
        assert_eq!((d.shape_ratio, d.shape_scale), (RATIO_DEFAULT, SCALE_DEFAULT));
        assert!(!on(d), "off by default");
        assert!(on(on_settings(0.01, 1.0)));
        assert!(!on(WaterEffects {
            enabled: false,
            ..on_settings(1.0, 1.0)
        }));
        // the uniform's vec4 and the cross-fade's two start zero, in their
        // region of the shared uniform (water_uniform_tail.rs)
        let bytes = super::super::uniform_tail::initial_uniform();
        assert_eq!(super::super::uniform_tail::SHAPE_BYTES, 48);
        let ours = &bytes[UNIFORM_OFFSET as usize..(UNIFORM_OFFSET + 48) as usize];
        assert!(ours.iter().all(|b| *b == 0));
        // the levels fit their block, the second map's rows under the first's
        let x_last = BLOCK_WIDTH - (BLOCK_WIDTH >> (LEVELS - 1));
        assert!(x_last < BLOCK_WIDTH && GROUND_HEIGHT == super::super::ocean::GROUND_HEIGHT + 2 * SIZE);
    }

    #[test]
    fn the_shader_constants_are_these() {
        let spectrum = include_str!("shaders/water_spectrum.wgsl");
        assert!(spectrum.contains(&format!("const SHAPE_SIZE:i32={SIZE};")));
        assert!(spectrum.contains(&format!("const SHAPE_TOP:f32={TOP:?};")));
        assert!(spectrum.contains(&format!("const SHAPE_FILTER:f32={FILTER:?};")));
        let layers: Vec<String> = LAYERS
            .iter()
            .map(|l| format!("vec4<f32>({:?},{:?},{:?},{:?})", l[0], l[1], l[2], l[3]))
            .collect();
        assert!(spectrum.contains(&format!("const SHAPE_LAYERS=array<vec4<f32>,3>({});", layers.join(","))));
        let common = include_str!("shaders/common.wgsl");
        assert!(common.contains(&format!("const SHAPE_ROW:i32={ROW};")));
        assert!(common.contains(&format!("const SHAPE_NEXT_ROW:i32={NEXT_ROW};")));
        assert!(
            common.contains(
                "ocean_shape: vec4<f32>, ocean_fine: vec4<f32>, ocean_next: vec4<f32>, shape: vec4<f32>, shape_next: vec4<f32>, shape_fade: vec4<f32> };"
            )
        );
    }

    #[test]
    fn a_cross_fade_mixes_the_two_shapes_by_its_share() {
        let first = Arc::new(analytic_map());
        // the second map: the same field, half as high and moving another way
        let half = HeightMap::from_slopes(
            &grid(SIZE as usize)
                .iter()
                .map(|(_, s)| [s[0] * 0.5, s[1] * 0.5])
                .collect::<Vec<_>>(),
        );
        let next_waves = [-0.2, 0.9, 0.7, 0.1];
        let one = on_settings(1.0, 20.0);
        let alone = Shape {
            map: Some(first.clone()),
            waves: [0.6, -0.8, -0.3, 0.95],
            ..Default::default()
        };
        let fading = |blend: f32| Shape {
            next: half.clone().map(Arc::new),
            next_waves,
            blend,
            ..alone.clone()
        };
        let second = Shape {
            map: half.clone().map(Arc::new),
            waves: next_waves,
            ..Default::default()
        };
        for (x, y, t) in [(43.0, 118.0, 7.3), (-12.0, 3.5, 0.0), (300.0, -77.0, 1234.5)] {
            let p = Vec2::new(x, y);
            let (a, b) = (alone.at(p, t, one), second.at(p, t, one));
            // no share: the first shape exactly; all of it: the second's
            assert_eq!(fading(0.0).at(p, t, one), a);
            assert!((fading(1.0).at(p, t, one) - b).abs().max_element() < 1e-4 * b.abs().max_element().max(1.0));
            assert_eq!(fading(0.25).at(p, t, one), a.lerp(b, 0.25));
            assert_ne!(a, b, "two different shapes");
            // not made yet: the first alone, whatever the share
            let waiting = Shape { next: None, ..fading(0.5) };
            assert_eq!(waiting.at(p, t, one), a);
            // the shape off: nothing, fade or not
            assert_eq!(fading(0.5).at(p, t, on_settings(0.0, 20.0)), Vec3::ZERO);
        }
        assert!(fading(0.0).fading().is_none() && fading(0.3).fading().is_some());
    }

    #[test]
    fn cpu_and_gpu_shape_agree() {
        use crate::pipeline_cache::CachedPipelines;
        let _gpu = crate::test_gpu::lock();
        let instance = wgpu::Instance::default();
        let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
            return;
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).expect("GPU device");
        let map = analytic_map();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shape probe"),
            size: wgpu::Extent3d {
                width: BLOCK_WIDTH,
                height: ROWS,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            texture.as_image_copy(),
            bytemuck::cast_slice(&map.block()),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(BLOCK_WIDTH * 8),
                rows_per_image: None,
            },
            texture.size(),
        );
        let source = format!(
            "{}\n{}",
            include_str!("shaders/water_spectrum.wgsl"),
            r#"
@group(0) @binding(0) var<storage,read_write> result:array<vec4<f32>>;
@group(0) @binding(1) var map:texture_2d<f32>;
@compute @workgroup_size(1)
fn probe(@builtin(global_invocation_id) id:vec3<u32>) {
    let q=f32(id.x);
    let p=vec2<f32>(q*13.7-150.0,40.0-q*9.1);
    let time=3.0+q*1.3;
    let waves=vec4<f32>(0.6,-0.8,-0.3,0.95);
    let scale=1.0+q*4.1;
    let spacing=select(0.29,2.0,(id.x&1u)==1u);
    result[id.x]=vec4<f32>(water_shape(map,0,p,time,waves,scale,spacing),0.0);
}
"#
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CPU / rendered water shape agreement"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline_cached(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });
        const QUERIES: u64 = 24;
        let bytes = QUERIES * 16;
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
        let view = texture.create_view(&Default::default());
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
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
        readback.map_async(wgpu::MapMode::Read, ..bytes, |r| r.expect("shape readback"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("GPU completion");
        let mapped = readback.get_mapped_range(..bytes).expect("mapped shape");
        let samples = crate::readback::f32x4s(&mapped);
        for (query, gpu) in samples.iter().enumerate().take(QUERIES as usize) {
            let q = query as f32;
            let p = Vec2::new(q * 13.7 - 150.0, 40.0 - q * 9.1);
            let spacing = if query % 2 == 1 { 2.0 } else { 0.29 };
            let cpu = map.shape(p, 3.0 + q * 1.3, [0.6, -0.8, -0.3, 0.95], 1.0 + q * 4.1, spacing);
            for (e, v) in cpu.to_array().iter().zip(gpu) {
                assert!((e - v).abs() < 2e-3 * e.abs().max(0.05), "query {query}: CPU {cpu} GPU {gpu:?}");
            }
        }
        drop(mapped);
        readback.unmap();
    }
}
