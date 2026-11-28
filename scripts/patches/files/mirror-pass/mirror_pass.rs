// aurora-local-patch: mirror-pass-module
//! The planar mirror's pass made cheaper, with no change to a mirror on the
//! screen (local patch set).
//!
//! The pass (`refl_passes` in renderer.rs) draws the scene again from the
//! mirror's camera, into an image up to the world's size, for one reader:
//! `mirror_tap` (common.wgsl) gives that image to the surfaces inside the
//! probe's box, in front of the mirror's plane or at most [`TAP_BEHIND`]
//! behind it, glossier than 0.75; every other surface takes the probes.
//! Nearly all of its cost follows the pixels it shades. What it no longer
//! does:
//!
//! - shade what lies behind the mirror: the fragments its clip plane drops
//!   leave before their shading (`reflection_behind`, common.wgsl), and the
//!   draws wholly behind that plane are culled, in the GPU bin as in the CPU
//!   lists, the blended one included ([`cull_planes`]);
//! - run when nothing on the screen can show the image: the volume of those
//!   surfaces is off the screen, or thinner than [`TINY_PX`] in the user's
//!   view (then the frame is one without a mirror, the probes show), or
//!   hidden behind the main view's depth (then the pass draws nothing:
//!   [`MirrorPass::encode_test`], in the same frame, with no wait);
//! - draw what the screen cannot read: its scissor keeps that volume's
//!   screen rect and the fade band mirror_tap gives the rays that leave the
//!   image ([`TAP_FADE`]), its culling takes that sub-frustum, and
//!   mirror_tap fades the rays that leave the rect as those that leave the
//!   image.
//!
//! A surface flat on the mirror reads its own point of the screen, inside
//! that rect: for it nothing changes. A glossy surface inside the volume
//! that is not parallel to the mirror (a normal map, a curved object) can
//! reflect a ray that leaves the rect: that ray fades to the probes, as a ray
//! leaving the image always did.
//!
//! The Legacy Viewer leaves its hero probe's faces alone while the probe is
//! occluded (LLHeroProbeManager::update, indra/newview/llheroprobemanager.cpp:
//! `!mProbes[0]->mOccluded`), which the occlusion query of a cube around the
//! probe's bounding sphere tells a frame or more late, never with the eye
//! inside it (LLReflectionMap::doOcclusion, indra/newview/llreflectionmap.cpp);
//! the probe keeps what it drew last. Here the test is of the volume that
//! can show the image, in the frame it draws, and a mirror left out shows
//! the probes.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec4};

use crate::pipeline_cache::CachedPipelines;
use crate::types::{DrawIndexedIndirect, MirrorParams};

/// What the mirror pass keeps behind the mirror's plane (m): the clip plane
/// of its frame (refl_frame's `keep_offset`), and so of its culling.
pub const KEEP_BEHIND: f32 = 0.01;
/// The culling keeps a millimetre more than the clip: a draw it leaves out
/// has no fragment the clip keeps, whatever the rounding of its bounds.
const CULL_SLACK: f32 = 1e-3;
/// mirror_tap gives the mirror image to surfaces up to this far behind the
/// plane (m): its `clip`, (n·p + d) · 9.5 + 0.5, is above zero there.
pub const TAP_BEHIND: f32 = 0.5 / 9.5;
/// The band, a share of the image, over which mirror_tap fades the rays
/// that leave the image (`out * 20.0`), and now those that leave the rect.
pub const TAP_FADE: f32 = 0.05;
/// Shorter side, in world pixels, of the volume's rect on the screen under
/// which the user's view leaves the mirror out: a surface that thin shows a
/// pixel or two of reflection, where the mirror image and the probes cannot
/// be told apart, the less so under TAA's jitter of up to a pixel.
pub const TINY_PX: f32 = 2.0;
/// A mirror that became tiny is still drawn this long after the last frame
/// it was not: no flicker at the threshold. One that grows past it is drawn
/// at once.
pub const TINY_HOLD: Duration = Duration::from_millis(500);
/// What stands in front of the mirror keeps its textures for it this long
/// after the mirror was last drawn and not found hidden ([`Shown`]).
pub const SHOWN_HOLD: Duration = Duration::from_secs(1);
/// A side of mirror_tap's rect that the image's edge bounds already.
pub const OPEN: f32 = 1e4;
/// mirror_tap's rect with every side open: as without the patch.
pub const OPEN_RECT: [f32; 4] = [-OPEN, -OPEN, OPEN, OPEN];
/// A point at or behind the camera plane leaves the projection unbounded
/// (hiz_sphere_visible keeps such a sphere the same way).
const MIN_W: f32 = 1e-3;
/// `Test::bin` without the GPU culling.
const NO_BIN: u32 = u32::MAX;

/// The corners of the probe's box on the kept side of mirror_tap's cut,
/// and the points where the box's edges cross the cut: the convex hull of
/// every point where mirror_tap can give the mirror image.
#[derive(Debug, Clone, Copy)]
pub struct Volume {
    points: [Vec3; 20],
    len: usize,
}

/// The twelve edges of a box whose corner `k` has its x, y, z signs in the
/// bits 0, 1, 2 of `k`.
const EDGES: [(usize, usize); 12] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

/// A volume seen through a view-projection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Projected {
    /// Nothing where mirror_tap gives the image.
    Empty,
    /// A point at or behind the camera plane (the camera within or beside
    /// the volume): no rect bounds it.
    Unbounded,
    Rect(ScreenRect),
}

/// A rect of the screen in mirror_tap's coordinates (uv, y down) and the
/// nearest depth inside it (reverse-Z: the largest).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenRect {
    pub lo: Vec2,
    pub hi: Vec2,
    pub nearest: f32,
}

impl Volume {
    /// `world_to_box`: MirrorParams::world_to_box ([-0.5, 0.5]^3 inside);
    /// `plane`: the mirror's n.xyz and d (n·p + d = 0), n of unit length.
    pub fn new(world_to_box: Mat4, plane: Vec4) -> Volume {
        let to_world = world_to_box.inverse();
        let n = plane.truncate();
        let corner = |k: usize| {
            let sign = |bit: usize| if k & bit != 0 { 0.5 } else { -0.5 };
            to_world.transform_point3(Vec3::new(sign(1), sign(2), sign(4)))
        };
        let corners: [Vec3; 8] = std::array::from_fn(corner);
        // the cut: n·p + d + TAP_BEHIND >= 0 keeps the point
        let side = corners.map(|c| n.dot(c) + plane.w + TAP_BEHIND);
        let mut v = Volume {
            points: [Vec3::ZERO; 20],
            len: 0,
        };
        for (c, s) in corners.iter().zip(side) {
            if s >= 0.0 {
                v.push(*c);
            }
        }
        for (a, b) in EDGES {
            if (side[a] < 0.0) != (side[b] < 0.0) {
                v.push(corners[a].lerp(corners[b], side[a] / (side[a] - side[b])));
            }
        }
        v
    }

    fn push(&mut self, p: Vec3) {
        if self.len < self.points.len() {
            self.points[self.len] = p;
            self.len += 1;
        }
    }

    pub fn points(&self) -> &[Vec3] {
        &self.points[..self.len]
    }

    /// The volume's rect through `view_proj` (a perspective: the projection
    /// of a convex volume in front of the camera plane is inside the rect of
    /// its points' projections).
    pub fn project(&self, view_proj: Mat4) -> Projected {
        if self.len == 0 {
            return Projected::Empty;
        }
        let mut lo = Vec2::splat(f32::INFINITY);
        let mut hi = Vec2::splat(f32::NEG_INFINITY);
        let mut nearest = f32::NEG_INFINITY;
        for p in self.points() {
            let c = view_proj * p.extend(1.0);
            // (a bad matrix too: when in doubt, the whole image)
            if c.w.is_nan() || c.w <= MIN_W {
                return Projected::Unbounded;
            }
            let ndc = c.truncate() / c.w;
            let uv = Vec2::new(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
            if !uv.is_finite() || !ndc.z.is_finite() {
                return Projected::Unbounded;
            }
            lo = lo.min(uv);
            hi = hi.max(uv);
            nearest = nearest.max(ndc.z);
        }
        Projected::Rect(ScreenRect { lo, hi, nearest })
    }
}

/// Why the mirror found for the frame was left out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirrorSkip {
    /// The volume where mirror_tap gives its image is off the screen.
    OffScreen,
    /// That volume is less than TINY_PX across on the screen (the user's
    /// view only), for TINY_HOLD.
    Tiny,
}

/// What the mirror's pass does this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plan {
    /// Left out: the frame is then one without a mirror.
    pub skip: Option<MirrorSkip>,
    /// The pass's scissor in texels of the mirror image (x, y, width,
    /// height); None: the whole image.
    pub scissor: Option<[u32; 4]>,
    /// mirror_tap's rect (uv: lo.x, lo.y, hi.x, hi.y; OPEN on the sides the
    /// image's edge bounds).
    pub rect: [f32; 4],
    /// From the pass's clip space to the scissor's: its sub-frustum's sides
    /// for the culling ([`cull_planes`]).
    pub narrow: Mat4,
    /// The occlusion test's rect on the main view's depth.
    pub test: Option<ScreenRect>,
    /// Share of the mirror image inside the scissor.
    pub area: f32,
}

impl Plan {
    /// As before the patch: the whole image, no occlusion test.
    pub const FULL: Plan = Plan {
        skip: None,
        scissor: None,
        rect: OPEN_RECT,
        narrow: Mat4::IDENTITY,
        test: None,
        area: 1.0,
    };

    fn skipped(why: MirrorSkip) -> Plan {
        Plan {
            skip: Some(why),
            area: 0.0,
            ..Plan::FULL
        }
    }

    pub fn drawn(&self) -> bool {
        self.skip.is_none()
    }
}

/// The mirror's plan from frame to frame: only the tiny mirror's hold.
#[derive(Debug, Default, Clone, Copy)]
pub struct Planner {
    /// The last frame of the user's view whose mirror was not tiny.
    wanted: Option<Instant>,
}

impl Planner {
    /// This frame's plan for the mirror `m`, through the main view's
    /// `unjittered` view-projection (mirror_tap's and the pass's) and the
    /// `jittered` one (the depth's), the world's size in pixels and the
    /// mirror image's in texels. `users_view`: the main view is the user's
    /// camera (not a photo at its own size, a 360° face nor an external
    /// output), the only one that leaves a tiny mirror out.
    #[allow(clippy::too_many_arguments)]
    pub fn plan(
        &mut self,
        m: &MirrorParams,
        unjittered: Mat4,
        jittered: Mat4,
        world: (u32, u32),
        image: (u32, u32),
        users_view: bool,
        now: Instant,
    ) -> Plan {
        let n = m.normal.normalize_or(Vec3::Z);
        let volume = Volume::new(m.world_to_box, n.extend(-n.dot(m.point)));
        let r = match volume.project(unjittered) {
            Projected::Rect(r) => r,
            // (never with the plane through the box's centre)
            Projected::Empty => return Plan::skipped(MirrorSkip::OffScreen),
            Projected::Unbounded => {
                if users_view {
                    self.wanted = Some(now);
                }
                return Plan::FULL;
            }
        };
        let px = Vec2::new(world.0.max(1) as f32, world.1.max(1) as f32);
        // two pixels of margin: TAA's jitter (under one) and the rasterization
        let margin = 2.0 / px;
        if r.hi.x < -margin.x || r.hi.y < -margin.y || r.lo.x > 1.0 + margin.x || r.lo.y > 1.0 + margin.y {
            return Plan::skipped(MirrorSkip::OffScreen);
        }
        if users_view {
            let seen = (r.hi.min(Vec2::ONE) - r.lo.max(Vec2::ZERO)).max(Vec2::ZERO) * px;
            if seen.min_element() >= TINY_PX {
                self.wanted = Some(now);
            } else if !self.wanted.is_some_and(|t| now.saturating_duration_since(t) < TINY_HOLD) {
                return Plan::skipped(MirrorSkip::Tiny);
            }
        }
        let (scissor, rect, area) = scissor(&r, image);
        let test = match volume.project(jittered) {
            Projected::Rect(t) => Some(t),
            _ => None,
        };
        Plan {
            skip: None,
            scissor,
            rect,
            narrow: scissor.map_or(Mat4::IDENTITY, |s| narrow(s, image)),
            test,
            area,
        }
    }
}

/// The pass's scissor (texels), mirror_tap's rect and the scissor's share
/// of the image, for the volume's rect `r` on an image of `size` texels.
/// Every read mirror_tap gives a weight lands inside the scissor: inside the
/// rect grown by a texel (rounding), or in the fade band TAP_FADE past it,
/// with one texel more for the bilinear filter and one for the rounding of
/// the scissor's edges. A side whose scissor edge reaches the image's edge is
/// open: what lies past the rect on that side is drawn, as before.
pub fn scissor(r: &ScreenRect, size: (u32, u32)) -> (Option<[u32; 4]>, [f32; 4], f32) {
    let (w, h) = (size.0.max(1) as f32, size.1.max(1) as f32);
    let texel = Vec2::new(1.0 / w, 1.0 / h);
    let lo = r.lo - texel;
    let hi = r.hi + texel;
    let x0 = ((lo.x - TAP_FADE) * w).floor() - 1.0;
    let y0 = ((lo.y - TAP_FADE) * h).floor() - 1.0;
    let x1 = ((hi.x + TAP_FADE) * w).ceil() + 1.0;
    let y1 = ((hi.y + TAP_FADE) * h).ceil() + 1.0;
    let open = [x0 <= 0.0, y0 <= 0.0, x1 >= w, y1 >= h];
    let (sx0, sy0) = (x0.clamp(0.0, w), y0.clamp(0.0, h));
    let (sx1, sy1) = (x1.clamp(0.0, w), y1.clamp(0.0, h));
    if open.iter().all(|o| *o) || sx1 <= sx0 || sy1 <= sy0 {
        return (None, OPEN_RECT, 1.0);
    }
    let rect = [
        if open[0] { -OPEN } else { lo.x },
        if open[1] { -OPEN } else { lo.y },
        if open[2] { OPEN } else { hi.x },
        if open[3] { OPEN } else { hi.y },
    ];
    let s = [sx0 as u32, sy0 as u32, (sx1 - sx0) as u32, (sy1 - sy0) as u32];
    (Some(s), rect, (s[2] as f32 * s[3] as f32) / (w * h))
}

/// From the pass's clip space to that of the scissor `s` of an image of
/// `size` texels: the scissor's rect becomes the whole clip square, depth
/// and w untouched.
pub fn narrow(s: [u32; 4], size: (u32, u32)) -> Mat4 {
    let (w, h) = (size.0.max(1) as f32, size.1.max(1) as f32);
    let x0 = 2.0 * s[0] as f32 / w - 1.0;
    let x1 = 2.0 * (s[0] + s[2]) as f32 / w - 1.0;
    // texel rows go down, clip y up
    let y0 = 1.0 - 2.0 * (s[1] + s[3]) as f32 / h;
    let y1 = 1.0 - 2.0 * s[1] as f32 / h;
    let (cx, hx) = ((x0 + x1) * 0.5, ((x1 - x0) * 0.5).max(1e-6));
    let (cy, hy) = ((y0 + y1) * 0.5, ((y1 - y0) * 0.5).max(1e-6));
    Mat4::from_cols(
        Vec4::new(1.0 / hx, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0 / hy, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        Vec4::new(-cx / hx, -cy / hy, 0.0, 1.0),
    )
}

/// The mirror pass's culling planes, for the GPU bin (CullFrame::mirror)
/// and the CPU lists alike: the sides and near plane of its view
/// (`view_proj`, refl_frame's) narrowed to the scissor (`narrow`), and its
/// clip plane in place of the sixth, which accepts everything in a
/// reverse-Z infinite view: a draw whose sphere lies wholly behind the
/// mirror's `plane` (n.xyz of unit length facing the camera, d) by more than
/// KEEP_BEHIND has no fragment `reflection_clip` keeps; one across it stays.
pub fn cull_planes(view_proj: Mat4, narrow: Mat4, plane: Vec4) -> [[f32; 4]; 6] {
    let mut planes = crate::gpu_cull::view_planes(narrow * view_proj).map(|p| p.to_array());
    planes[5] = [plane.x, plane.y, plane.z, plane.w + KEEP_BEHIND + CULL_SLACK];
    planes
}

/// What became of the mirror this frame (RenderStats::mirror_pass).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MirrorStats {
    /// A mirror was asked for, with an image to draw into.
    pub found: bool,
    /// Why it was left out, if it was (no pass then).
    pub skip: Option<MirrorSkip>,
    /// Its pass ran the occlusion test.
    pub tested: bool,
    /// Verdicts of earlier frames' tests read back since the last frame,
    /// and those that found the mirror hidden.
    pub verdicts: u32,
    pub hidden: u32,
    /// The newest verdict of the tests running since the last frame without
    /// one found it hidden.
    pub hidden_last: bool,
    /// Draws of the pass: its CPU-built lists and its GPU bin (read back a
    /// frame or two late).
    pub draws: u32,
    /// Share of the mirror image inside the pass's scissor.
    pub area: f32,
}

impl MirrorStats {
    /// The renderer left the mirror out, or last found it hidden: what
    /// stands in front of it no longer needs its textures for it. A frame
    /// that found no mirror says nothing.
    pub fn left_out(&self) -> bool {
        self.found && (self.skip.is_some() || self.hidden_last)
    }

    /// For a frame with no result of its own (main_thread.rs): the state
    /// carried, the verdicts not counted twice.
    pub fn carried(&self) -> MirrorStats {
        MirrorStats {
            verdicts: 0,
            hidden: 0,
            ..*self
        }
    }
}

/// Whether the mirror was drawn and seen lately, for the occlusion
/// streaming's exemption of what stands in front of it (scene/mod.rs): from
/// the frame it is, until SHOWN_HOLD after the last such frame.
#[derive(Debug, Default, Clone, Copy)]
pub struct Shown {
    last: Option<Instant>,
    exempt: bool,
}

impl Shown {
    pub fn note(&mut self, shown: bool, now: Instant) {
        if shown {
            self.last = Some(now);
        }
        self.exempt = self.last.is_some_and(|t| now.saturating_duration_since(t) < SHOWN_HOLD);
    }

    pub fn exempt(&self) -> bool {
        self.exempt
    }
}

/// The occlusion test's verdicts, from frame to frame.
#[derive(Debug, Default, Clone, Copy)]
struct Verdicts {
    /// Frames counted, and the last one without a test.
    frame: u64,
    untested: u64,
    /// The newest verdict read back (its frame, hidden).
    newest: Option<(u64, bool)>,
    /// Read back since the last frame, and those that found it hidden.
    read: u32,
    hidden: u32,
}

impl Verdicts {
    /// The verdict of the test of `frame`.
    fn take(&mut self, frame: u64, hidden: bool) {
        self.read += 1;
        self.hidden += u32::from(hidden);
        if frame > self.untested && self.newest.is_none_or(|(f, _)| frame > f) {
            self.newest = Some((frame, hidden));
        }
    }

    /// End of a frame: (read, hidden, newest hidden). A frame without a
    /// test forgets the verdicts of before it.
    fn end_frame(&mut self, tested: bool) -> (u32, u32, bool) {
        if !tested {
            self.untested = self.frame;
            self.newest = None;
        }
        let read = std::mem::take(&mut self.read);
        let hidden = std::mem::take(&mut self.hidden);
        (read, hidden, self.newest.is_some_and(|(_, h)| h))
    }
}

/// The occlusion test's parameters (mirror_test.wgsl `Test`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct TestParams {
    rect: [f32; 4],
    screen: [f32; 2],
    nearest: f32,
    bin: u32,
    ranges: [[u32; 4]; 3],
}

struct Bound {
    indirect: wgpu::Buffer,
    counts: wgpu::Buffer,
    hiz: wgpu::TextureView,
    group: wgpu::BindGroup,
}

struct Slot {
    buffer: wgpu::Buffer,
    ready: Arc<AtomicBool>,
    pending: bool,
    frame: u64,
}

/// The mirror pass's plan, occlusion test and statistics (renderer.rs).
pub struct MirrorPass {
    planner: Planner,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    params: wgpu::Buffer,
    /// The sky's draw (vertex count, instances, firsts): the test gives it
    /// one instance or none, and its copy is the verdict read back.
    sky: wgpu::Buffer,
    /// Stands for the GPU bins' counts without the GPU culling.
    no_counts: wgpu::Buffer,
    /// Made again only when the draws, the counts or the pyramid change.
    bound: Option<Bound>,
    readback: [Slot; 3],
    copied: Option<usize>,
    verdicts: Verdicts,
    /// This frame's pass ran the test: its sky draws from `sky`.
    tested: bool,
}

impl MirrorPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mirror test"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mirror_test.wgsl").into()),
        });
        let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let buffer = |ty: wgpu::BufferBindingType| wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let storage = buffer(wgpu::BufferBindingType::Storage { read_only: false });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mirror test"),
            entries: &[
                entry(0, buffer(wgpu::BufferBindingType::Uniform)),
                entry(
                    1,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                entry(2, storage),
                entry(3, storage),
                entry(4, storage),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mirror test"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline_cached(&wgpu::ComputePipelineDescriptor {
            label: Some("mirror test"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cs_mirror_test"),
            compilation_options: Default::default(),
            cache: None,
        });
        let params = crate::gpu_alloc::buffer_init(
            device,
            "mirror test",
            bytemuck::bytes_of(&TestParams::zeroed()),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let sky = crate::gpu_alloc::buffer_init(
            device,
            "mirror sky draw",
            bytemuck::cast_slice(&[3u32, 1, 0, 0]),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC,
        );
        let no_counts = crate::gpu_alloc::buffer_init(
            device,
            "mirror test counts",
            bytemuck::cast_slice(&[0u32; 4]),
            wgpu::BufferUsages::STORAGE,
        );
        let readback = std::array::from_fn(|_| Slot {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mirror test readback"),
                size: crate::readback::readback_size(16),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            ready: Arc::new(AtomicBool::new(false)),
            pending: false,
            frame: 0,
        });
        MirrorPass {
            planner: Planner::default(),
            layout,
            pipeline,
            params,
            sky,
            no_counts,
            bound: None,
            readback,
            copied: None,
            verdicts: Verdicts::default(),
            tested: false,
        }
    }

    /// Start of a frame: the verdicts read back since the last.
    pub fn harvest(&mut self) {
        self.verdicts.frame += 1;
        self.tested = false;
        for s in &mut self.readback {
            if !s.pending || !s.ready.load(Ordering::Acquire) {
                continue;
            }
            if let Ok(view) = s.buffer.get_mapped_range(..)
                && let Some(bytes) = view.get(..16)
            {
                let sky = crate::readback::u32_array::<4>(bytes);
                self.verdicts.take(s.frame, sky[1] == 0);
            }
            s.buffer.unmap();
            s.pending = false;
            s.ready.store(false, Ordering::Release);
        }
    }

    /// This frame's plan for the mirror (Planner::plan).
    #[allow(clippy::too_many_arguments)]
    pub fn plan(
        &mut self,
        m: &MirrorParams,
        unjittered: Mat4,
        jittered: Mat4,
        world: (u32, u32),
        image: (u32, u32),
        users_view: bool,
        now: Instant,
    ) -> Plan {
        self.planner.plan(m, unjittered, jittered, world, image, users_view, now)
    }

    /// The occlusion test, after the Hi-Z pyramid `hiz` of the main view's
    /// phase 1 depth (of the world's size `world`), before anything reads
    /// the pass's draws: `rect`, Plan::test; `indirect`, the frame's
    /// CPU-built draws, `lists` the pass's ones in it (byte offset, count);
    /// `bin`, the GPU bins' counts and the pass's bin, with the GPU culling.
    /// Hidden, they all get no instance, its sky too ([`Self::sky_draw`]).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_test(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        rect: &ScreenRect,
        hiz: &wgpu::TextureView,
        world: (u32, u32),
        indirect: &wgpu::Buffer,
        bin: Option<(&wgpu::Buffer, usize)>,
        lists: [(u64, u32); 5],
    ) {
        let stride = std::mem::size_of::<DrawIndexedIndirect>() as u64;
        let first = |k: usize| (lists[k].0 / stride) as u32;
        let params = TestParams {
            rect: [rect.lo.x, rect.lo.y, rect.hi.x, rect.hi.y],
            screen: [world.0 as f32, world.1 as f32],
            nearest: rect.nearest,
            bin: bin.map_or(NO_BIN, |(_, b)| b as u32),
            ranges: [
                [first(0), lists[0].1, first(1), lists[1].1],
                [first(2), lists[2].1, first(3), lists[3].1],
                [first(4), lists[4].1, 0, 0],
            ],
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        let counts = bin.map_or(&self.no_counts, |(c, _)| c);
        if !self
            .bound
            .as_ref()
            .is_some_and(|b| b.indirect == *indirect && b.counts == *counts && b.hiz == *hiz)
        {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mirror test"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(hiz),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: indirect.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: counts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.sky.as_entire_binding(),
                    },
                ],
            });
            self.bound = Some(Bound {
                indirect: indirect.clone(),
                counts: counts.clone(),
                hiz: hiz.clone(),
                group,
            });
        }
        let Some(bound) = &self.bound else {
            return;
        };
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mirror test"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bound.group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        if let Some(k) = self.readback.iter().position(|s| !s.pending) {
            encoder.copy_buffer_to_buffer(&self.sky, 0, &self.readback[k].buffer, 0, 16);
            self.copied = Some(k);
        }
        self.tested = true;
    }

    /// The mirror pass's sky draw this frame: the test's (one instance, or
    /// none when it found the mirror hidden) once the test ran.
    pub fn sky_draw(&self) -> Option<&wgpu::Buffer> {
        self.tested.then_some(&self.sky)
    }

    /// After the frame's submit: map the verdict it copied.
    pub fn after_submit(&mut self) {
        let Some(k) = self.copied.take() else {
            return;
        };
        let frame = self.verdicts.frame;
        let s = &mut self.readback[k];
        s.pending = true;
        s.frame = frame;
        let ready = s.ready.clone();
        s.buffer.map_async(wgpu::MapMode::Read, .., move |r| {
            if r.is_ok() {
                ready.store(true, Ordering::Release);
            }
        });
    }

    /// This frame's statistics: `plan`, the mirror's if one was asked for;
    /// `draws`, its pass's draws.
    pub fn stats(&mut self, plan: Option<&Plan>, draws: u32) -> MirrorStats {
        let (verdicts, hidden, hidden_last) = self.verdicts.end_frame(self.tested);
        let drawn = plan.filter(|p| p.drawn());
        MirrorStats {
            found: plan.is_some(),
            skip: plan.and_then(|p| p.skip),
            tested: self.tested,
            verdicts,
            hidden,
            hidden_last,
            draws: drawn.map_or(0, |_| draws),
            area: drawn.map_or(0.0, |p| p.area),
        }
    }
}

/// `reflection_behind` (common.wgsl) for a lane of value `d` (distance to
/// the clip plane) and fine derivatives `dx`, `dy`: it leaves before its
/// shading.
#[cfg(test)]
fn behind(d: f32, dx: f32, dy: f32) -> bool {
    d < -4.0 * (dx.abs() + dy.abs()) - 1e-3
}

/// mirror_tap's fade (common.wgsl) of a read at `uv`, with the rect `rect`.
#[cfg(test)]
fn tap_fade(uv: Vec2, rect: [f32; 4]) -> f32 {
    let out_image = (-uv.x).max(uv.x - 1.0).max((-uv.y).max(uv.y - 1.0));
    let out_rect = (rect[0] - uv.x).max(uv.x - rect[2]).max((rect[1] - uv.y).max(uv.y - rect[3]));
    (1.0 - out_image.max(out_rect) * 20.0).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    /// A seeded xorshift32 giving numbers in 0..1.
    fn random(mut seed: u32) -> impl FnMut() -> f32 {
        move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed >> 8) as f32 / (1u32 << 24) as f32
        }
    }

    /// A mirror at `p` facing `n`, its probe box `size` across.
    fn mirror(p: Vec3, n: Vec3, size: Vec3) -> MirrorParams {
        let n = n.normalize();
        let rot = Quat::from_rotation_arc(Vec3::Z, n);
        MirrorParams {
            point: p,
            normal: n,
            world_to_box: Mat4::from_scale_rotation_translation(size, rot, p).inverse(),
        }
    }

    fn plane_of(m: &MirrorParams) -> Vec4 {
        let n = m.normal.normalize();
        n.extend(-n.dot(m.point))
    }

    /// renderer.rs `reflection_matrix`: points mirrored about n·p + d = 0.
    fn reflection(n: Vec3, d: f32) -> Mat4 {
        Mat4::from_cols(
            Vec4::new(1.0 - 2.0 * n.x * n.x, -2.0 * n.x * n.y, -2.0 * n.x * n.z, 0.0),
            Vec4::new(-2.0 * n.x * n.y, 1.0 - 2.0 * n.y * n.y, -2.0 * n.y * n.z, 0.0),
            Vec4::new(-2.0 * n.x * n.z, -2.0 * n.y * n.z, 1.0 - 2.0 * n.z * n.z, 0.0),
            Vec4::new(-2.0 * d * n.x, -2.0 * d * n.y, -2.0 * d * n.z, 1.0),
        )
    }

    fn up_for(dir: Vec3) -> Vec3 {
        if dir.normalize().z.abs() > 0.9 { Vec3::X } else { Vec3::Z }
    }

    /// The main view's view-projection from `eye` towards `at` (the
    /// viewer's reverse-Z infinite projection, 16:9).
    fn camera(eye: Vec3, at: Vec3) -> Mat4 {
        let proj = glam::camera::rh::proj::directx::perspective_infinite_reverse(1.0, 16.0 / 9.0, 0.1);
        proj * glam::camera::rh::view::look_at_mat4(eye, at, up_for(at - eye))
    }

    const WORLD: (u32, u32) = (1920, 1080);

    fn uv_of(view_proj: Mat4, p: Vec3) -> Option<Vec2> {
        let c = view_proj * p.extend(1.0);
        (c.w > 0.0).then(|| Vec2::new(c.x / c.w * 0.5 + 0.5, 0.5 - c.y / c.w * 0.5))
    }

    fn rect_of(volume: &Volume, view_proj: Mat4) -> ScreenRect {
        match volume.project(view_proj) {
            Projected::Rect(r) => r,
            other => panic!("not a rect: {other:?}"),
        }
    }

    /// Points where mirror_tap gives the image: inside the box (max |lp| <
    /// 0.5) and in front of its cut.
    fn tap_points(m: &MirrorParams, seed: u32, count: usize) -> Vec<Vec3> {
        let mut rnd = random(seed);
        let to_world = m.world_to_box.inverse();
        let plane = plane_of(m);
        let mut out = Vec::new();
        while out.len() < count {
            let lp = Vec3::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5) * 0.999;
            let p = to_world.transform_point3(lp);
            if plane.truncate().dot(p) + plane.w + TAP_BEHIND > 0.0 {
                out.push(p);
            }
        }
        out
    }

    const AXES: [Vec3; 6] = [Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y, Vec3::Z, Vec3::NEG_Z];

    fn tilted() -> [Vec3; 3] {
        [
            Vec3::new(1.0, 1.0, 1.0).normalize(),
            Vec3::new(0.3, -0.9, 0.2).normalize(),
            Vec3::new(-0.5, 0.1, -0.8).normalize(),
        ]
    }

    /// Plane side: a sphere in front, across the plane or within the
    /// clip's centimetre behind it stays; one wholly behind it goes, for
    /// mirrors facing every axis and tilted ones; the GPU culling's test
    /// (sphere_in) reads the same planes.
    #[test]
    fn the_culling_drops_only_what_lies_wholly_behind_the_mirror() {
        let p = Vec3::new(120.0, 80.0, 25.0);
        for n in AXES.into_iter().chain(tilted()) {
            let m = mirror(p, n, Vec3::new(2.0, 2.0, 0.4));
            let plane = plane_of(&m);
            let eye = p + n * 6.0 + n.any_orthonormal_vector() * 0.5;
            let vp = camera(eye, p) * reflection(n, plane.w);
            let planes = cull_planes(vp, Mat4::IDENTITY, plane);
            let r = 0.5;
            let kept = |offset: f32| crate::gpu_cull::sphere_in(&planes, p + n * offset, r);
            assert!(kept(2.0), "in front, {n}");
            assert!(kept(0.0) && kept(-0.3), "across, {n}");
            // reaching into the clip's centimetre behind the plane
            assert!(kept(-r - KEEP_BEHIND * 0.5), "within the kept centimetre, {n}");
            assert!(!kept(-r - KEEP_BEHIND - 0.01), "just behind, {n}");
            assert!(!kept(-2.0) && !kept(-4.0), "behind, {n}");
            // the clip plane is the sixth, and the others are the view's
            let view = crate::gpu_cull::view_planes(vp).map(|q| q.to_array());
            assert_eq!(planes[..5], view[..5], "{n}");
            assert_eq!(planes[5][3], plane.w + KEEP_BEHIND + CULL_SLACK);
        }
    }

    /// A sphere wholly behind the plane has no point the fragment clip
    /// keeps; one the culling keeps has some, or is within its slack.
    #[test]
    fn the_culling_plane_is_the_fragment_clip() {
        let mut rnd = random(7);
        for n in AXES.into_iter().chain(tilted()) {
            let p = Vec3::new(10.0, -4.0, 3.0);
            let plane = plane_of(&mirror(p, n, Vec3::ONE));
            let clip = plane + Vec4::new(0.0, 0.0, 0.0, KEEP_BEHIND);
            let planes = cull_planes(Mat4::IDENTITY, Mat4::IDENTITY, plane);
            for _ in 0..200 {
                let c = p + Vec3::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5) * 4.0;
                let r = rnd() * 1.5;
                let culled = Vec4::from_array(planes[5]).truncate().dot(c) + planes[5][3] < -r;
                // the sphere's point nearest the kept side
                let best = clip.truncate().dot(c) + clip.w + r;
                if culled {
                    assert!(best < 0.0, "culled with a kept point: {n} {c} {r}");
                } else {
                    assert!(best >= -CULL_SLACK - 1e-4, "kept with no point: {n} {c} {r}");
                }
            }
        }
    }

    /// The early discard leaves a lane only when its whole quad is behind
    /// the plane: random planes seen in perspective (the distance to the
    /// plane over 1 / w and 1 / w itself are affine on the screen), their
    /// depth within a factor of 3 across the quad, as anywhere farther than
    /// half a pixel from the surface's vanishing line; with fine derivatives
    /// (SPIR-V's) and with coarse ones, one lane's row and column for the
    /// whole quad (what Metal's dfdx may give).
    #[test]
    fn the_early_discard_never_splits_a_quad_with_a_kept_lane() {
        let mut rnd = random(91);
        let mut early = 0;
        for _ in 0..200_000 {
            let a = Vec3::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5) * Vec3::new(2.0, 2.0, 80.0);
            let b = Vec3::new(rnd() - 0.5, rnd() - 0.5, rnd()) * Vec3::new(0.02, 0.02, 1.0);
            let (x, y) = (rnd() * 100.0, rnd() * 100.0);
            let inv_w = |dx: f32, dy: f32| b.x * (x + dx) + b.y * (y + dy) + b.z;
            let w = [inv_w(0.0, 0.0), inv_w(1.0, 0.0), inv_w(0.0, 1.0), inv_w(1.0, 1.0)];
            let (lo, hi) = (
                w.iter().copied().fold(f32::INFINITY, f32::min),
                w.iter().copied().fold(0.0, f32::max),
            );
            if lo <= 1e-6 || hi > 3.0 * lo {
                continue;
            }
            let d = |dx: f32, dy: f32| (a.x * (x + dx) + a.y * (y + dy) + a.z) / inv_w(dx, dy);
            let q = [d(0.0, 0.0), d(1.0, 0.0), d(0.0, 1.0), d(1.0, 1.0)];
            // fine derivatives: each lane's row and column
            let fine = [
                behind(q[0], q[1] - q[0], q[2] - q[0]),
                behind(q[1], q[1] - q[0], q[3] - q[1]),
                behind(q[2], q[3] - q[2], q[2] - q[0]),
                behind(q[3], q[3] - q[2], q[3] - q[1]),
            ];
            // coarse ones: one lane's row and column for all four, whichever
            // lane the GPU takes
            let mut coarse = false;
            for (r0, r1) in [(0, 1), (2, 3)] {
                for (c0, c1) in [(0, 2), (1, 3)] {
                    coarse |= q.iter().any(|v| behind(*v, q[r1] - q[r0], q[c1] - q[c0]));
                }
            }
            if fine.iter().any(|g| *g) || coarse {
                early += 1;
                assert!(q.iter().all(|v| *v < 0.0), "a kept lane in a quad that lost one: {q:?}");
            }
        }
        assert!(early > 1000, "the early discard took {early} quads");
    }

    /// Camera in front, oblique, partly off the screen: every point where
    /// mirror_tap gives the image projects inside the rect, the rect inside
    /// mirror_tap's, and the scissor holds every read with a weight.
    #[test]
    fn the_rect_holds_every_point_that_takes_the_mirror() {
        let p = Vec3::new(50.0, 50.0, 22.0);
        let cases = [
            ("in front", Vec3::X, p + Vec3::X * 5.0, p),
            ("oblique", Vec3::X, p + Vec3::new(3.0, 4.0, 1.5), p),
            ("tilted", Vec3::new(1.0, 0.4, 0.3), p + Vec3::new(6.0, -2.0, 2.0), p),
            ("partly off", Vec3::Y, p + Vec3::new(0.0, 6.0, 0.0), p + Vec3::new(5.5, 0.0, 0.0)),
            ("facing down", Vec3::NEG_Z, p - Vec3::Z * 4.0 + Vec3::X, p),
        ];
        for (name, n, eye, at) in cases {
            let m = mirror(p, n, Vec3::new(1.5, 1.0, 0.3));
            let vp = camera(eye, at);
            let volume = Volume::new(m.world_to_box, plane_of(&m));
            let r = rect_of(&volume, vp);
            for q in tap_points(&m, 3, 2000) {
                let uv = uv_of(vp, q).expect("in front of the camera");
                assert!(
                    uv.cmpge(r.lo - 1e-5).all() && uv.cmple(r.hi + 1e-5).all(),
                    "{name}: {uv} outside {r:?}"
                );
            }
            let image = (1600, 900);
            let (s, rect, area) = scissor(&r, image);
            assert!(area > 0.0 && area <= 1.0, "{name}");
            if name == "partly off" {
                assert!(r.hi.x > 1.0 || r.lo.x < 0.0, "{name}: {r:?}");
            }
            let Some(s) = s else {
                assert_eq!(rect, OPEN_RECT, "{name}");
                continue;
            };
            // mirror_tap's rect holds the volume's
            assert!(
                rect[0] <= r.lo.x && rect[1] <= r.lo.y && rect[2] >= r.hi.x && rect[3] >= r.hi.y,
                "{name}"
            );
            // every read with a weight samples texels of the scissor
            let (w, h) = (image.0 as f32, image.1 as f32);
            for j in 0..400 {
                for i in 0..400 {
                    let uv = Vec2::new(i as f32 / 399.0 * 1.2 - 0.1, j as f32 / 399.0 * 1.2 - 0.1);
                    if tap_fade(uv, rect) <= 0.0 || tap_fade(uv, OPEN_RECT) <= 0.0 {
                        continue;
                    }
                    let t = (uv * Vec2::new(w, h) - 0.5).floor().clamp(Vec2::ZERO, Vec2::new(w - 2.0, h - 2.0));
                    let inside =
                        |x: f32, y: f32| x >= s[0] as f32 && y >= s[1] as f32 && x < (s[0] + s[2]) as f32 && y < (s[1] + s[3]) as f32;
                    assert!(inside(t.x, t.y) && inside(t.x + 1.0, t.y + 1.0), "{name}: {uv} reads outside {s:?}");
                }
            }
            // a surface flat on the mirror reads its own point: no fade there
            for q in tap_points(&m, 5, 500) {
                let uv = uv_of(vp, q).expect("in front");
                assert_eq!(tap_fade(uv, rect), tap_fade(uv, OPEN_RECT), "{name}: {uv}");
            }
        }
    }

    /// The sub-frustum: what projects into the scissor stays, what lies
    /// beside it in the view goes.
    #[test]
    fn the_narrowed_planes_keep_the_scissor() {
        let p = Vec3::new(30.0, 30.0, 30.0);
        let m = mirror(p, Vec3::X, Vec3::new(1.0, 1.0, 0.2));
        let vp = camera(p + Vec3::new(8.0, 1.0, 0.5), p);
        let volume = Volume::new(m.world_to_box, plane_of(&m));
        let image = (1920, 1080);
        let (s, _, _) = scissor(&rect_of(&volume, vp), image);
        let s = s.expect("a scissor for a small mirror");
        let planes = crate::gpu_cull::view_planes(narrow(s, image) * vp).map(|q| q.to_array());
        let mut rnd = random(11);
        let (w, h) = (image.0 as f32, image.1 as f32);
        for _ in 0..5000 {
            let c = p + Vec3::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5) * 6.0;
            let Some(uv) = uv_of(vp, c) else {
                continue;
            };
            let t = uv * Vec2::new(w, h);
            let in_scissor = t.x > s[0] as f32 && t.y > s[1] as f32 && t.x < (s[0] + s[2]) as f32 && t.y < (s[1] + s[3]) as f32;
            if in_scissor {
                assert!(crate::gpu_cull::sphere_in(&planes, c, 0.0), "{c} projects into the scissor");
            }
            // a point well beside the scissor, in front of the camera
            let far_beside = t.x < s[0] as f32 - 40.0
                || t.y < s[1] as f32 - 40.0
                || t.x > (s[0] + s[2]) as f32 + 40.0
                || t.y > (s[1] + s[3]) as f32 + 40.0;
            let dist = (c - (p + Vec3::new(8.0, 1.0, 0.5))).length();
            if far_beside && dist > 1.0 {
                assert!(!crate::gpu_cull::sphere_in(&planes, c, 0.01), "{c} beside the scissor");
            }
        }
        // the whole image narrows nothing
        let whole = narrow([0, 0, image.0, image.1], image);
        assert!(whole.abs_diff_eq(Mat4::IDENTITY, 1e-6), "{whole}");
    }

    /// The camera inside the volume, or the volume behind the camera: no
    /// rect bounds it, the pass is as before.
    #[test]
    fn a_camera_in_or_beside_the_volume_draws_as_before() {
        let p = Vec3::new(10.0, 10.0, 10.0);
        let m = mirror(p, Vec3::X, Vec3::new(4.0, 4.0, 4.0));
        let mut planner = Planner::default();
        let now = Instant::now();
        let inside = planner.plan(
            &m,
            camera(p + Vec3::X * 0.5, p + Vec3::X * 3.0),
            Mat4::IDENTITY,
            WORLD,
            WORLD,
            true,
            now,
        );
        assert_eq!(inside, Plan::FULL);
        let behind = planner.plan(
            &m,
            camera(p + Vec3::X * 6.0, p + Vec3::X * 12.0),
            Mat4::IDENTITY,
            WORLD,
            WORLD,
            true,
            now,
        );
        assert_eq!(behind, Plan::FULL);
        // a bad matrix: as before too
        let nan = Mat4::from_cols_array(&[f32::NAN; 16]);
        assert_eq!(planner.plan(&m, nan, nan, WORLD, WORLD, true, now), Plan::FULL);
    }

    /// Off the screen: left out at once, in any view; a mirror back on the
    /// screen is drawn at once.
    #[test]
    fn a_mirror_off_the_screen_is_left_out_at_once() {
        let p = Vec3::new(10.0, 10.0, 10.0);
        let m = mirror(p, Vec3::X, Vec3::new(1.0, 1.0, 0.2));
        let mut planner = Planner::default();
        let t = Instant::now();
        let on = camera(p + Vec3::X * 5.0, p);
        // turned 60 degrees away: the volume in front of the camera, beside the view
        let eye = p + Vec3::X * 5.0;
        let off = camera(eye, eye + Vec3::new(-0.5, 0.866, 0.0));
        assert!(planner.plan(&m, on, on, WORLD, WORLD, true, t).drawn());
        for users_view in [true, false] {
            let plan = planner.plan(&m, off, off, WORLD, WORLD, users_view, t);
            assert_eq!(plan.skip, Some(MirrorSkip::OffScreen));
            assert_eq!((plan.scissor, plan.test, plan.area), (None, None, 0.0));
        }
        assert!(planner.plan(&m, on, on, WORLD, WORLD, true, t + Duration::from_millis(16)).drawn());
    }

    /// Tiny: left out after TINY_HOLD since the last frame it was not, drawn
    /// again at once when it grows back; never in a photo or another view.
    #[test]
    fn a_tiny_mirror_is_left_out_after_its_hold() {
        let p = Vec3::new(10.0, 10.0, 10.0);
        let m = mirror(p, Vec3::X, Vec3::new(1.0, 1.0, 0.05));
        let near = camera(p + Vec3::X * 5.0, p);
        let far = camera(p + Vec3::X * 2000.0, p);
        let volume = Volume::new(m.world_to_box, plane_of(&m));
        let r = rect_of(&volume, far);
        assert!(((r.hi - r.lo) * Vec2::new(WORLD.0 as f32, WORLD.1 as f32)).min_element() < TINY_PX);
        let mut planner = Planner::default();
        let t = Instant::now();
        let at = |ms: u64| t + Duration::from_millis(ms);
        // never drawn before: left out at once
        assert_eq!(planner.plan(&m, far, far, WORLD, WORLD, true, t).skip, Some(MirrorSkip::Tiny));
        // a photo or another view draws it
        assert!(planner.plan(&m, far, far, WORLD, WORLD, false, t).drawn());
        assert!(planner.plan(&m, near, near, WORLD, WORLD, true, at(100)).drawn());
        assert!(planner.plan(&m, far, far, WORLD, WORLD, true, at(200)).drawn(), "held");
        assert!(planner.plan(&m, far, far, WORLD, WORLD, true, at(599)).drawn(), "held");
        assert_eq!(planner.plan(&m, far, far, WORLD, WORLD, true, at(600)).skip, Some(MirrorSkip::Tiny));
        // a photo frame between does not hold it either way
        assert!(planner.plan(&m, far, far, WORLD, WORLD, false, at(650)).drawn());
        assert_eq!(planner.plan(&m, far, far, WORLD, WORLD, true, at(700)).skip, Some(MirrorSkip::Tiny));
        assert!(planner.plan(&m, near, near, WORLD, WORLD, true, at(710)).drawn(), "at once");
    }

    /// The scissor: the whole image once it would reach every edge; the
    /// sides it does not reach take mirror_tap's rect.
    #[test]
    fn the_scissor_opens_the_sides_at_the_image_edge() {
        let size = (1000, 500);
        let small = ScreenRect {
            lo: Vec2::new(0.4, 0.4),
            hi: Vec2::new(0.5, 0.6),
            nearest: 0.1,
        };
        let (s, rect, area) = scissor(&small, size);
        let s = s.expect("a scissor");
        assert_eq!(rect, [0.4 - 0.001, 0.4 - 0.002, 0.5 + 0.001, 0.6 + 0.002]);
        // the rect, a texel of rounding and the fade band on each side, and a
        // texel for the filter
        assert!((346..=348).contains(&s[0]) && (552..=554).contains(&(s[0] + s[2])), "{s:?}");
        assert!((area - (s[2] * s[3]) as f32 / 500_000.0).abs() < 1e-6);
        let left = ScreenRect {
            lo: Vec2::new(0.02, 0.3),
            ..small
        };
        let (s, rect, _) = scissor(&left, size);
        assert_eq!(s.map(|s| s[0]), Some(0));
        assert_eq!(rect[0], -OPEN);
        assert!(rect[1] > 0.0 && rect[2] < 1.0 && rect[3] < 1.0);
        let big = ScreenRect {
            lo: Vec2::new(0.01, -0.2),
            hi: Vec2::new(0.97, 1.3),
            nearest: 0.1,
        };
        assert_eq!(scissor(&big, size), (None, OPEN_RECT, 1.0));
    }

    /// Shown: from the frame the mirror is, for SHOWN_HOLD after the last.
    #[test]
    fn what_stands_before_the_mirror_keeps_its_textures_a_second_after_it_shows() {
        let t = Instant::now();
        let at = |ms: u64| t + Duration::from_millis(ms);
        let mut s = Shown::default();
        s.note(false, t);
        assert!(!s.exempt());
        s.note(true, at(10));
        assert!(s.exempt());
        s.note(false, at(500));
        assert!(s.exempt());
        s.note(false, at(1009));
        assert!(s.exempt());
        s.note(false, at(1010));
        assert!(!s.exempt());
        s.note(true, at(3000));
        assert!(s.exempt());
    }

    /// The renderer's word: left out or last found hidden; nothing from a
    /// frame without a mirror; a frame with no result keeps the state but
    /// not the counts.
    #[test]
    fn the_stats_tell_a_mirror_left_out() {
        let drawn = MirrorStats {
            found: true,
            tested: true,
            verdicts: 2,
            hidden: 1,
            draws: 40,
            area: 0.25,
            ..Default::default()
        };
        assert!(!drawn.left_out());
        assert!(
            MirrorStats {
                hidden_last: true,
                ..drawn
            }
            .left_out()
        );
        assert!(
            MirrorStats {
                skip: Some(MirrorSkip::Tiny),
                ..drawn
            }
            .left_out()
        );
        assert!(!MirrorStats::default().left_out());
        let carried = drawn.carried();
        assert_eq!((carried.verdicts, carried.hidden, carried.draws, carried.found), (0, 0, 40, true));
    }

    /// Verdicts: counted as they come; the newest of the tests since the last
    /// frame without one tells hidden, an older one coming late does not.
    #[test]
    fn the_newest_verdict_tells_hidden() {
        let mut v = Verdicts {
            frame: 10,
            ..Default::default()
        };
        v.take(8, true);
        v.take(7, false);
        assert_eq!(v.end_frame(true), (2, 1, true));
        v.frame = 11;
        v.take(9, false);
        assert_eq!(v.end_frame(true), (1, 0, false));
        // a frame without a test forgets them; the late ones are counted
        v.frame = 12;
        assert_eq!(v.end_frame(false), (0, 0, false));
        v.frame = 13;
        v.take(10, true);
        assert_eq!(v.end_frame(true), (1, 1, false));
        v.frame = 14;
        v.take(13, true);
        assert_eq!(v.end_frame(true), (1, 1, true));
    }

    /// The occlusion test: valid, written for Vulkan and Metal, its
    /// parameters laid out as the shader's.
    #[test]
    fn the_occlusion_test_shader_is_valid_on_every_backend() {
        let src = include_str!("shaders/mirror_test.wgsl");
        let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));
        let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));
        let span = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("Test"))
            .map(|(_, t)| match t.inner {
                naga::TypeInner::Struct { span, .. } => span,
                _ => 0,
            });
        assert_eq!(span, Some(std::mem::size_of::<TestParams>() as u32));
        write_backends(&module, &info, naga::ShaderStage::Compute, "cs_mirror_test");
    }

    /// The scene's reflection entry points with the early discard, and
    /// mirror_tap's rect: valid, written for Vulkan and Metal (the module
    /// as Metal takes it, object_shader_source).
    #[test]
    fn the_reflection_shaders_are_valid_on_every_backend() {
        let src = format!(
            "{}\n{}\n{}",
            crate::COMMON_SHADER,
            crate::object_shader_source(wgpu::Features::empty()),
            include_str!("shaders/env.wgsl")
        );
        assert!(src.contains("fn reflection_behind(") && src.contains("frame.mirror_rect"));
        let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)));
        let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)));
        // the reflection passes' (early discard), the main view's (mirror_tap)
        for entry in [
            "fs_reflect",
            "fs_reflect_blend",
            "fs_terrain",
            "fs_terrain_pbr",
            "fs_mask",
            "fs_blend",
        ] {
            write_backends(&module, &info, naga::ShaderStage::Fragment, entry);
        }
    }

    fn write_backends(module: &naga::Module, info: &naga::valid::ModuleInfo, stage: naga::ShaderStage, entry: &str) {
        // the overrides at their defaults, as the pipelines that do not set them
        let constants = naga::back::PipelineConstants::default();
        let (module, info) = naga::back::pipeline_constants::process_overrides(module, info, Some((stage, entry)), &constants)
            .unwrap_or_else(|e| panic!("overrides {entry}: {e:?}"));
        let (module, info) = (&*module, &*info);
        let spv = naga::back::spv::Options {
            lang_version: (1, 3),
            ..Default::default()
        };
        let pipeline = naga::back::spv::PipelineOptions {
            shader_stage: stage,
            entry_point: entry.to_string(),
        };
        let words = naga::back::spv::write_vec(module, info, &spv, Some(&pipeline)).unwrap_or_else(|e| panic!("SPIR-V {entry}: {e}"));
        assert!(!words.is_empty());
        let msl = naga::back::msl::Options {
            lang_version: (3, 0),
            ..Default::default()
        };
        let pipeline = naga::back::msl::PipelineOptions {
            entry_point: Some((stage, entry.to_string())),
            ..Default::default()
        };
        let (text, _) = naga::back::msl::write_string(module, info, &msl, &pipeline).unwrap_or_else(|e| panic!("MSL {entry}: {e}"));
        assert!(!text.is_empty());
    }
}
