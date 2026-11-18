// aurora-local-patch: aurora-shader-host
//! aurora-shader: a generic host for user post effects written in WGSL,
//! drawn on the finished world image (after tone mapping and the scene's
//! MSAA / TAA resolve, before the viewer's spatial anti-aliasing, the photo
//! look, the selection outline, RLVa's vision spheres, the name tags, the
//! HUD attachments and the interface).
//!
//! The chain follows the AA setting, so its own edges (ink lines, fringes)
//! are smoothed like the scene's: under SMAA it runs between the post pass
//! and SMAA, on SMAA's 8-bit input; under FXAA the post pass leaves its FXAA
//! to the host's finish pass ([`ChainAa`]); under MSAA, whose resolve comes
//! before tone mapping, the finish is an FXAA masked to what the chain
//! changed sharply; with AA off, nothing. Under TAA and FSR (in TAA's
//! place) the chain reads the frame's jittered depth after the scene's
//! history has resolved it, so its lines would move by up to half a pixel
//! from one frame to the next: the finish resolves what the chain changed
//! over the frames instead, reprojected and clamped like the scene's TAA
//! (`fs_resolve`, `AURORA_SHADER_TEMPORAL`).
//!
//! The effects and their `config.toml` live in `<config>/aurora-shader/`
//! (`demo/aurora-shader/` in demo mode), installed by the separate
//! aurora-shader repository; this host knows nothing about any effect. It
//! runs them in the order `config.toml` gives, each reading the previous
//! one's image, under the binding contract its header names: v1
//! (`shaders/user_post_prelude.wgsl`; the full text is in the aurora-shader
//! repository's CONTRACT.md), or v2, which adds the depth of the blended
//! faces (`shaders/user_post_prelude_v2.wgsl`, [`FrameInputs::alpha_depth`]).
//! A v1 effect gets the same prelude, `Frame` and bindings as before v2.
//!
//! The Legacy Viewer has no such pass: its users get it from an LD_PRELOAD
//! hook on OpenGL (sl-shader), which draws on the default framebuffer at the
//! same point of the frame. Here it is a pass of the renderer: it gets the
//! real depth, projection and normals, the frame's own TAA jitter, and the
//! photos.
//!
//! Cost: nothing at all without the folder (one atomic load a frame); with
//! effects, a full-screen pass each, an optional one-workgroup compute
//! `prepare` each, a normals pass when an effect reads them, and above
//! water one depth pass that puts the water surface into the effects'
//! depth (the renderer's water writes none). Pipelines
//! are built on a worker thread; until a new version is ready the previous
//! one keeps drawing. `AURORA_PROFILE=1` logs the host's GPU times.
//!
//! A GUI (Preferences › Extras) reads the effects and edits `config.toml`
//! through the public items of `user_post_gui.rs`, re-exported here.
//!
//! The photo mode can also take the whole chain as one of its looks
//! ("Otoa's Shader", 52-shader-photo-look): the chain then draws only at
//! that layer's place in the look stack ([`Place::Layer`],
//! [`UserPost::encode_layer`]), from the image the looks before it left,
//! with the same depth, normals and finish rules, never also at its slot.

#[path = "user_post_config.rs"]
mod config;
#[path = "user_post_gui.rs"]
mod gui;
#[path = "user_post_library.rs"]
mod library;
#[cfg(test)]
#[path = "user_post_tests.rs"]
mod tests;

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, mpsc};
use std::time::{Duration, Instant};

use glam::{Mat4, Vec2};

use crate::pipeline_cache::CachedPipelines;
pub use gui::{
    ConfigView, Control, Edit, EffectInfo, Param, available, bool_token, config_path, edit_config, effects, generation, installed,
    names_token, number_token, numbers_token, push_ordered, read_config, running, save_config, write_atomic,
};
use library::{Compiled, Library};

/// Format of the images between two effects (like the photo look's: 8 bits
/// between steps, as sl-shader recaptured RGBA8 between its plugins).
const INTER: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const NORMALS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// The temporal finish's history: what the chain changed (signed, -1..1),
/// finer than the 8 bits of a step so that a slow blend does not stall.
const HISTORY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// The temporal finish's weight of the current frame, per frame like the
/// scene's TAA (renderer.rs): the noise it averages is the jitter, which
/// changes every frame whatever the frame rate; the neighbourhood clamp,
/// not time, keeps it from trailing behind motion.
const TEMPORAL_WEIGHT: f32 = 0.1;
/// Bytes of the temporal finish's uniform (`Resolve` in the host shader).
const RESOLVE_SIZE: u64 = 96;
/// The effects' depth with the water in it (`fs_depth`), like `depth_ss`.
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Frames without the chain before its images are let go (looks.rs).
const KEEP_FRAMES: u32 = 120;
/// Bytes of `Frame` (contracts v1 and v2) and the stride of its copies in one
/// buffer: slot 0 for the normals pass, then one per effect.
const FRAME_SIZE: u64 = 336;
const FRAME_STRIDE: u64 = 512;
const FRAME_SLOTS: u64 = 1 + library::MAX_EFFECTS as u64;
const STATE_SIZE: u64 = 4096;
const FLAG_GEOMETRY_NORMALS: u32 = 1;
const FLAG_TAA: u32 = 2;
const FLAG_PHOTO: u32 = 4;
const FLAG_VR_EYE: u32 = 8;
const FLAG_STATE_RESET: u32 = 16;
/// Contract v2 only: `alpha_depth` holds the blended faces this frame.
const FLAG_ALPHA_DEPTH: u32 = 32;
/// Passes timed in a frame: the water depth, normals, prepare + draw per
/// effect, the blit or the AA finish.
const MAX_TIMED_PASSES: u32 = 3 + 2 * library::MAX_EFFECTS as u32;
const PROFILE_EVERY: Duration = Duration::from_secs(2);

/// Where the effects are read from, before the renderer starts. The
/// viewer passes `<config>/aurora-shader`; `AURORA_SHADER_DIR` replaces it
/// (tests, demo captures) and `AURORA_SHADER=0` turns the host off for the
/// launch (should an effect ever hang the GPU).
pub fn configure(dir: PathBuf) {
    if std::env::var("AURORA_SHADER").is_ok_and(|v| v.trim() == "0") {
        log::info!("aurora-shader: off for this launch (AURORA_SHADER=0)");
        return;
    }
    let dir = std::env::var_os("AURORA_SHADER_DIR")
        .filter(|v| !v.is_empty())
        .map_or(dir, PathBuf::from);
    log::info!("aurora-shader: effects folder {}", dir.display());
    match temporal_mode() {
        TemporalMode::Auto => {}
        TemporalMode::Off => log::info!("aurora-shader: no temporal finish for this launch (AURORA_SHADER_TEMPORAL=0)"),
        TemporalMode::Always => log::info!("aurora-shader: temporal finish under every AA (AURORA_SHADER_TEMPORAL=always)"),
    }
    library::watch(dir);
}

/// When the chain's output is resolved over the frames
/// (`AURORA_SHADER_TEMPORAL`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TemporalMode {
    /// `0`: never; the masked FXAA under TAA and FSR, as before.
    Off,
    /// Unset or `1`: under TAA and FSR, whose jitter moves the depth the
    /// effects read.
    Auto,
    /// `always`: under every AA but a photo's, for comparisons (there is no
    /// jitter to average then: it only steadies what changes from one frame
    /// to the next).
    Always,
}

impl TemporalMode {
    fn parse(v: Option<&str>) -> TemporalMode {
        match v.map(str::trim) {
            Some("0" | "off") => TemporalMode::Off,
            Some("always") => TemporalMode::Always,
            _ => TemporalMode::Auto,
        }
    }
}

fn temporal_mode() -> TemporalMode {
    static MODE: OnceLock<TemporalMode> = OnceLock::new();
    *MODE.get_or_init(|| TemporalMode::parse(std::env::var("AURORA_SHADER_TEMPORAL").ok().as_deref()))
}

fn profiling() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("AURORA_PROFILE").is_some())
}

/// What kind of image the frame is.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Context {
    /// A photo (`Capture::Snapshot`): drawn only with `[host] photos`.
    pub(crate) photo: bool,
    /// An eye of the VR headset: drawn only with `[host] vr`.
    pub(crate) vr_eye: bool,
    /// The anti-aliasing the host itself does on the chain's output.
    pub(crate) aa: ChainAa,
}

/// Where the chain draws in a frame ([`place`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Place {
    /// Nowhere.
    Off,
    /// At its own slot: after tone mapping, before the spatial AA and the
    /// photo's looks.
    Normal,
    /// Only at the place of the photo look "Otoa's Shader" in the photo's
    /// look stack (`looks::LookOp::UserChain`, 52-shader-photo-look), on the
    /// image the looks before it left; not at its own slot.
    Layer,
}

/// Where the chain draws: nowhere without an effect (`[host] enabled =
/// false`, an empty `order`, no folder) or on a VR eye without `[host] vr`;
/// at the look's place when `layer` (the photo mode's live view or a photo
/// whose look stack holds "Otoa's Shader", ticked): there it was asked for
/// as a look, so a photo has it whatever `[host] photos` says, like the live
/// view it is taken from; otherwise at its own slot, a photo only with
/// `[host] photos`. Never both: the frame draws the chain once.
pub(crate) fn place(effects: bool, photos: bool, vr: bool, context: Context, layer: bool) -> Place {
    if !effects || (context.vr_eye && !vr) {
        Place::Off
    } else if layer {
        Place::Layer
    } else if context.photo && !photos {
        Place::Off
    } else {
        Place::Normal
    }
}

/// The anti-aliasing of the chain's output, from the viewer's AA setting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum ChainAa {
    /// None: AA off, or SMAA (which runs after the chain, on its result).
    #[default]
    Off,
    /// The post pass's FXAA, moved to the chain's end (the whole image).
    Fxaa,
    /// FXAA only where the chain drew sharp changes: the scene is resolved
    /// already (MSAA; TAA and FSR with `AURORA_SHADER_TEMPORAL=0`) and must
    /// not be blurred twice.
    FxaaMasked,
    /// What the chain changed, resolved over the frames (`fs_resolve`): the
    /// history follows the motion vectors (else the camera, through the
    /// depth) and is clamped to the current 3 x 3 neighbourhood, as the
    /// scene's TAA does. Under TAA and FSR the depth the effects read is
    /// jittered, so their lines (ink, fringes) land up to half a pixel off
    /// from one frame to the next; this averages them as the scene's own
    /// edges are, without blurring them.
    Temporal,
}

impl ChainAa {
    /// `msaa_samples`: the scene's (1 when the GPU has no MSAA: the scene is
    /// aliased, so is the chain). A TAA (or FSR) photo starts a new history,
    /// its scene is not anti-aliased: the FXAA is not masked there.
    pub(crate) fn for_setting(aa: crate::types::AntiAliasing, msaa_samples: u32, photo: bool) -> ChainAa {
        ChainAa::for_setting_with(aa, msaa_samples, photo, temporal_mode())
    }

    /// [`Self::for_setting`] under a given `AURORA_SHADER_TEMPORAL`. A photo
    /// is one frame: never temporal.
    fn for_setting_with(aa: crate::types::AntiAliasing, msaa_samples: u32, photo: bool, temporal: TemporalMode) -> ChainAa {
        use crate::types::AntiAliasing as Aa;
        let always = temporal == TemporalMode::Always && !photo;
        match aa {
            _ if always => ChainAa::Temporal,
            Aa::Fxaa => ChainAa::Fxaa,
            // FSR runs in the TAA pass's place, before tone mapping
            Aa::Taa | Aa::Fsr(_) if photo => ChainAa::Fxaa,
            Aa::Taa | Aa::Fsr(_) if temporal == TemporalMode::Off => ChainAa::FxaaMasked,
            Aa::Taa | Aa::Fsr(_) => ChainAa::Temporal,
            Aa::Msaa(_) if msaa_samples > 1 => ChainAa::FxaaMasked,
            Aa::None | Aa::Msaa(_) | Aa::Smaa => ChainAa::Off,
        }
    }

    /// The finish at the place of the photo look "Otoa's Shader"
    /// ([`Place::Layer`]). The image there is the finished one: the frame's
    /// own AA (SMAA, FXAA, MSAA, TAA, FSR) has run on it, and the looks
    /// before may have changed it (grain, a blur), so it is never
    /// anti-aliased a second time as a whole: only the chain's own sharp
    /// changes get the masked FXAA. Under TAA and FSR the effects still read
    /// the jittered depth, so on the live view the chain's change is resolved
    /// over the frames as at its own slot (the lines would move every frame
    /// otherwise); a photo is one frame: the masked FXAA. Without AA, or
    /// with an MSAA the GPU cannot do, the scene is aliased: so is the chain.
    pub(crate) fn for_layer(aa: crate::types::AntiAliasing, msaa_samples: u32, photo: bool) -> ChainAa {
        ChainAa::for_layer_with(aa, msaa_samples, photo, temporal_mode())
    }

    /// [`Self::for_layer`] under a given `AURORA_SHADER_TEMPORAL`.
    fn for_layer_with(aa: crate::types::AntiAliasing, msaa_samples: u32, photo: bool, temporal: TemporalMode) -> ChainAa {
        use crate::types::AntiAliasing as Aa;
        match aa {
            _ if temporal == TemporalMode::Always && !photo => ChainAa::Temporal,
            Aa::Taa | Aa::Fsr(_) if !photo && temporal != TemporalMode::Off => ChainAa::Temporal,
            Aa::None => ChainAa::Off,
            Aa::Msaa(_) if msaa_samples <= 1 => ChainAa::Off,
            _ => ChainAa::FxaaMasked,
        }
    }

    fn label(self) -> &'static str {
        match self {
            ChainAa::Off => "blit",
            ChainAa::Fxaa => "fxaa",
            ChainAa::FxaaMasked => "fxaa masked",
            ChainAa::Temporal => "temporal",
        }
    }
}

/// The frame's inputs, at the chain's slot of `render_view`.
pub(crate) struct FrameInputs<'a> {
    /// The scene's depth at the chain's size, reversed-Z: `targets.depth_ss`,
    /// or FSR's display depth when FSR upscales (the water images stay the
    /// world's size; `fs_depth` reads them by size ratio). It has no water:
    /// the water pass writes no depth (it reads this one for what is under
    /// it), so `water` is merged in before the effects see it.
    pub(crate) depth: &'a wgpu::TextureView,
    /// GTAO's mesh normals, when this frame drew them.
    pub(crate) mesh_normals: Option<&'a wgpu::TextureView>,
    /// The water surface, above water only (under water the water fog owns
    /// the distance, and the surface seen from below must not hide it).
    pub(crate) water: Option<WaterInputs<'a>>,
    /// Contract v2's `alpha_depth`: the occlusion's depth (`targets.ao_depth`,
    /// made with the ambient occlusion or FSR), the prepass with the blended
    /// faces at least half covered drawn over it (#882; under FSR also the
    /// fainter ones from 10 %, for their motion), same size and jitter as
    /// `depth_ss`. Only when the chain's depth is `depth_ss` itself: FSR's
    /// upscaled display depth is made from this one, so it already holds
    /// those faces and is not at its size. None: the effects bind their own
    /// depth there, and see nothing in front.
    pub(crate) alpha_depth: Option<&'a wgpu::TextureView>,
    /// Unjittered.
    pub(crate) proj: Mat4,
    pub(crate) view: Mat4,
    pub(crate) draw_distance: f32,
    pub(crate) time: f32,
    /// `photo_part(photo_tile)`.
    pub(crate) tile: [f32; 4],
    pub(crate) pixel_scale: f32,
    /// `depth_jitter`: window pixels, y down.
    pub(crate) depth_jitter: Vec2,
    /// The window's size (`depth_jitter`'s pixels).
    pub(crate) output_size: (u32, u32),
    pub(crate) taa: bool,
    pub(crate) frame_index: u64,
    /// The depth prepass's normals and motion target (renderer.rs
    /// `motion_view`: zw the screen motion since the last frame in uv,
    /// without the jitter, 0 where it drew nothing), when this frame wrote
    /// it; the world's size. For the temporal finish.
    pub(crate) motion: Option<&'a wgpu::TextureView>,
    /// This frame's unjittered clip space (raw depth z) to the previous
    /// frame's (`FrameUniforms::motion_reproj`): the temporal finish's
    /// reprojection where nothing moved on its own.
    pub(crate) reproject: Mat4,
    /// The scene moved as a whole (a teleport, a region crossing, the VR
    /// session's start or end): the temporal finish starts over.
    pub(crate) cut: bool,
}

/// `FrameInputs::depth_jitter` in the chain's pixels (`size`).
fn chain_jitter(f: &FrameInputs, size: (u32, u32)) -> Vec2 {
    let (w, h) = (size.0.max(1) as f32, size.1.max(1) as f32);
    f.depth_jitter * Vec2::new(w / f.output_size.0.max(1) as f32, h / f.output_size.1.max(1) as f32)
}

/// `Resolve` of the host shader (std140, 96 bytes): the temporal finish.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ResolveU {
    reproject: [f32; 16],
    /// xy: where a pixel's depth and motion are, in the chain's pixels
    jitter: [f32; 4],
    /// x: the current frame's weight; y: 1 to start over; z: 1 when the
    /// motion vectors are bound
    params: [f32; 4],
}

const _: () = assert!(size_of::<ResolveU>() as u64 == RESOLVE_SIZE);

/// The renderer's nearest water surface, drawn before scene pass A with the
/// same frame uniform (TAA jitter, photo tile, VR eye) and size as
/// `depth_ss`.
pub(crate) struct WaterInputs<'a> {
    /// `targets.water_interface`: Depth32Float, 1 / (1 + view distance) of
    /// the nearest water or water-exclusion face (env.wgsl
    /// water_interface_depth), 0 where there is none; not tested against
    /// the scene.
    pub(crate) interface: &'a wgpu::TextureView,
    /// `targets.water_mask`: R8Unorm, 1 water, 0 where an exclusion face (a
    /// hull) is nearer.
    pub(crate) mask: &'a wgpu::TextureView,
}

/// `Frame` of the prelude (std140-compatible, 336 bytes).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FrameU {
    proj: [f32; 16],
    inv_proj: [f32; 16],
    view: [f32; 16],
    inv_view: [f32; 16],
    size: [f32; 4],
    jitter: [f32; 4],
    tile: [f32; 4],
    time: f32,
    delta: f32,
    frame: u32,
    flags: u32,
    near: f32,
    draw_distance: f32,
    pixel_scale: f32,
    api: u32,
}

const _: () = assert!(size_of::<FrameU>() as u64 == FRAME_SIZE);

impl FrameU {
    /// `api`: the contract of the effect that reads it (1 for the host's own
    /// passes), as its `Frame::api`.
    fn new(f: &FrameInputs, size: (u32, u32), delta: f32, flags: u32, api: u32) -> FrameU {
        let (w, h) = (size.0.max(1) as f32, size.1.max(1) as f32);
        // window pixels to the world's (render scale)
        let jitter = chain_jitter(f, size);
        FrameU {
            proj: f.proj.to_cols_array(),
            inv_proj: f.proj.inverse().to_cols_array(),
            view: f.view.to_cols_array(),
            inv_view: f.view.inverse().to_cols_array(),
            size: [w, h, 1.0 / w, 1.0 / h],
            jitter: [jitter.x, jitter.y, jitter.x / w, jitter.y / h],
            tile: f.tile,
            time: f.time,
            delta,
            frame: f.frame_index as u32,
            flags,
            // reversed-Z infinite perspective: clip z = near, so the view
            // distance is near / d (also for a tile's or an eye's proj)
            near: f.proj.w_axis.z,
            draw_distance: f.draw_distance,
            pixel_scale: f.pixel_scale,
            api,
        }
    }
}

/// An effect's pipelines, for one version of its source.
struct Built {
    hash: u64,
    compiled: Arc<Compiled>,
    /// Drawing into an intermediate image.
    inter: wgpu::RenderPipeline,
    /// Drawing into the frame (its format), when it is not sRGB.
    out: Option<wgpu::RenderPipeline>,
    prepare: Option<wgpu::ComputePipeline>,
}

/// An effect's GPU side, kept across versions of its source.
struct EffectGpu {
    name: String,
    built: Option<Built>,
    /// The version being built on a worker.
    building: Option<(u64, mpsc::Receiver<Result<Built, String>>)>,
    /// A version the GPU refused, and why: not built again.
    failed: Option<(u64, String)>,
    params: wgpu::Buffer,
    params_bg: wgpu::BindGroup,
    state: wgpu::Buffer,
    /// `state` is zeroed before the next draw (a new version).
    reset: bool,
}

/// The images of the chain, for one size and format.
struct Targets {
    size: (u32, u32),
    /// What the post pass draws into (the chain's format: the frame's, or
    /// SMAA's input format under SMAA).
    input: wgpu::TextureView,
    /// The same, as effects read it (without sRGB decoding).
    input_read: wgpu::TextureView,
    inter: [Option<wgpu::TextureView>; 2],
    normals: Option<wgpu::TextureView>,
    /// The scene depth with the water surface merged in (`fs_depth`), made
    /// the first frame that has water and kept with the other images.
    depth: Option<wgpu::TextureView>,
    /// The temporal finish's history, read from one while the other is
    /// written ([`UserPost::parity`]); only while the finish is temporal.
    history: Option<[wgpu::TextureView; 2]>,
    /// The chain's output at its photo look's place ([`Place::Layer`]): the
    /// view drawn into, in the chain's format, and the same as the looks read
    /// it (without sRGB decoding), made the first frame it draws there.
    layer_out: Option<[wgpu::TextureView; 2]>,
}

struct Gpu {
    format: wgpu::TextureFormat,
    layout: wgpu::PipelineLayout,
    group0: wgpu::BindGroupLayout,
    group1: wgpu::BindGroupLayout,
    host: wgpu::ShaderModule,
    normals_group: wgpu::BindGroupLayout,
    normals_pipeline: wgpu::RenderPipeline,
    /// The water depth pass: the scene depth, the water interface and mask
    /// into `Targets::depth`.
    depth_group: wgpu::BindGroupLayout,
    depth_pipeline: wgpu::RenderPipeline,
    /// sRGB frames: the decoding copy from the last effect's image.
    blit: Option<(wgpu::BindGroupLayout, wgpu::RenderPipeline)>,
    /// The AA finish, made the first time a frame needs it.
    finish: Option<Finish>,
    linear: wgpu::Sampler,
    point: wgpu::Sampler,
    frames: wgpu::Buffer,
    /// Bound as `normals` when no effect reads them, and as GTAO's when
    /// the frame has none.
    blank: wgpu::TextureView,
    effects: Vec<EffectGpu>,
    targets: Option<Targets>,
    timer: Option<Timer>,
}

fn entry(binding: u32, visibility: wgpu::ShaderStages, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty,
        count: None,
    }
}

fn texture(sample_type: wgpu::TextureSampleType) -> wgpu::BindingType {
    wgpu::BindingType::Texture {
        sample_type,
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    }
}

fn uniform(min: u64) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: wgpu::BufferSize::new(min),
    }
}

fn host_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    vs: &wgpu::ShaderModule,
    fs: &wgpu::ShaderModule,
    entry: &str,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline_cached(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: vs,
            entry_point: Some("vs_fullscreen"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: fs,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn make_texture(
    device: &wgpu::Device,
    label: &str,
    size: (u32, u32),
    format: wgpu::TextureFormat,
    view_formats: &[wgpu::TextureFormat],
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.0.max(1),
            height: size.1.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats,
    })
}

impl Gpu {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Gpu {
        let both = wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE;
        let fragment = wgpu::ShaderStages::FRAGMENT;
        let float = wgpu::TextureSampleType::Float { filterable: true };
        let unfiltered = wgpu::TextureSampleType::Float { filterable: false };
        // explicit layouts: a binding an effect does not use is harmless
        let group0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aurora-shader host"),
            entries: &[
                entry(0, both, texture(float)),
                entry(1, both, texture(wgpu::TextureSampleType::Depth)),
                entry(2, both, texture(float)),
                entry(3, both, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
                entry(4, both, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering)),
                entry(5, both, uniform(FRAME_SIZE)),
                // read_write, also in fs_main, which only reads it: a
                // writable storage buffer is core WebGPU in the fragment
                // stage (only the vertex stage needs VERTEX_WRITABLE_STORAGE)
                entry(
                    6,
                    both,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(STATE_SIZE),
                    },
                ),
                // contract v2's `alpha_depth`; a v1 effect, which does not
                // declare it, is built with the same layout (unused)
                entry(7, both, texture(wgpu::TextureSampleType::Depth)),
            ],
        });
        let group1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aurora-shader params"),
            entries: &[entry(0, both, uniform(library::PARAMS_SIZE as u64))],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("aurora-shader effect"),
            bind_group_layouts: &[Some(&group0), Some(&group1)],
            immediate_size: 0,
        });
        let host = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("aurora-shader host"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/user_post_host.wgsl").into()),
        });
        let normals_group = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aurora-shader normals"),
            entries: &[
                entry(0, fragment, texture(wgpu::TextureSampleType::Depth)),
                entry(1, fragment, texture(unfiltered)),
                entry(2, fragment, uniform(FRAME_SIZE)),
                // the scene depth without the water: tells water pixels,
                // whose mesh normal is the seabed's
                entry(7, fragment, texture(wgpu::TextureSampleType::Depth)),
            ],
        });
        let normals_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("aurora-shader normals"),
            bind_group_layouts: &[Some(&normals_group)],
            immediate_size: 0,
        });
        let normals_pipeline = host_pipeline(
            device,
            "aurora-shader normals",
            &normals_layout,
            &host,
            &host,
            "fs_normals",
            NORMALS_FORMAT,
        );
        let depth_group = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aurora-shader depth"),
            entries: &[
                entry(2, fragment, uniform(FRAME_SIZE)),
                entry(7, fragment, texture(wgpu::TextureSampleType::Depth)),
                entry(8, fragment, texture(wgpu::TextureSampleType::Depth)),
                entry(9, fragment, texture(unfiltered)),
            ],
        });
        let depth_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("aurora-shader depth"),
            bind_group_layouts: &[Some(&depth_group)],
            immediate_size: 0,
        });
        // no colour target: fs_depth only writes the depth (host_pipeline
        // always has one)
        let depth_pipeline = device.create_render_pipeline_cached(&wgpu::RenderPipelineDescriptor {
            label: Some("aurora-shader depth"),
            layout: Some(&depth_layout),
            vertex: wgpu::VertexState {
                module: &host,
                entry_point: Some("vs_fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &host,
                entry_point: Some("fs_depth"),
                compilation_options: Default::default(),
                targets: &[],
            }),
            multiview_mask: None,
            cache: None,
        });
        let blit = format.is_srgb().then(|| {
            let group = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora-shader blit"),
                entries: &[entry(3, fragment, texture(unfiltered))],
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("aurora-shader blit"),
                bind_group_layouts: &[Some(&group)],
                immediate_size: 0,
            });
            let pipeline = host_pipeline(device, "aurora-shader blit", &layout, &host, &host, "fs_blit", format);
            (group, pipeline)
        });
        let sampler = |label: &str, filter: wgpu::FilterMode| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(label),
                mag_filter: filter,
                min_filter: filter,
                ..Default::default()
            })
        };
        Gpu {
            format,
            layout,
            group0,
            group1,
            host,
            normals_group,
            normals_pipeline,
            depth_group,
            depth_pipeline,
            blit,
            finish: None,
            linear: sampler("aurora-shader linear", wgpu::FilterMode::Linear),
            point: sampler("aurora-shader point", wgpu::FilterMode::Nearest),
            frames: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("aurora-shader frame"),
                size: FRAME_STRIDE * FRAME_SLOTS,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            blank: make_texture(device, "aurora-shader blank", (1, 1), NORMALS_FORMAT, &[]).create_view(&Default::default()),
            effects: Vec::new(),
            targets: None,
            timer: None,
        }
    }

    /// The buffers of a new effect of the chain.
    fn add_effect(&mut self, device: &wgpu::Device, name: &str) {
        let buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let params = buffer("aurora-shader params", library::PARAMS_SIZE as u64, wgpu::BufferUsages::UNIFORM);
        let params_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora-shader params"),
            layout: &self.group1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params.as_entire_binding(),
            }],
        });
        self.effects.push(EffectGpu {
            name: name.to_owned(),
            built: None,
            building: None,
            failed: None,
            params,
            params_bg,
            // (zero on creation: wgpu clears new buffers)
            state: buffer("aurora-shader state", STATE_SIZE, wgpu::BufferUsages::STORAGE),
            reset: false,
        });
    }

    /// Makes the GPU side match the library: starts the builds of new
    /// versions, takes the finished ones, forgets removed effects. True
    /// when an effect's buffers were just made (its params must be written).
    fn sync(&mut self, device: &wgpu::Device, library: &Library) -> bool {
        self.effects.retain(|g| library.effects.iter().any(|e| e.name == g.name));
        let mut fresh = false;
        let format = self.format;
        let (layout, host) = (self.layout.clone(), self.host.clone());
        for e in &library.effects {
            if !self.effects.iter().any(|g| g.name == e.name) {
                self.add_effect(device, &e.name);
                fresh = true;
            }
            let Some(g) = self.effects.iter_mut().find(|g| g.name == e.name) else {
                continue;
            };
            harvest(g);
            let want = e.compiled.hash;
            let running = g.built.as_ref().map(|b| b.hash);
            let pending = g.building.as_ref().map(|(h, _)| *h);
            if running != Some(want) && pending != Some(want) && g.failed.as_ref().is_none_or(|(h, _)| *h != want) {
                g.building = Some((want, start_build(device, &layout, &host, e.compiled.clone(), format)));
            }
        }
        fresh
    }

    /// The chain's images and their bytes (for the log of a release).
    fn footprint(&self) -> (usize, u64) {
        let Some(t) = &self.targets else { return (0, 0) };
        let pixels = u64::from(t.size.0) * u64::from(t.size.1);
        let inter = t.inter.iter().flatten().count();
        let normals = usize::from(t.normals.is_some());
        let depth = usize::from(t.depth.is_some());
        let history = 2 * usize::from(t.history.is_some());
        let layer = usize::from(t.layer_out.is_some());
        let images = 1 + inter + normals + depth + history + layer;
        // input, intermediates, the water depth and the look's output: 4
        // bytes a pixel; normals and the history: Rgba16Float
        (
            images,
            pixels * (4 * (1 + inter + depth + layer) as u64 + 8 * (normals + history) as u64),
        )
    }

    /// The images for `steps` effects at `size`; intermediate images,
    /// normals and the temporal history only when used. `finish`: a pass
    /// follows the last effect (the AA finish, or the sRGB blit); `layer`:
    /// the chain draws at its photo look's place. True when the history was
    /// just made (nothing in it yet).
    #[allow(clippy::too_many_arguments)]
    fn ensure_targets(
        &mut self,
        device: &wgpu::Device,
        size: (u32, u32),
        steps: usize,
        normals: bool,
        finish: bool,
        temporal: bool,
        layer: bool,
    ) -> bool {
        let format = self.format;
        if self.targets.as_ref().is_none_or(|t| t.size != size) {
            // the post pipeline is compiled for the frame's format (SMAA's
            // input format under SMAA), so its target is in it (looks.rs)
            let raw = format.remove_srgb_suffix();
            let srgb_views = if format.is_srgb() { vec![raw] } else { Vec::new() };
            let input = make_texture(device, "aurora-shader input", size, format, &srgb_views);
            let input_read = input.create_view(&wgpu::TextureViewDescriptor {
                format: Some(raw),
                ..Default::default()
            });
            self.targets = Some(Targets {
                size,
                input: input.create_view(&Default::default()),
                input_read,
                inter: [None, None],
                normals: None,
                depth: None,
                history: None,
                layer_out: None,
            });
        }
        let Some(t) = &mut self.targets else { return false };
        if layer && t.layer_out.is_none() {
            let raw = format.remove_srgb_suffix();
            let srgb_views = if format.is_srgb() { vec![raw] } else { Vec::new() };
            let out = make_texture(device, "aurora-shader look", size, format, &srgb_views);
            let read = out.create_view(&wgpu::TextureViewDescriptor {
                format: Some(raw),
                ..Default::default()
            });
            t.layer_out = Some([out.create_view(&Default::default()), read]);
        }
        // images written between steps: all but the last effect, plus the
        // last one when the AA finish or the sRGB blit follows
        let writes = steps.saturating_sub(1) + usize::from(finish || format.is_srgb());
        for (i, slot) in t.inter.iter_mut().enumerate() {
            if i < writes.min(2) && slot.is_none() {
                let label = if i == 0 { "aurora-shader a" } else { "aurora-shader b" };
                *slot = Some(make_texture(device, label, size, INTER, &[]).create_view(&Default::default()));
            }
        }
        if normals && t.normals.is_none() {
            t.normals = Some(make_texture(device, "aurora-shader normals", size, NORMALS_FORMAT, &[]).create_view(&Default::default()));
        }
        if !temporal {
            t.history = None;
            return false;
        }
        if t.history.is_some() {
            return false;
        }
        let image = |label| make_texture(device, label, size, HISTORY_FORMAT, &[]).create_view(&Default::default());
        t.history = Some([image("aurora-shader history a"), image("aurora-shader history b")]);
        true
    }
}

/// The AA finish: FXAA (`user_post_host.wgsl`) from the last effect's image
/// into the frame, masked or not; on sRGB frames it also decodes, as the
/// blit does.
struct Finish {
    group: wgpu::BindGroupLayout,
    fxaa: wgpu::RenderPipeline,
    masked: wgpu::RenderPipeline,
    /// The temporal finish (`fs_resolve`): the frame and the next history.
    resolve_group: wgpu::BindGroupLayout,
    resolve: wgpu::RenderPipeline,
    resolve_buffer: wgpu::Buffer,
}

impl Finish {
    fn new(device: &wgpu::Device, host: &wgpu::ShaderModule, format: wgpu::TextureFormat) -> Finish {
        let fragment = wgpu::ShaderStages::FRAGMENT;
        let float = wgpu::TextureSampleType::Float { filterable: true };
        let group = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aurora-shader finish"),
            entries: &[
                entry(4, fragment, texture(float)),
                entry(5, fragment, texture(float)),
                entry(6, fragment, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("aurora-shader finish"),
            bind_group_layouts: &[Some(&group)],
            immediate_size: 0,
        });
        let (fxaa, masked) = if format.is_srgb() {
            ("fs_fxaa_srgb", "fs_fxaa_masked_srgb")
        } else {
            ("fs_fxaa", "fs_fxaa_masked")
        };
        let unfiltered = wgpu::TextureSampleType::Float { filterable: false };
        let resolve_group = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aurora-shader temporal"),
            entries: &[
                entry(4, fragment, texture(float)),
                entry(5, fragment, texture(float)),
                entry(6, fragment, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
                entry(10, fragment, texture(float)),
                entry(11, fragment, texture(unfiltered)),
                entry(12, fragment, texture(wgpu::TextureSampleType::Depth)),
                entry(13, fragment, uniform(RESOLVE_SIZE)),
            ],
        });
        let resolve_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("aurora-shader temporal"),
            bind_group_layouts: &[Some(&resolve_group)],
            immediate_size: 0,
        });
        // two targets: the frame, and the history the next frame reads
        let resolve = device.create_render_pipeline_cached(&wgpu::RenderPipelineDescriptor {
            label: Some("aurora-shader temporal"),
            layout: Some(&resolve_layout),
            vertex: wgpu::VertexState {
                module: host,
                entry_point: Some("vs_fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: host,
                entry_point: Some(if format.is_srgb() { "fs_resolve_srgb" } else { "fs_resolve" }),
                compilation_options: Default::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: HISTORY_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });
        Finish {
            fxaa: host_pipeline(device, "aurora-shader fxaa", &layout, host, host, fxaa, format),
            masked: host_pipeline(device, "aurora-shader fxaa masked", &layout, host, host, masked, format),
            group,
            resolve_group,
            resolve,
            resolve_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("aurora-shader temporal"),
                size: RESOLVE_SIZE,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        }
    }
}

/// Takes the finished build of an effect, if any.
fn harvest(g: &mut EffectGpu) {
    let name = g.name.clone();
    let Some((hash, rx)) = &g.building else { return };
    let hash = *hash;
    let result = match rx.try_recv() {
        Ok(result) => result,
        Err(mpsc::TryRecvError::Empty) => return,
        Err(mpsc::TryRecvError::Disconnected) => Err("the build worker stopped".to_owned()),
    };
    g.building = None;
    match result {
        Ok(built) => {
            let c = &built.compiled;
            log::info!("aurora-shader: {} ({}) ready", c.file, c.title.as_deref().unwrap_or("untitled"));
            g.built = Some(built);
            g.reset = true;
            g.failed = None;
        }
        Err(e) => {
            log::warn!(
                "aurora-shader: {name}.wgsl: the GPU refused it: {e}; {}",
                if g.built.is_some() {
                    "the previous version keeps running"
                } else {
                    "skipped until it is fixed"
                }
            );
            g.failed = Some((hash, e));
        }
    }
}

/// Builds an effect's pipelines on a worker thread, inside error scopes
/// (thread-local in wgpu 30 with its `std` feature: they catch this
/// thread's errors only, not the frame's), with a clone of the device.
fn start_build(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    host: &wgpu::ShaderModule,
    compiled: Arc<Compiled>,
    format: wgpu::TextureFormat,
) -> mpsc::Receiver<Result<Built, String>> {
    let (tx, rx) = mpsc::sync_channel(1);
    let (device, layout, host) = (device.clone(), layout.clone(), host.clone());
    let spawned = std::thread::Builder::new()
        .name("aurora-shader-build".into())
        .stack_size(8 << 20)
        .spawn(move || {
            let built = catch_unwind(AssertUnwindSafe(|| build(&device, &layout, &host, compiled, format)))
                .unwrap_or_else(|_| Err("internal error while building".to_owned()));
            // (the receiver is gone when the effect was removed meanwhile)
            let _ = tx.send(built);
        });
    if let Err(e) = spawned {
        // the receiver reports the worker gone: a failed build
        log::warn!("aurora-shader: cannot start a build worker: {e}");
    }
    rx
}

fn build(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    host: &wgpu::ShaderModule,
    compiled: Arc<Compiled>,
    format: wgpu::TextureFormat,
) -> Result<Built, String> {
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&compiled.file),
        source: wgpu::ShaderSource::Wgsl(compiled.source.as_ref().into()),
    });
    let inter = host_pipeline(device, &compiled.file, layout, host, &module, "fs_main", INTER);
    let out = (!format.is_srgb()).then(|| host_pipeline(device, &compiled.file, layout, host, &module, "fs_main", format));
    let prepare = compiled.prepare.then(|| {
        device.create_compute_pipeline_cached(&wgpu::ComputePipelineDescriptor {
            label: Some(&compiled.file),
            layout: Some(layout),
            module: &module,
            entry_point: Some("prepare"),
            compilation_options: Default::default(),
            cache: None,
        })
    });
    let internal = pollster::block_on(internal.pop());
    let validation = pollster::block_on(validation.pop());
    if let Some(e) = validation.or(internal) {
        return Err(e.to_string().split_whitespace().collect::<Vec<_>>().join(" "));
    }
    Ok(Built {
        hash: compiled.hash,
        compiled,
        inter,
        out,
        prepare,
    })
}

/// GPU times of the chain (AURORA_PROFILE, devices with TIMESTAMP_QUERY):
/// pass-level timestamps resolved into one of three read-back buffers,
/// mapped the next frame (after the frame's submission), averaged and
/// logged every two seconds.
struct Timer {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    slots: Vec<TimerSlot>,
    period_ns: f32,
    /// Per label, total ms; and frames measured.
    sums: Vec<(String, f64)>,
    frames: u32,
    pixels: (u32, u32),
    since: Instant,
}

#[derive(PartialEq)]
enum SlotState {
    Free,
    /// Copied into by a frame not yet known to be submitted.
    Recorded,
    Mapping,
}

struct TimerSlot {
    buffer: wgpu::Buffer,
    ready: Arc<AtomicBool>,
    state: SlotState,
    /// The label of each pass (queries 2i, 2i + 1).
    labels: Vec<String>,
}

impl Timer {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Timer {
        let bytes = u64::from(MAX_TIMED_PASSES) * 2 * 8;
        Timer {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("aurora-shader timer"),
                ty: wgpu::QueryType::Timestamp,
                count: MAX_TIMED_PASSES * 2,
            }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("aurora-shader timer"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            slots: (0..3)
                .map(|_| TimerSlot {
                    buffer: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("aurora-shader timer readback"),
                        size: crate::readback::readback_size(bytes),
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                    ready: Arc::new(AtomicBool::new(false)),
                    state: SlotState::Free,
                    labels: Vec::new(),
                })
                .collect(),
            period_ns: queue.get_timestamp_period(),
            sums: Vec::new(),
            frames: 0,
            pixels: (0, 0),
            since: Instant::now(),
        }
    }

    /// Maps what the last frames recorded, reads what is mapped, logs.
    fn harvest(&mut self) {
        for slot in &mut self.slots {
            match slot.state {
                SlotState::Recorded => {
                    slot.ready.store(false, Ordering::Release);
                    let ready = slot.ready.clone();
                    slot.buffer
                        .map_async(wgpu::MapMode::Read, .., move |r| ready.store(r.is_ok(), Ordering::Release));
                    slot.state = SlotState::Mapping;
                }
                SlotState::Mapping if slot.ready.load(Ordering::Acquire) => {
                    if let Ok(view) = slot.buffer.get_mapped_range(..) {
                        let ts = timestamps(&view);
                        for (i, label) in slot.labels.iter().enumerate() {
                            let (Some(&a), Some(&b)) = (ts.get(2 * i), ts.get(2 * i + 1)) else {
                                break;
                            };
                            let ms = b.saturating_sub(a) as f64 * f64::from(self.period_ns) / 1e6;
                            match self.sums.iter_mut().find(|(l, _)| l == label) {
                                Some((_, sum)) => *sum += ms,
                                None => self.sums.push((label.clone(), ms)),
                            }
                        }
                        self.frames += 1;
                    }
                    slot.buffer.unmap();
                    slot.state = SlotState::Free;
                }
                _ => {}
            }
        }
        if self.since.elapsed() >= PROFILE_EVERY && self.frames > 0 {
            let n = f64::from(self.frames);
            let total: f64 = self.sums.iter().map(|(_, s)| s).sum::<f64>() / n;
            let parts: Vec<String> = self.sums.iter().map(|(l, s)| format!("{l} {:.3} ms", s / n)).collect();
            log::info!(
                "aurora-shader gpu: {}, total {total:.3} ms ({} frames, {}x{})",
                parts.join(", "),
                self.frames,
                self.pixels.0,
                self.pixels.1
            );
            self.sums.clear();
            self.frames = 0;
            self.since = Instant::now();
        }
    }
}

/// The timestamps of a mapped readback, copied out as `u64`s. The mapped
/// bytes are not 8-aligned on every driver (wgpu only aligns the offset
/// inside the buffer, not the host pointer): `bytemuck::cast_slice` to
/// `&[u64]` panics there, so the bytes are copied into an aligned `Vec`.
fn timestamps(bytes: &[u8]) -> Vec<u64> {
    let n = bytes.len() / 8;
    let mut ts = vec![0u64; n];
    bytemuck::cast_slice_mut::<u64, u8>(&mut ts).copy_from_slice(&bytes[..n * 8]);
    ts
}

/// The user effect chain of one `Backend` (each VR eye has its own).
#[derive(Default)]
pub(crate) struct UserPost {
    generation: u64,
    library: Option<Arc<Library>>,
    gpu: Option<Gpu>,
    /// This frame's effects, as indices into `gpu.effects`.
    steps: Vec<usize>,
    /// This frame draws the normals.
    normals: bool,
    params_dirty: bool,
    context: Context,
    idle: u32,
    last_frame: Option<Instant>,
    delta: f32,
    /// The temporal finish's history this frame reads (`Targets::history`
    /// index); the other one is written.
    parity: usize,
    /// The history holds the last frame's result for the same chain, AA
    /// and size: false starts it over.
    history_kept: bool,
    /// The chain, its AA and its place (its own slot, or its photo look's)
    /// of the last frame drawn, which the history belongs to.
    history_of: (Vec<usize>, ChainAa, bool),
    /// This frame draws the chain at its photo look's place
    /// ([`Self::encode_layer`]), not at its own slot.
    layer: bool,
}

impl UserPost {
    /// Before the post pass: whether the chain draws this frame. Then the
    /// post pass draws into [`Self::input`] and [`Self::encode`] draws the
    /// chain into the frame, or into SMAA's input under SMAA (`format` is
    /// then SMAA's). Without effects: one atomic load.
    // (52-shader-photo-look calls `prepare_for` in its place)
    #[allow(dead_code)]
    pub(crate) fn prepare(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat, size: (u32, u32), context: Context) -> bool {
        self.prepare_for(device, format, size, context, false)
    }

    /// [`Self::prepare`]; `layer`: the frame's look stack holds the photo
    /// look "Otoa's Shader" ([`Place::Layer`]). True then means the chain
    /// draws there, through [`Self::encode_layer`], and not at its own slot
    /// (the post pass keeps its target, [`Self::encode`] is not called).
    pub(crate) fn prepare_for(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        size: (u32, u32),
        context: Context,
        layer: bool,
    ) -> bool {
        let generation = library::generation();
        if generation != self.generation {
            self.generation = generation;
            self.library = library::current();
            self.params_dirty = true;
        }
        self.prepare_layered(device, format, size, context, layer)
    }

    #[cfg(test)]
    fn prepare_with(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat, size: (u32, u32), context: Context) -> bool {
        self.prepare_layered(device, format, size, context, false)
    }

    fn prepare_layered(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        size: (u32, u32),
        context: Context,
        layer: bool,
    ) -> bool {
        let Some(library) = self
            .library
            .clone()
            .filter(|l| place(!l.effects.is_empty(), l.photos, l.vr, context, layer) != Place::Off)
        else {
            self.idle();
            return false;
        };
        if let Some(timer) = self.gpu.as_mut().and_then(|g| g.timer.as_mut()) {
            timer.harvest();
        }
        if self.gpu.as_ref().is_some_and(|g| g.format != format) {
            self.gpu = None;
        }
        let gpu = self.gpu.get_or_insert_with(|| Gpu::new(device, format));
        if context.aa != ChainAa::Off && gpu.finish.is_none() {
            gpu.finish = Some(Finish::new(device, &gpu.host, format));
        }
        self.params_dirty |= gpu.sync(device, &library);
        // in the library's order, those with pipelines (a first version
        // still building joins a few frames later)
        self.steps.clear();
        let mut normals = false;
        for e in &library.effects {
            if let Some(i) = gpu.effects.iter().position(|g| g.name == e.name && g.built.is_some()) {
                normals |= gpu.effects[i].built.as_ref().is_some_and(|b| b.compiled.reads_normals);
                self.steps.push(i);
            }
        }
        if self.steps.is_empty() {
            self.idle();
            return false;
        }
        let finish = context.aa != ChainAa::Off;
        let temporal = context.aa == ChainAa::Temporal;
        let size = (size.0.max(1), size.1.max(1));
        let fresh = gpu.ensure_targets(device, size, self.steps.len(), normals, finish, temporal, layer);
        // the history goes with the image it was made for: a new one, a
        // frame without the chain, another chain, another AA or another
        // place start over (as does an effect's new version, in `encode`)
        let same = self.history_of.0 == self.steps && self.history_of.1 == context.aa && self.history_of.2 == layer;
        self.history_kept &= temporal && !fresh && same && self.last_frame.is_some();
        if !same {
            self.history_of = (self.steps.clone(), context.aa, layer);
        }
        self.layer = layer;
        self.normals = normals;
        self.context = context;
        self.idle = 0;
        let now = Instant::now();
        self.delta = self
            .last_frame
            .map_or(0.0, |t| now.duration_since(t).as_secs_f32().clamp(0.0, 0.25));
        self.last_frame = Some(now);
        true
    }

    /// A frame without the chain. With no effect at all (`[host] enabled =
    /// false`, an empty `order`, no folder) everything goes at once: the
    /// images, the pipelines, the buffers; the frame is then the unpatched
    /// one, at the cost of the generation check of `prepare`. Effects left
    /// out of this frame only (a photo without `[host] photos`, a VR eye)
    /// keep their pipelines, and their images go after a while.
    fn idle(&mut self) {
        self.last_frame = None;
        if self.library.as_ref().is_none_or(|l| l.effects.is_empty()) {
            self.idle = 0;
            if let Some(gpu) = self.gpu.take() {
                let (images, bytes) = gpu.footprint();
                log::info!(
                    "aurora-shader: off; released {} pipeline(s), {images} image(s) ({:.1} MiB) and the host's buffers",
                    gpu.effects.len(),
                    bytes as f64 / (1024.0 * 1024.0)
                );
            }
            return;
        }
        self.idle = self.idle.saturating_add(1);
        if self.idle > KEEP_FRAMES
            && let Some(gpu) = &mut self.gpu
        {
            gpu.targets = None;
        }
    }

    /// The image the post pass draws into while the chain is on (after
    /// `prepare` returned true).
    pub(crate) fn input(&self) -> Option<&wgpu::TextureView> {
        self.gpu.as_ref()?.targets.as_ref().map(|t| &t.input)
    }

    /// Draws the chain from the input into `out` (the chain's format and
    /// size), then its AA finish. Returns the passes drawn.
    pub(crate) fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        f: &FrameInputs,
        out: &wgpu::TextureView,
    ) -> u32 {
        self.encode_from(device, queue, encoder, f, None, out)
    }

    /// At the place of the photo look "Otoa's Shader" (after
    /// `prepare_for(.., layer: true)` returned true): the chain from `src`,
    /// the image the looks before it left (8-bit, gamma-encoded, the chain's
    /// size), into its own image, then its AA finish. Returns that image as
    /// the looks read it, and the passes drawn; None when the chain does not
    /// draw there this frame (the look then leaves the image as it is).
    pub(crate) fn encode_layer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        f: &FrameInputs,
        src: &wgpu::TextureView,
    ) -> Option<(wgpu::TextureView, u32)> {
        if !self.layer {
            return None;
        }
        let [out, read] = self.gpu.as_ref()?.targets.as_ref()?.layer_out.clone()?;
        let passes = self.encode_from(device, queue, encoder, f, Some(src), &out);
        (passes > 0).then_some((read, passes))
    }

    /// [`Self::encode`] from `src` instead of the input when given.
    fn encode_from(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        f: &FrameInputs,
        src: Option<&wgpu::TextureView>,
        out: &wgpu::TextureView,
    ) -> u32 {
        let (Some(gpu), Some(library)) = (&mut self.gpu, &self.library) else {
            return 0;
        };
        if self.params_dirty {
            for e in &library.effects {
                if let Some(g) = gpu.effects.iter().find(|g| g.name == e.name) {
                    queue.write_buffer(&g.params, 0, &e.params);
                }
            }
            self.params_dirty = false;
        }
        // the water depth's image, the first frame with water (a resize
        // makes new targets, so it follows the size)
        if f.water.is_some()
            && let Some(t) = &mut gpu.targets
            && t.depth.is_none()
        {
            t.depth = Some(make_texture(device, "aurora-shader depth", t.size, DEPTH_FORMAT, &[]).create_view(&Default::default()));
        }
        if profiling() && gpu.timer.is_none() && device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            gpu.timer = Some(Timer::new(device, queue));
        }
        let Gpu {
            ref group0,
            ref normals_group,
            ref normals_pipeline,
            ref depth_group,
            ref depth_pipeline,
            ref blit,
            ref finish,
            ref linear,
            ref point,
            ref frames,
            ref blank,
            ref mut effects,
            ref targets,
            ref mut timer,
            ..
        } = *gpu;
        let Some(t) = targets.as_ref() else { return 0 };
        let size = t.size;
        // what the first effect reads, and the finish compares with
        let input = src.unwrap_or(&t.input_read);

        // the frame's uniforms: slot 0 for the normals, one per effect
        let mut flags = 0;
        flags |= if f.mesh_normals.is_some() { FLAG_GEOMETRY_NORMALS } else { 0 };
        flags |= if f.taa { FLAG_TAA } else { 0 };
        flags |= if self.context.photo { FLAG_PHOTO } else { 0 };
        flags |= if self.context.vr_eye { FLAG_VR_EYE } else { 0 };
        let mut bytes = vec![0u8; (FRAME_STRIDE * (1 + self.steps.len() as u64)) as usize];
        for slot in 0..=self.steps.len() {
            let effect = if slot > 0 { effects.get(self.steps[slot - 1]) } else { None };
            let reset = effect.is_some_and(|g| g.reset);
            // a v1 effect sees the flags and `api` it always saw
            let api = effect.and_then(|g| g.built.as_ref()).map_or(1, |b| b.compiled.api);
            let alpha = api >= 2 && f.alpha_depth.is_some();
            let mut slot_flags = flags;
            slot_flags |= if reset { FLAG_STATE_RESET } else { 0 };
            slot_flags |= if alpha { FLAG_ALPHA_DEPTH } else { 0 };
            let u = FrameU::new(f, size, self.delta, slot_flags, api);
            let at = slot * FRAME_STRIDE as usize;
            bytes[at..at + FRAME_SIZE as usize].copy_from_slice(bytemuck::bytes_of(&u));
        }
        queue.write_buffer(frames, 0, &bytes);
        let frame_binding = |slot: usize| {
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: frames,
                offset: slot as u64 * FRAME_STRIDE,
                size: wgpu::BufferSize::new(FRAME_SIZE),
            })
        };

        // timestamps of each pass, into a free read-back slot
        let slot = timer
            .as_mut()
            .and_then(|t| t.slots.iter().position(|s| s.state == SlotState::Free).map(|i| (t, i)));
        let queries = slot.as_ref().map(|(t, _)| &t.queries);
        let mut labels: Vec<String> = Vec::new();
        let mut timed = |label: &str| -> Option<(u32, u32)> {
            queries?;
            labels.push(label.to_owned());
            let i = labels.len() as u32 - 1;
            Some((2 * i, 2 * i + 1))
        };
        let render_ts = |q: Option<(u32, u32)>| {
            q.zip(queries).map(|((b, e), qs)| wgpu::RenderPassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            })
        };
        let compute_ts = |q: Option<(u32, u32)>| {
            q.zip(queries).map(|((b, e), qs)| wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(b),
                end_of_pass_write_index: Some(e),
            })
        };
        let mut passes = 0u32;
        let draw = |encoder: &mut wgpu::CommandEncoder,
                    label: &str,
                    target: &wgpu::TextureView,
                    pipeline: &wgpu::RenderPipeline,
                    groups: &[&wgpu::BindGroup],
                    ts: Option<wgpu::RenderPassTimestampWrites>| {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: ts,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            for (i, g) in groups.iter().enumerate() {
                pass.set_bind_group(i as u32, *g, &[]);
            }
            pass.draw(0..3, 0..1);
        };

        // ---- the effects' depth: the scene's with the nearest visible water
        // surface in it (the water pass writes no depth), above water only
        let depth = match (&f.water, &t.depth) {
            (Some(water), Some(view)) => {
                let tex = wgpu::BindingResource::TextureView;
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("aurora-shader depth"),
                    layout: depth_group,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: frame_binding(0),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: tex(f.depth),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: tex(water.interface),
                        },
                        wgpu::BindGroupEntry {
                            binding: 9,
                            resource: tex(water.mask),
                        },
                    ],
                });
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("aurora-shader depth"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: render_ts(timed("depth")),
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(depth_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..3, 0..1);
                drop(pass);
                passes += 1;
                view
            }
            _ => f.depth,
        };

        // ---- normals, when an effect reads them
        let normals_view = match (&t.normals, self.normals) {
            (Some(view), true) => {
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("aurora-shader normals"),
                    layout: normals_group,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(depth),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(f.mesh_normals.unwrap_or(blank)),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: frame_binding(0),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: wgpu::BindingResource::TextureView(f.depth),
                        },
                    ],
                });
                let ts = render_ts(timed("normals"));
                draw(encoder, "aurora-shader normals", view, normals_pipeline, &[&bg], ts);
                passes += 1;
                view
            }
            _ => blank,
        };

        // ---- the effects, each reading the previous one's image
        let n = self.steps.len();
        let aa = self.context.aa;
        // the last effect draws the frame unless a pass follows it
        let last_pass = aa != ChainAa::Off || blit.is_some();
        for (i, &k) in self.steps.iter().enumerate() {
            let Some(g) = effects.get(k) else { continue };
            let Some(built) = &g.built else { continue };
            let src = if i == 0 { Some(input) } else { t.inter[(i - 1) % 2].as_ref() };
            let direct = i + 1 == n && !last_pass;
            let (dst, pipeline) = if direct {
                (Some(out), built.out.as_ref())
            } else {
                (t.inter[i % 2].as_ref(), Some(&built.inter))
            };
            let (Some(src), Some(dst), Some(pipeline)) = (src, dst, pipeline) else {
                continue;
            };
            if g.reset {
                encoder.clear_buffer(&g.state, 0, None);
            }
            let tex = wgpu::BindingResource::TextureView;
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("aurora-shader host"),
                layout: group0,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: tex(src),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: tex(depth),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: tex(normals_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(linear),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(point),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: frame_binding(i + 1),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: g.state.as_entire_binding(),
                    },
                    // without the blended faces, the effects' own depth: the
                    // same as `depth` everywhere, so nothing is in front (no
                    // image to make, and always the right size)
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: tex(f.alpha_depth.unwrap_or(depth)),
                    },
                ],
            });
            let name = g.name.as_str();
            if let Some(prepare) = &built.prepare {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("aurora-shader prepare"),
                    timestamp_writes: compute_ts(timed(name)),
                });
                pass.set_pipeline(prepare);
                pass.set_bind_group(0, &bg, &[]);
                pass.set_bind_group(1, &g.params_bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
                drop(pass);
                passes += 1;
            }
            let ts = render_ts(timed(name));
            draw(encoder, name, dst, pipeline, &[&bg, &g.params_bg], ts);
            passes += 1;
        }
        // ---- the AA finish, else on sRGB frames the decoding blit
        let last = t.inter[(n - 1) % 2].as_ref().filter(|_| last_pass);
        let finish = finish.as_ref().filter(|_| aa != ChainAa::Off);
        let history = t.history.as_ref().filter(|_| aa == ChainAa::Temporal);
        if let (Some(finish), Some(last), Some(history)) = (finish, last, history) {
            // what the chain changed, resolved over the frames: from one
            // history into the other, the frame beside it
            let reset = !self.history_kept || f.cut || self.steps.iter().any(|&k| effects.get(k).is_some_and(|g| g.reset));
            let jitter = chain_jitter(f, size);
            let u = ResolveU {
                reproject: f.reproject.to_cols_array(),
                jitter: [jitter.x, jitter.y, 0.0, 0.0],
                params: [
                    TEMPORAL_WEIGHT,
                    if reset { 1.0 } else { 0.0 },
                    if f.motion.is_some() { 1.0 } else { 0.0 },
                    0.0,
                ],
            };
            queue.write_buffer(&finish.resolve_buffer, 0, bytemuck::bytes_of(&u));
            let read = self.parity;
            let tex = wgpu::BindingResource::TextureView;
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("aurora-shader temporal"),
                layout: &finish.resolve_group,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: tex(last),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: tex(input),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(linear),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: tex(&history[read]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: tex(f.motion.unwrap_or(blank)),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: tex(depth),
                    },
                    wgpu::BindGroupEntry {
                        binding: 13,
                        resource: finish.resolve_buffer.as_entire_binding(),
                    },
                ],
            });
            let attachment = |view| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("aurora-shader finish"),
                color_attachments: &[attachment(out), attachment(&history[1 - read])],
                depth_stencil_attachment: None,
                timestamp_writes: render_ts(timed(aa.label())),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&finish.resolve);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
            drop(pass);
            passes += 1;
            self.parity = 1 - read;
            self.history_kept = true;
        } else if let (Some(finish), Some(last)) = (finish, last) {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("aurora-shader finish"),
                layout: &finish.group,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(last),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(input),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(linear),
                    },
                ],
            });
            let pipeline = if aa == ChainAa::FxaaMasked { &finish.masked } else { &finish.fxaa };
            let ts = render_ts(timed(aa.label()));
            draw(encoder, "aurora-shader finish", out, pipeline, &[&bg], ts);
            passes += 1;
        } else if let (Some((group, pipeline)), Some(last)) = (blit, last) {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("aurora-shader blit"),
                layout: group,
                entries: &[wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(last),
                }],
            });
            let ts = render_ts(timed(ChainAa::Off.label()));
            draw(encoder, "aurora-shader blit", out, pipeline, &[&bg], ts);
            passes += 1;
        }
        for &k in &self.steps {
            if let Some(g) = effects.get_mut(k) {
                g.reset = false;
            }
        }
        if let Some((timer, i)) = slot
            && !labels.is_empty()
        {
            let count = labels.len() as u32 * 2;
            encoder.resolve_query_set(&timer.queries, 0..count, &timer.resolve, 0);
            encoder.copy_buffer_to_buffer(&timer.resolve, 0, &timer.slots[i].buffer, 0, Some(u64::from(count) * 8));
            timer.slots[i].labels = labels;
            timer.slots[i].state = SlotState::Recorded;
            timer.pixels = size;
        }
        passes
    }

    /// Tests: runs `library` without the watcher.
    #[cfg(test)]
    fn set_library(&mut self, library: Arc<Library>) {
        self.library = Some(library);
        self.params_dirty = true;
    }
}
