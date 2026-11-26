// aurora-local-patch: scene-impostors-render-module
//! Scene impostors (Preferences › Graphics › Quality › Distance and detail;
//! local patch set). With a long draw distance most of what a frame draws
//! is far away and hardly changes from one frame to the next. The objects
//! wholly beyond the impostor distance of a capture point (where the
//! camera was when the capture began) are drawn once into a cube panorama
//! around that point, colour and depth, lit, hazed and fogged as the main
//! pass draws them (`fs_reflect` with the probes, from the capture point),
//! one face a frame, into the slot behind the one shown. Each frame the
//! shown panorama is drawn into the main view with its depth
//! (scene_panorama.wgsl), traced from the camera's own place: the near
//! terrain, water and objects pass in front of the far ones or behind them
//! as they should, and a far building stays on its far hill. Its depth goes
//! into the single-sample depth right after the prepass
//! ([`ScenePanorama::encode_depth`]): the ambient occlusion and the contact
//! shadows see the far objects as before.
//!
//! The objects it stands in for leave the main view in the GPU culling
//! (`face_mask` in cull.wgsl, [`face_panorama`] here); their shadows,
//! reflections and probes are left as they were. Terrain, water, avatars
//! and what they wear, animated objects, HUDs, blended and glowing faces,
//! the selection and the highlights never go into it
//! (`object_flags::NO_IMPOSTOR`, set by the viewer), nor do particles.
//!
//! The viewer stamps its GPU object table (`CullObject::stamp`, [`restamp`]
//! keeps the stamp of an entry written again unchanged, [`detail_stamp`]
//! that of an object that only swapped its level of detail): a capture only
//! takes, and the panorama only stands in for, the objects unchanged since
//! that capture began. What moves, or arrived or changed later, is drawn as
//! usual until a capture takes it, and an object a panorama holds that
//! moves, goes or changes makes a new capture due at once. Clicks and the
//! hover find the far objects as before: on the CPU from a depth that holds
//! the panorama, and in a drawn pick (the faces under one pixel) from bin
//! 30, which holds on that frame what the panorama stands in for under the
//! pixel ([`face_panorama`]). The panorama stays shown on every frame (a
//! frame without it would draw the far field live, a visible pop with each
//! pick of the hover); that frame captures no face, and a frame that draws
//! a probe shadow map (bin 30 taken) makes the pick wait one frame
//! ([`ScenePanorama::pick_waits`]).
//!
//! The panorama begins at the impostor distance, even where the sun's
//! shadows reach (they are not drawn in it: the far objects it holds lose
//! the sun's shadows on them, [`start`]), and is drawn only above the water,
//! in the window's view (no photo, no headset).
//!
//! When the switch is off, or the impostor distance is not short of the
//! draw distance, nothing here is allocated or drawn and the culling leaves
//! everything in place: the frame is the one drawn without it.

use crate::gpu_cull::{CullFrame, CullObject, NO_OBJECT, frame_flags, object_flags, sphere_in, view_planes};
use crate::pipeline_cache::CachedPipelines;
use crate::types::FrameParams;
use bytemuck::{Pod, Zeroable};
use glam::{DVec3, Mat4, Vec3};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

/// Near plane of the captures (m): what they hold is past 64 m.
pub const NEAR: f32 = 1.0;
/// A new capture once the camera is this part of the impostor distance away
/// from the capture point. The panorama is traced from the camera's place,
/// so all that is left is what the moved eye would see behind the edge of a
/// far object: at most that move over the object's distance, so 0.01 rad
/// against the sky for an object right at the impostor distance (about
/// 10 px at 1080p with a 60° view, just before the refresh) and less the
/// farther it is. At the default 120 m that is a capture every 1.2 m,
/// about three a second at a walk, six faces each.
pub const MOVE: f32 = 0.01;
/// Past this part of the impostor distance the panorama is no longer shown
/// (a teleport, a camera jump): the far objects are drawn as usual until a
/// new one is complete.
pub const LEAVE: f32 = 0.10;
/// A new capture after this long anyway: what changed far away, the sky.
pub const MAX_AGE: Duration = Duration::from_secs(4);
/// The sun turned by more than this (degrees): a new capture at once.
pub const SUN_DEGREES: f32 = 2.0;
/// A sky or fog colour changed by more than this part of itself: a new
/// capture at once.
pub const SKY_CHANGE: f32 = 0.05;
/// The wanted face size changed by more than this part: a new capture.
pub const SIZE_CHANGE: f32 = 0.25;
pub const MIN_SIZE: u32 = 256;
/// 6 faces of colour (8 bytes a texel) and depth (4) in each of the two
/// slots: 604 MB at this size, 531 MB at 1080p with a 60° view (1920 px).
pub const MAX_SIZE: u32 = 2048;
/// The main view's light-mode bits the captures shade with too: its tangent
/// frame and its specular anti-aliasing (renderer.rs; common.wgsl
/// TANGENT_VERTEX, TANGENT_UNIT, SPECULAR_AA).
pub const MAIN_SHADING_BITS: u32 = 0x1800 | 0x8000;
/// The objects a panorama never holds (`object_flags::NO_IMPOSTOR` is set by
/// the viewer: blended and glowing faces, animated textures, water
/// exclusion, the selection and the highlights).
const KEPT: u32 = object_flags::AVATAR | object_flags::RIGGED | object_flags::NO_PROBE | object_flags::NO_IMPOSTOR;
/// Another level of detail of an object may move its bounds by this part
/// of its radius and still be the same object for a panorama
/// ([`detail_stamp`]).
const DETAIL_SLACK: f32 = 0.25;
/// The coverage a pixel needs without multisampling; with it the colour
/// pass turns the coverage into samples (alpha to coverage).
const COVER_SINGLE: f32 = 0.5;
const COVER_MSAA: f32 = 1.0 / 32.0;

/// What the viewer asks of a frame (`FrameParams::panorama`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PanoramaParams {
    /// The rule (the viewer's `scene_impostors::active`): the switch on and
    /// an impostor distance short of the draw distance, in the world view.
    pub active: bool,
    /// Impostor distance (m).
    pub distance: f32,
    /// Draw distance (m).
    pub draw_distance: f32,
    /// Changes with the main region (a crossing, a teleport): a new capture.
    pub epoch: u32,
    /// Global position of the render origin (m). The captures keep their
    /// point in global coordinates: a crossing that moves the origin keeps
    /// the panorama where it is in the world.
    pub origin: [f64; 2],
    /// The stamp of this frame's object table (`CullObject::stamp`).
    pub stamp: u32,
}

/// What a capture is made with, compared with the frame's for a refresh.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Look {
    pub distance: f32,
    pub draw_distance: f32,
    pub size: u32,
    pub epoch: u32,
    pub sun: Vec3,
    /// Sun, zenith, horizon, ground and fog colours.
    pub colours: [Vec3; 5],
    pub fog_density: f32,
    pub water_height: f32,
    pub classic: bool,
    pub no_post: bool,
}

/// Where the panorama begins (m): the impostor distance, with the sun's
/// shadows on too. A capture draws no sun shadows, so the far objects
/// between the impostor distance and the shadow distance lose the sun's
/// shadows on them; the shadows they cast on the live ground stay (the
/// shadow map still draws them, and the contact shadows and the ambient
/// occlusion read the panorama's depth, [`ScenePanorama::encode_depth`]).
/// Otoa's choice (2026-11-09), over starting it past the shadows' reach
/// (shadow distance / 0.9: 284 m with shadows to 256 m). The shadow
/// arguments stay so that the call says what is ignored.
pub fn start(distance: f32, _shadows: bool, _shadow_distance: f32) -> f32 {
    distance
}

impl Look {
    pub fn of(f: &FrameParams, size: u32) -> Look {
        Look {
            distance: start(f.panorama.distance, f.shadows, f.shadow_distance),
            draw_distance: f.panorama.draw_distance,
            size,
            epoch: f.panorama.epoch,
            sun: f.sun_dir.normalize_or(Vec3::Z),
            colours: [f.sun_color, f.sky_zenith, f.sky_horizon, f.ground_color, f.fog_color],
            fog_density: f.fog_density,
            water_height: f.water_height,
            classic: f.sky.classic,
            no_post: f.sky.no_post,
        }
    }
}

/// Whether a capture made under `then` is out of date under `now`: the
/// distances, the region, the face size, the sun or the sky changed.
pub fn outdated(then: &Look, now: &Look) -> bool {
    let colour_moved = |(a, b): (&Vec3, &Vec3)| (*a - *b).length() > SKY_CHANGE * a.length().max(b.length()).max(0.02);
    then.distance != now.distance
        || then.draw_distance != now.draw_distance
        || then.epoch != now.epoch
        || then.classic != now.classic
        || then.no_post != now.no_post
        || (then.size as f32 - now.size as f32).abs() > SIZE_CHANGE * then.size as f32
        || then.sun.dot(now.sun) < SUN_DEGREES.to_radians().cos()
        || then.colours.iter().zip(&now.colours).any(colour_moved)
        || (then.fog_density - now.fog_density).abs() > SKY_CHANGE * then.fog_density.abs().max(now.fog_density.abs()).max(1e-4)
        || (then.water_height - now.water_height).abs() > 0.05
}

/// A panorama: where and when its capture began, and under what.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Made {
    /// Capture point, global (m).
    pub centre: DVec3,
    pub look: Look,
    pub at: Instant,
    /// The objects older than this stamp are in it (`CullObject::stamp`).
    pub stamp: u32,
    /// [`changes`] when its capture began.
    pub changes: u32,
}

/// Whether the panorama `made` wants a new capture, the camera at `eye`
/// (global) under `look`, `changes` objects of a panorama having changed
/// so far ([`changes`]).
pub fn refresh_due(made: &Made, eye: DVec3, look: &Look, changes: u32, now: Instant) -> bool {
    (eye - made.centre).length() > f64::from(MOVE * made.look.distance)
        || now.saturating_duration_since(made.at) >= MAX_AGE
        || made.changes != changes
        || outdated(&made.look, look)
}

/// The panorama shown and the one being captured as the culling tests them
/// (`in_panorama`), for [`restamp`] on the main thread: the capture point
/// (render coordinates) and where the panorama begins, its draw distance
/// and its stamp, as f32 / u32 bits (all 0: none).
static SETS: [[AtomicU32; 6]; 2] = [const { [const { AtomicU32::new(0) }; 6] }; 2];
/// Objects held by a panorama that changed since (`restamp`).
static CHANGES: AtomicU32 = AtomicU32::new(0);

/// How many objects held by a panorama changed so far.
pub fn changes() -> u32 {
    CHANGES.load(Ordering::Relaxed)
}

fn publish(k: usize, set: Option<([f32; 4], f32, u32)>) {
    let (p, reach, stamp) = set.unwrap_or_default();
    let words = [p[0], p[1], p[2], p[3], reach].map(f32::to_bits);
    for (slot, v) in SETS[k].iter().zip(words.into_iter().chain([stamp])) {
        slot.store(v, Ordering::Relaxed);
    }
}

/// Whether a panorama (shown or being captured) holds an object with this
/// sphere and stamp.
fn held(sphere: [f32; 4], stamp: u32) -> bool {
    let c = Vec3::new(sphere[0], sphere[1], sphere[2]);
    SETS.iter().any(|s| {
        let f = |i: usize| f32::from_bits(s[i].load(Ordering::Relaxed));
        in_panorama([f(0), f(1), f(2), f(3)], f(4), s[5].load(Ordering::Relaxed), stamp, c, sphere[3])
    })
}

/// `GpuTable::set` of the object table (gpu_cull.rs): `new` replaces `old`.
/// An entry written again with nothing changed but its stamp keeps the old
/// one (the viewer writes the frame's stamp into every entry it writes: a
/// new level of detail or texture is still the object its panorama holds,
/// refreshed with it). An object a panorama holds that moved, went, or
/// changed its flags is drawn as usual from now on while the panorama
/// still holds its old picture: a new capture is due at once.
pub fn restamp(old: &CullObject, new: CullObject) -> CullObject {
    if new.stamp == old.stamp {
        return new;
    }
    let unchanged = CullObject { stamp: old.stamp, ..new } == *old;
    if unchanged {
        return *old;
    }
    let drawn = old.flags & object_flags::ACTIVE != 0 && old.flags & (object_flags::HUD | object_flags::HIDDEN | KEPT) == 0;
    if drawn && old.avatar == NO_OBJECT && old.anchor == 0 && held(old.sphere, old.stamp) {
        CHANGES.fetch_add(1, Ordering::Relaxed);
    }
    new
}

/// The stamp of an entry written again because the object swapped its
/// level of detail and nothing else (the viewer's `ObjGpu::lod_swap`; its
/// bounds are that level's): its old one when only the sphere changed, by
/// little, and no panorama gains or loses it by that, so that a panorama
/// holding it keeps it (its picture there holds, as for a new texture);
/// else the new one, a change like any other. Without it, every level of
/// detail the walk swaps far away would make a capture due at once.
pub fn detail_stamp(old: &CullObject, new: &CullObject) -> u32 {
    let centre = |o: &CullObject| Vec3::new(o.sphere[0], o.sphere[1], o.sphere[2]);
    let r = old.sphere[3];
    let only_sphere = CullObject {
        sphere: old.sphere,
        stamp: old.stamp,
        ..*new
    } == *old;
    let close = (centre(new) - centre(old)).length() <= DETAIL_SLACK * r && (new.sphere[3] - r).abs() <= DETAIL_SLACK * r;
    if only_sphere && close && held(new.sphere, old.stamp) == held(old.sphere, old.stamp) {
        old.stamp
    } else {
        new.stamp
    }
}

/// Whether the panorama `made` may still be shown with the camera at `eye`.
pub fn shown_from(made: &Made, eye: DVec3) -> bool {
    (eye - made.centre).length() <= f64::from(LEAVE * made.look.distance)
}

/// Side of the faces (px) for a view `height` pixels high with a vertical
/// field of view `fov_y`: as many pixels per radian in the middle of a
/// face (N / 2) as in the middle of the view (H / 2 / tan(fov / 2)), so a
/// far object is as sharp as it would be drawn directly; in steps of 64.
pub fn face_size(height: u32, fov_y: f32, max_dimension: u32) -> u32 {
    let tan = (fov_y.clamp(0.05, 3.0) * 0.5).tan().max(1e-3);
    let n = (height.max(1) as f32 / tan).ceil() as u32;
    let cap = max_dimension.clamp(MIN_SIZE, MAX_SIZE);
    n.div_ceil(64).saturating_mul(64).clamp(MIN_SIZE, cap)
}

/// Projection of the captures: a square 90° face, reverse Z, infinite.
pub fn face_proj() -> Mat4 {
    glam::camera::rh::proj::directx::perspective_infinite_reverse(std::f32::consts::FRAC_PI_2, 1.0, NEAR)
}

/// `in_panorama` of cull.wgsl: the sphere lies wholly beyond the impostor
/// distance of a panorama's capture point (`p[3]`; 0: none) and within its
/// draw distance `reach`, its object unchanged since the capture began.
pub fn in_panorama(p: [f32; 4], reach: f32, stamp: u32, object_stamp: u32, c: Vec3, r: f32) -> bool {
    let near = (c - Vec3::new(p[0], p[1], p[2])).length() - r;
    p[3] > 0.0 && object_stamp < stamp && near > p[3] && near <= reach
}

/// The panorama part of `face_mask` in cull.wgsl (kept line for line):
/// whether the panorama drawn this frame stands in for the object (it
/// leaves the main view), and whether bin 30 takes it. `n`: the object
/// count. Bin 30 holds what the face captured this frame draws (it leaves
/// out what the main view's small object culling would, its pixel scale
/// from the capture point), or, on a frame with a drawn pick
/// (`panorama[2][2]` 1, `panorama_planes` the pick's narrowed view), what
/// the panorama stands in for under the pick's pixel, as the main view
/// would have drawn it there (its draw distance, its small object culling):
/// the pick still finds a far object. A frame never has both.
pub fn face_panorama(frame: &CullFrame, o: &CullObject, n: u32, c: Vec3, r: f32) -> (bool, bool) {
    let impostor = o.flags & KEPT == 0 && o.avatar >= n;
    let far_front = impostor && in_panorama(frame.panorama[0], frame.panorama[2][0], frame.panorama_stamps[0], o.stamp, c, r);
    let eye = Vec3::new(frame.panorama[1][0], frame.panorama[1][1], frame.panorama[1][2]);
    let free = frame.sizes[3] & frame_flags::PROBE_SHADOW == 0;
    let pano_face = impostor
        && free
        && in_panorama(frame.panorama[1], frame.panorama[2][1], frame.panorama_stamps[1], o.stamp, c, r)
        && sphere_in(&frame.panorama_planes, c, r)
        && r * frame.dist[2] / (c - eye).length().max(0.1) >= 1.5;
    let d = (c - Vec3::new(frame.eye[0], frame.eye[1], frame.eye[2])).length();
    let picked = far_front
        && free
        && frame.panorama[2][2] > 0.0
        && sphere_in(&frame.panorama_planes, c, r)
        && d - r <= frame.eye[3]
        && r * frame.dist[2] / d.max(0.1) >= 1.5;
    (far_front, pano_face || picked)
}

/// The panorama vectors of a `CullFrame` (all zero: none): the shown one's
/// capture point and start, the captured one's, then x the shown one's
/// draw distance, y the captured one's, z 1 when bin 30 holds the drawn
/// pick's part of the shown one; the planes of the face captured (of the
/// pick's view on such a frame); the stamps of the two.
pub type CullPanorama = ([[f32; 4]; 3], [[f32; 4]; 6], [u32; 4]);

/// The texture formats of the scene (renderer.rs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Formats {
    /// The scene's colour, which `fs_reflect` draws the captures in.
    pub colour: wgpu::TextureFormat,
    pub depth: wgpu::TextureFormat,
    /// The G-buffer of the screen-space reflections, when there is one.
    pub gbuf: Option<wgpu::TextureFormat>,
}

/// The frame, as [`ScenePanorama::plan`] needs it.
pub struct FrameState {
    pub params: PanoramaParams,
    /// The panorama can be used: GPU lists, the window's own frame (no
    /// photo, no cut-out), the camera above the water (the captures see the
    /// water's fog as an eye above it does).
    pub usable: bool,
    /// The narrowed view-projection (render coordinates) of this frame's
    /// drawn pick (a click, the hover over posed surfaces: the faces under
    /// one pixel, drawn again), if any. With a panorama shown, bin 30 then
    /// holds the faces it stands in for under that pixel, and no face is
    /// captured.
    pub pick: Option<Mat4>,
    /// Bin 30 is free: no probe shadow map is drawn this frame.
    pub capture: bool,
    /// Camera, render coordinates.
    pub eye: Vec3,
    pub look: Look,
    pub now: Instant,
    /// The main view's camera-relative view-projection, as the scene is
    /// drawn (with the TAA jitter).
    pub view_proj: Mat4,
    pub scene_size: (u32, u32),
    pub samples: u32,
    pub formats: Formats,
}

/// A face captured this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceCapture {
    pub face: u32,
    /// Capture point, render coordinates.
    pub centre: Vec3,
    pub size: u32,
    slot: usize,
}

/// What a frame does with the panorama.
#[derive(Debug, Clone, Copy, Default)]
pub struct Plan {
    pub cull: CullPanorama,
    pub capture: Option<FaceCapture>,
    /// The panorama is drawn into the main view.
    pub shown: bool,
    /// Bin 30 holds the faces the shown panorama stands in for under the
    /// drawn pick's pixel: the pick pass draws it.
    pub pick: bool,
}

/// A capture under way.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Back {
    slot: usize,
    made: Made,
    /// Faces planned so far.
    faces: u32,
}

/// The two slots' schedule, without the GPU (tested alone).
#[derive(Debug, Default)]
struct Schedule {
    front: Option<(usize, Made)>,
    back: Option<Back>,
    /// A frame went by since the switch came on: the viewer's stamps reach
    /// the object table a frame after its parameters.
    warm: bool,
}

/// What a frame of the schedule does.
#[derive(Debug, Default, PartialEq)]
struct Step {
    /// Slot and face captured this frame, and that capture.
    capture: Option<(usize, u32, Made)>,
    /// The panorama shown this frame.
    front: Option<(usize, Made)>,
    /// Bin 30 is the drawn pick's this frame (a panorama is shown).
    pick: bool,
}

/// One frame of the schedule, the camera at `eye` (global) under `look`
/// with the object stamp `stamp` and [`changes`] `changes`. `pick`: the
/// frame draws a pick (bin 30 is its while a panorama is shown, no face is
/// captured); `capture`: bin 30 is free of a probe shadow map. `room(slot)`
/// makes the slot ready for a capture (false: the memory refused it,
/// nothing starts).
#[allow(clippy::too_many_arguments)]
fn step(
    s: &mut Schedule,
    usable: bool,
    pick: bool,
    capture: bool,
    eye: DVec3,
    look: &Look,
    (stamp, changes): (u32, u32),
    now: Instant,
    mut room: impl FnMut(usize) -> bool,
) -> Step {
    // a complete capture becomes the one shown (its last face went out with
    // the frame before, submitted before this one)
    if let Some(b) = s.back.filter(|b| b.faces >= 6) {
        s.front = Some((b.slot, b.made));
        s.back = None;
    }
    // too far from it (a teleport, a camera jump), or another region: the
    // render origin moved and every object was written again, all of them
    // drawn as usual until the next panorama is complete
    let stale = |made: &Made| !shown_from(made, eye) || made.look.epoch != look.epoch;
    if s.front.is_some_and(|(_, made)| stale(&made)) {
        s.front = None;
    }
    let warm = std::mem::replace(&mut s.warm, true);
    if !usable || !warm {
        return Step::default();
    }
    // a capture under way that the frame outdated begins again
    if s.back.is_some_and(|b| outdated(&b.made.look, look)) {
        s.back = None;
    }
    let due = s.front.is_none_or(|(_, made)| refresh_due(&made, eye, look, changes, now));
    if due && s.back.is_none() {
        let slot = match s.front {
            Some((0, _)) => 1,
            _ => 0,
        };
        if room(slot) {
            s.back = Some(Back {
                slot,
                made: Made {
                    centre: eye,
                    look: *look,
                    at: now,
                    stamp,
                    changes,
                },
                faces: 0,
            });
        }
    }
    let mut out = Step {
        front: s.front,
        pick: pick && s.front.is_some(),
        ..Step::default()
    };
    if capture
        && !out.pick
        && let Some(b) = &mut s.back
    {
        out.capture = Some((b.slot, b.faces, b.made));
        b.faces += 1;
    }
    out
}

/// Whether a panorama may be shown this frame: one is, or a capture is
/// complete (it becomes the one shown at the next [`step`]).
fn may_show(s: &Schedule) -> bool {
    s.front.is_some() || s.back.is_some_and(|b| b.faces >= 6)
}

/// The composite's uniforms (`Pano` in scene_panorama.wgsl).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct PanoU {
    vp: [[f32; 4]; 4],
    inv_vp: [[f32; 4]; 4],
    centre: [f32; 4],
    screen: [f32; 4],
}

/// The composite's shader, layouts and pipelines, made on first use.
struct Gpu {
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    module: wgpu::ShaderModule,
    uniform: wgpu::Buffer,
    /// Into the single-sample depth, after the prepass.
    depth: Option<wgpu::RenderPipeline>,
    /// Into the scene pass, by sample count and G-buffer.
    colour: Vec<((u32, Option<wgpu::TextureFormat>), wgpu::RenderPipeline)>,
}

impl Gpu {
    fn new(device: &wgpu::Device) -> Gpu {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene panorama"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/scene_panorama.wgsl").into()),
        });
        let texture = |binding, sample_type| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type,
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene panorama"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                texture(1, wgpu::TextureSampleType::Float { filterable: false }),
                texture(2, wgpu::TextureSampleType::Depth),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene panorama"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene panorama composite"),
            size: size_of::<PanoU>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Gpu {
            layout,
            pipeline_layout,
            module,
            uniform,
            depth: None,
            colour: Vec::new(),
        }
    }

    /// The colour pass of the scene pass (`Some((samples, gbuf))`), or the
    /// pass into the single-sample depth (`None`).
    fn pipeline(
        &self,
        device: &wgpu::Device,
        formats: &Formats,
        colour: Option<(u32, Option<wgpu::TextureFormat>)>,
    ) -> wgpu::RenderPipeline {
        let mut targets = Vec::new();
        if let Some((_, gbuf)) = colour {
            // (the alpha holds the glow amount: left as it is)
            targets.push(Some(wgpu::ColorTargetState {
                format: formats.colour,
                blend: None,
                write_mask: wgpu::ColorWrites::COLOR,
            }));
            if let Some(format) = gbuf {
                // nothing for the screen-space reflections: the sky's, rough
                targets.push(Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::empty(),
                }));
            }
        }
        let samples = colour.map_or(1, |(n, _)| n.max(1));
        device.create_render_pipeline_cached(&wgpu::RenderPipelineDescriptor {
            label: Some(if colour.is_some() {
                "scene panorama"
            } else {
                "scene panorama depth"
            }),
            layout: Some(&self.pipeline_layout),
            vertex: wgpu::VertexState {
                module: &self.module,
                entry_point: Some("vs_full"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.module,
                entry_point: Some(if colour.is_some() { "fs_colour" } else { "fs_depth" }),
                compilation_options: Default::default(),
                targets: &targets,
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: formats.depth,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: samples,
                mask: !0,
                // soft edges, sample by sample, with the depth of each
                alpha_to_coverage_enabled: samples > 1,
            },
            multiview_mask: None,
            cache: None,
        })
    }
}

/// One panorama: six faces of colour and depth, and what draws them.
struct Slot {
    size: u32,
    _colour: wgpu::Texture,
    _depth: wgpu::Texture,
    colour_faces: Vec<wgpu::TextureView>,
    depth_faces: Vec<wgpu::TextureView>,
    group: wgpu::BindGroup,
}

impl Slot {
    fn new(device: &wgpu::Device, gpu: &Gpu, size: u32, formats: &Formats) -> Slot {
        let texture = |label, format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 6,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let colour = texture("scene panorama colour", formats.colour);
        let depth = texture("scene panorama depth", formats.depth);
        let faces = |t: &wgpu::Texture| -> Vec<wgpu::TextureView> {
            (0..6)
                .map(|layer| {
                    t.create_view(&wgpu::TextureViewDescriptor {
                        label: Some("scene panorama face"),
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        base_array_layer: layer,
                        array_layer_count: Some(1),
                        ..Default::default()
                    })
                })
                .collect()
        };
        let all = |t: &wgpu::Texture| {
            t.create_view(&wgpu::TextureViewDescriptor {
                label: Some("scene panorama faces"),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        };
        let (colour_all, depth_all) = (all(&colour), all(&depth));
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene panorama"),
            layout: &gpu.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&colour_all),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&depth_all),
                },
            ],
        });
        Slot {
            size,
            colour_faces: faces(&colour),
            depth_faces: faces(&depth),
            _colour: colour,
            _depth: depth,
            group,
        }
    }

    /// Bytes of the two textures (12 a texel at the scene's formats).
    fn bytes(size: u32) -> u64 {
        u64::from(size) * u64::from(size) * 6 * 12
    }
}

/// The far-field panorama of the main view (owned by the render thread's
/// backend, renderer.rs).
#[derive(Default)]
pub struct ScenePanorama {
    slots: [Option<Slot>; 2],
    schedule: Schedule,
    gpu: Option<Gpu>,
    /// This frame: the slot drawn into the main view, and its colour
    /// pipeline (`Gpu::colour`).
    shown: Option<(usize, usize)>,
    backoff: crate::gpu_alloc::Backoff,
    /// A capture was complete since the switch came on (said once).
    said: bool,
}

impl ScenePanorama {
    /// What this frame does: the culling's vectors, the face it captures
    /// and whether the panorama is drawn (its uniforms written). Off, it
    /// drops everything it holds.
    pub fn plan(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, s: &FrameState) -> Plan {
        self.shown = None;
        let p = s.params;
        if !p.active || s.look.distance >= s.look.draw_distance {
            if self.gpu.is_some() || self.slots.iter().any(Option::is_some) {
                log::info!("scene impostors: off, panorama released");
            }
            if self.schedule.front.is_some() || self.schedule.back.is_some() {
                publish(0, None);
                publish(1, None);
            }
            let backoff = std::mem::take(&mut self.backoff);
            *self = ScenePanorama {
                backoff,
                ..Default::default()
            };
            return Plan::default();
        }
        let origin = DVec3::new(p.origin[0], p.origin[1], 0.0);
        let eye = origin + s.eye.as_dvec3();
        let was_shown = self.schedule.front.map(|(_, made)| made.at);
        let ScenePanorama {
            slots,
            gpu,
            backoff,
            schedule,
            ..
        } = self;
        let room = |slot: usize| {
            if slots[slot].as_ref().is_some_and(|x| x.size == s.look.size) {
                return true;
            }
            slots[slot] = None;
            if !backoff.ready(s.now) {
                return false;
            }
            let gpu = gpu.get_or_insert_with(|| Gpu::new(device));
            let (made, refused) = crate::gpu_alloc::scoped(device, || Slot::new(device, gpu, s.look.size, &s.formats));
            if refused {
                backoff.failed(s.now);
                log::warn!(
                    "scene impostors: no video memory for a panorama of {} px ({} MB): far objects drawn as usual",
                    s.look.size,
                    Slot::bytes(s.look.size) >> 20
                );
                return false;
            }
            backoff.succeeded();
            log::debug!(
                "scene impostors: panorama slot {slot}, {} px faces ({} MB)",
                s.look.size,
                Slot::bytes(s.look.size) >> 20
            );
            slots[slot] = Some(made);
            true
        };
        let step = step(
            schedule,
            s.usable,
            s.pick.is_some(),
            s.capture,
            eye,
            &s.look,
            (p.stamp, changes()),
            s.now,
            room,
        );
        // what the two panoramas hold, for `restamp` (whether shown or not)
        let set = |made: &Made| {
            (
                (made.centre - origin).as_vec3().extend(made.look.distance).to_array(),
                made.look.draw_distance,
                made.stamp,
            )
        };
        publish(0, self.schedule.front.map(|(_, made)| set(&made)));
        publish(1, self.schedule.back.map(|b| set(&b.made)));
        let mut plan = Plan::default();
        if let Some((slot, face, made)) = step.capture
            && let Some(x) = &self.slots[slot]
        {
            let centre = (made.centre - origin).as_vec3();
            plan.cull.0[1] = centre.extend(made.look.distance).to_array();
            plan.cull.0[2][1] = made.look.draw_distance;
            plan.cull.1 = view_planes(face_proj() * crate::probes::face_view(centre, face)).map(|v| v.to_array());
            plan.cull.2[1] = made.stamp;
            plan.capture = Some(FaceCapture {
                face,
                centre,
                size: x.size,
                slot,
            });
        }
        if let Some((slot, made)) = step.front
            && let Some(size) = self.slots[slot].as_ref().map(|x| x.size)
            && let Some(pipe) = self.pipelines(device, s)
            && let Some(gpu) = &self.gpu
        {
            if was_shown != Some(made.at) || !self.said {
                log::debug!(
                    "scene impostors: panorama of {size} px from ({:.0}, {:.0}, {:.0}) shown",
                    made.centre.x,
                    made.centre.y,
                    made.centre.z
                );
                if !self.said {
                    log::info!(
                        "scene impostors: the far field beyond {:.0} m is drawn from a {size} px panorama",
                        made.look.distance
                    );
                    self.said = true;
                }
            }
            plan.cull.0[0] = (made.centre - origin).as_vec3().extend(made.look.distance).to_array();
            plan.cull.0[2][0] = made.look.draw_distance;
            plan.cull.2[0] = made.stamp;
            let u = PanoU {
                vp: s.view_proj.to_cols_array_2d(),
                inv_vp: s.view_proj.as_dmat4().inverse().as_mat4().to_cols_array_2d(),
                centre: (made.centre - eye).as_vec3().extend(NEAR).to_array(),
                screen: [
                    s.scene_size.0.max(1) as f32,
                    s.scene_size.1.max(1) as f32,
                    size as f32,
                    if s.samples > 1 { COVER_MSAA } else { COVER_SINGLE },
                ],
            };
            queue.write_buffer(&gpu.uniform, 0, bytemuck::bytes_of(&u));
            self.shown = Some((slot, pipe));
            plan.shown = true;
            // a drawn pick: bin 30 holds the faces the panorama stands in
            // for under its pixel (face_panorama), and the panorama stays
            if step.pick
                && let Some(vp) = s.pick
            {
                plan.cull.0[2][2] = 1.0;
                plan.cull.1 = view_planes(vp).map(|v| v.to_array());
                plan.pick = true;
            }
        }
        plan
    }

    /// Whether this frame's drawn pick waits a frame: the frame draws a
    /// probe shadow map (`probe_shadow`), which takes bin 30, while a
    /// panorama may be shown, whose faces under the pick's pixel bin 30
    /// must hold. A probe draws its shadow map on its first face alone,
    /// so the pick waits one frame at most.
    pub fn pick_waits(&self, probe_shadow: bool) -> bool {
        probe_shadow && may_show(&self.schedule)
    }

    /// The colour pipeline of this frame's scene pass and the depth one,
    /// made the first time they are needed.
    fn pipelines(&mut self, device: &wgpu::Device, s: &FrameState) -> Option<usize> {
        let gpu = self.gpu.get_or_insert_with(|| Gpu::new(device));
        let key = (s.samples.max(1), s.formats.gbuf);
        if gpu.depth.is_none() {
            let pipe = gpu.pipeline(device, &s.formats, None);
            gpu.depth = Some(pipe);
        }
        if let Some(i) = gpu.colour.iter().position(|(k, _)| *k == key) {
            return Some(i);
        }
        let pipe = gpu.pipeline(device, &s.formats, Some(key));
        gpu.colour.push((key, pipe));
        Some(gpu.colour.len() - 1)
    }

    /// The colour and depth targets of the face captured this frame.
    pub fn capture_views(&self, c: &FaceCapture) -> Option<(&wgpu::TextureView, &wgpu::TextureView)> {
        let slot = self.slots.get(c.slot)?.as_ref()?;
        Some((slot.colour_faces.get(c.face as usize)?, slot.depth_faces.get(c.face as usize)?))
    }

    /// The panorama's depth into the single-sample depth, right after the
    /// prepass: the occlusion depth, the ambient occlusion and the contact
    /// shadows read it before the scene pass, so the far objects still cast
    /// their contact shadows and occlusion on the live ground. With MSAA the
    /// scene pass draws into a depth of its own and this one also serves
    /// the clicks, the hover and the effects; without, the scene pass
    /// starts from it (what lies behind the far objects is not shaded) and
    /// its composite writes their exact depth again.
    pub fn encode_depth(&self, encoder: &mut wgpu::CommandEncoder, depth: &wgpu::TextureView) {
        let Some((slot, _)) = self.shown else {
            return;
        };
        let (Some(slot), Some(pipe)) = (
            self.slots.get(slot).and_then(Option::as_ref),
            self.gpu.as_ref().and_then(|g| g.depth.as_ref()),
        ) else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("scene panorama depth"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, &slot.group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// The panorama drawn into the scene pass, after everything opaque
    /// (it binds a group 0 of its own: nothing may follow it in the pass).
    pub fn draw_composite(&self, pass: &mut wgpu::RenderPass<'_>) {
        let Some((slot, pipe)) = self.shown else {
            return;
        };
        let (Some(slot), Some((_, pipe))) = (
            self.slots.get(slot).and_then(Option::as_ref),
            self.gpu.as_ref().and_then(|g| g.colour.get(pipe)),
        ) else {
            return;
        };
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, &slot.group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_cull::{BIN_PROBE_SHADOW, CullFace, GpuCullView, face_bins, pass, world_spheres};

    fn look() -> Look {
        Look {
            distance: 256.0,
            draw_distance: 1024.0,
            size: 1920,
            epoch: 7,
            sun: Vec3::new(0.3, 0.2, 0.9).normalize(),
            colours: [
                Vec3::new(1.0, 0.9, 0.8),
                Vec3::new(0.2, 0.4, 0.9),
                Vec3::new(0.6, 0.7, 0.8),
                Vec3::splat(0.3),
                Vec3::new(0.5, 0.6, 0.7),
            ],
            fog_density: 0.002,
            water_height: 20.0,
            classic: false,
            no_post: false,
        }
    }

    #[test]
    fn the_faces_are_as_fine_as_the_view() {
        // 1080p at 60°: 1080 / tan(30°) = 1871, in steps of 64
        assert_eq!(face_size(1080, 60f32.to_radians(), 16384), 1920);
        assert_eq!(face_size(720, 90f32.to_radians(), 16384), 768);
        // a narrow view or a 4K screen: never more than 2048, nor the device's limit
        assert_eq!(face_size(2160, 30f32.to_radians(), 16384), MAX_SIZE);
        assert_eq!(face_size(2160, 60f32.to_radians(), 1024), 1024);
        assert_eq!(face_size(10, 2.0, 16384), MIN_SIZE);
    }

    #[test]
    fn a_capture_is_outdated_by_the_settings_the_region_and_the_sky() {
        let a = look();
        assert!(!outdated(&a, &a));
        for b in [
            Look { distance: 300.0, ..a },
            Look { draw_distance: 512.0, ..a },
            Look { epoch: 8, ..a },
            Look { classic: true, ..a },
            Look { size: 1408, ..a },
            Look {
                sun: (a.sun + Vec3::new(0.0, 0.06, 0.0)).normalize(),
                ..a
            },
            Look {
                colours: [a.colours[0] * 1.1, a.colours[1], a.colours[2], a.colours[3], a.colours[4]],
                ..a
            },
            Look { fog_density: 0.003, ..a },
        ] {
            assert!(outdated(&a, &b), "{b:?}");
        }
        // what goes unnoticed: a slight turn of the sun, a few per cent of a
        // colour, a size within a quarter
        for b in [
            Look {
                sun: (a.sun + Vec3::new(0.0, 0.01, 0.0)).normalize(),
                ..a
            },
            Look {
                colours: [a.colours[0] * 1.02, a.colours[1], a.colours[2], a.colours[3], a.colours[4]],
                ..a
            },
            Look { size: 1664, ..a },
        ] {
            assert!(!outdated(&a, &b), "{b:?}");
        }
    }

    #[test]
    fn a_refresh_follows_the_camera_the_clock_the_look_and_the_changes() {
        let at = Instant::now();
        let made = Made {
            centre: DVec3::new(1000.0, 2000.0, 30.0),
            look: look(),
            at,
            stamp: 5,
            changes: 4,
        };
        // 1 % of 256 m
        assert!(!refresh_due(&made, made.centre + DVec3::new(2.5, 0.0, 0.0), &look(), 4, at));
        assert!(refresh_due(&made, made.centre + DVec3::new(2.6, 0.0, 0.0), &look(), 4, at));
        assert!(!refresh_due(&made, made.centre, &look(), 4, at + Duration::from_millis(3900)));
        assert!(refresh_due(&made, made.centre, &look(), 4, at + MAX_AGE));
        assert!(refresh_due(&made, made.centre, &Look { epoch: 8, ..look() }, 4, at));
        // an object it holds moved, went or changed
        assert!(refresh_due(&made, made.centre, &look(), 5, at));
        // shown up to a tenth of the impostor distance away
        assert!(shown_from(&made, made.centre + DVec3::new(0.0, 25.0, 0.0)));
        assert!(!shown_from(&made, made.centre + DVec3::new(0.0, 26.0, 0.0)));
    }

    #[test]
    fn the_panorama_begins_at_the_impostor_distance_even_with_sun_shadows() {
        assert_eq!(start(120.0, false, 0.0), 120.0);
        // shadows reaching past it (quality 3: 192 m, 4: 256 m) do not push
        // it out: the objects in between lose the sun's shadows on them
        assert_eq!(start(120.0, true, 192.0), 120.0);
        assert_eq!(start(120.0, true, 256.0), 120.0);
        assert_eq!(start(64.0, true, 96.0), 64.0);
        assert_eq!(start(256.0, true, 48.0), 256.0);
    }

    #[test]
    fn the_schedule_shows_a_panorama_once_its_six_faces_are_drawn() {
        let mut s = Schedule::default();
        let at = Instant::now();
        let eye = DVec3::new(10.0, 20.0, 30.0);
        let l = look();
        let room = |_| true;
        // the first frame only warms up (the stamps are a frame behind)
        assert_eq!(step(&mut s, true, false, true, eye, &l, (1, 0), at, room), Step::default());
        let mut faces = Vec::new();
        for k in 0..6 {
            let st = step(&mut s, true, false, true, eye, &l, (2 + k, 0), at, room);
            assert_eq!(st.front, None, "nothing shown before the sixth face");
            let (slot, face, made) = st.capture.expect("a face each frame");
            assert_eq!(
                (slot, made.stamp, made.centre),
                (0, 2, eye),
                "one capture, begun on its first frame"
            );
            faces.push(face);
        }
        assert_eq!(faces, [0, 1, 2, 3, 4, 5]);
        let st = step(&mut s, true, false, true, eye, &l, (8, 0), at, room);
        assert_eq!(st.front.map(|(slot, made)| (slot, made.stamp)), Some((0, 2)));
        assert_eq!(st.capture, None, "nothing to refresh");
        // a frame with a drawn pick (a click, the hover) still shows it
        let st = step(&mut s, true, true, true, eye, &l, (9, 0), at, room);
        assert_eq!(st.front.map(|(slot, _)| slot), Some(0));
        assert!(st.pick, "bin 30 is the pick's");
        // the camera moved: the next capture goes into the other slot,
        // the first one stays shown until it is complete
        let moved = eye + DVec3::new(3.0, 0.0, 0.0);
        let st = step(&mut s, true, false, true, moved, &l, (10, 0), at, room);
        assert_eq!(st.capture.map(|(slot, face, _)| (slot, face)), Some((1, 0)));
        assert_eq!(st.front.map(|(slot, _)| slot), Some(0));
        // a probe shadow frame: no face, the capture waits
        let st = step(&mut s, true, false, false, moved, &l, (11, 0), at, room);
        assert_eq!(st.capture, None);
        // another region: the render origin moved, nothing shown, and the
        // capture under way begins again from here
        let st = step(&mut s, true, false, true, moved, &Look { epoch: 9, ..l }, (12, 0), at, room);
        assert_eq!(st.front, None);
        let (slot, face, made) = st.capture.expect("a new capture");
        assert_eq!((slot, face, made.stamp), (0, 0, 12));
        // a teleport: past a tenth of the distance, nothing shown any more
        let mut s = Schedule::default();
        for k in 0..8 {
            step(&mut s, true, false, true, eye, &l, (1 + k, 0), at, room);
        }
        assert!(s.front.is_some());
        let far = eye + DVec3::new(500.0, 0.0, 0.0);
        let st = step(&mut s, true, false, true, far, &l, (9, 0), at, room);
        assert_eq!(st.front, None);
        let (_, face, made) = st.capture.expect("a new capture");
        assert_eq!((face, made.centre, made.stamp), (0, far, 9));
    }

    #[test]
    fn a_change_of_what_the_panorama_holds_starts_a_capture_once() {
        let mut s = Schedule::default();
        let at = Instant::now();
        let eye = DVec3::ZERO;
        let l = look();
        for k in 0..8 {
            step(&mut s, true, false, true, eye, &l, (1 + k, 3), at, |_| true);
        }
        assert!(s.front.is_some() && s.back.is_none());
        assert_eq!(step(&mut s, true, false, true, eye, &l, (9, 3), at, |_| true).capture, None);
        let st = step(&mut s, true, false, true, eye, &l, (10, 4), at, |_| true);
        assert_eq!(st.capture.map(|(_, face, made)| (face, made.changes)), Some((0, 4)));
        // the capture goes on with more changes (the next one takes them)
        let st = step(&mut s, true, false, true, eye, &l, (11, 5), at, |_| true);
        assert_eq!(st.capture.map(|(_, face, made)| (face, made.changes)), Some((1, 4)));
    }

    #[test]
    fn an_entry_written_again_keeps_its_stamp_and_a_held_one_that_changes_counts() {
        // (the one test of the panoramas' sets and of the change count: the
        // tests run at once)
        let old = CullObject {
            sphere: [400.0, 0.0, 0.0, 4.0],
            flags: object_flags::ACTIVE,
            stamp: 3,
            ..CullObject::NONE
        };
        // off: no stamp on either side, the entry as it is written
        let off = CullObject { stamp: 0, ..old };
        let moved_off = CullObject { sphere: [1.0; 4], ..off };
        assert_eq!(restamp(&off, moved_off), moved_off);
        // a new level of detail, a texture: the same entry, its old stamp
        assert_eq!(restamp(&old, CullObject { stamp: 9, ..old }), old);
        // held by the panorama shown (capture point at the origin, from
        // 256 m, stamp 5)
        publish(0, Some(([0.0, 0.0, 0.0, 256.0], 1024.0, 5)));
        publish(1, None);
        let before = changes();
        let moved = CullObject {
            sphere: [410.0, 0.0, 0.0, 4.0],
            stamp: 9,
            ..old
        };
        assert_eq!(restamp(&old, moved), moved);
        assert_eq!(changes(), before + 1);
        // not held: near, changed since the capture, worn, kept near
        for o in [
            CullObject {
                sphere: [100.0, 0.0, 0.0, 4.0],
                ..old
            },
            CullObject { stamp: 6, ..old },
            CullObject { avatar: 2, ..old },
            CullObject {
                flags: old.flags | object_flags::NO_IMPOSTOR,
                ..old
            },
        ] {
            restamp(
                &o,
                CullObject {
                    sphere: [0.0; 4],
                    stamp: 9,
                    ..o
                },
            );
        }
        assert_eq!(changes(), before + 1);
        // gone (the table's empty entry)
        assert_eq!(restamp(&old, CullObject::NONE), CullObject::NONE);
        assert_eq!(changes(), before + 2);
        // another level of detail: other bounds, the same object; held
        // still, with its new sphere, and no capture due
        let detail = [400.3, 0.0, 0.0, 4.5];
        let swapped = CullObject {
            sphere: detail,
            stamp: 9,
            ..old
        };
        assert_eq!(detail_stamp(&old, &swapped), 3);
        let kept = CullObject { stamp: 3, ..swapped };
        assert_eq!(restamp(&old, kept), kept);
        assert_eq!(changes(), before + 2);
        // but moved, changed otherwise, or across where the panorama begins
        // (256.5 m, then 255.6 m away): a change like any other
        for (o, new) in [
            (old, [410.0, 0.0, 0.0, 4.0]),
            (old, [400.0, 0.0, 0.0, 6.0]),
            (
                CullObject {
                    sphere: [260.5, 0.0, 0.0, 4.0],
                    ..old
                },
                [260.5, 0.0, 0.0, 4.9],
            ),
        ] {
            let new = CullObject {
                sphere: new,
                stamp: 9,
                ..o
            };
            assert_eq!(detail_stamp(&o, &new), 9, "{:?}", new.sphere);
        }
        let flagged = CullObject {
            flags: old.flags | object_flags::NO_IMPOSTOR,
            ..swapped
        };
        assert_eq!(detail_stamp(&old, &flagged), 9);
        // off: no stamps
        assert_eq!(detail_stamp(&off, &CullObject { sphere: detail, ..off }), 0);
        publish(0, None);
    }

    #[test]
    fn a_drawn_pick_keeps_the_panorama_shown_and_holds_the_face_captured_back() {
        let mut s = Schedule::default();
        let at = Instant::now();
        let eye = DVec3::ZERO;
        let l = look();
        let room = |_| true;
        // no panorama yet: a pick takes nothing from the first capture
        step(&mut s, true, false, true, eye, &l, (1, 0), at, room);
        let st = step(&mut s, true, true, true, eye, &l, (2, 0), at, room);
        assert!(!st.pick && st.front.is_none());
        assert_eq!(st.capture.map(|(_, face, _)| face), Some(0));
        // (the sixth face goes out at the sixth frame, shown from the next)
        for k in 0..6 {
            step(&mut s, true, false, true, eye, &l, (3 + k, 0), at, room);
        }
        assert!(s.front.is_some());
        // a refresh under way: the frames with a pick show the panorama
        // and capture nothing, the next frame takes the next face
        let moved = eye + DVec3::new(3.0, 0.0, 0.0);
        let st = step(&mut s, true, false, true, moved, &l, (9, 0), at, room);
        assert_eq!(st.capture.map(|(slot, face, _)| (slot, face)), Some((1, 0)));
        for k in 0..3 {
            let st = step(&mut s, true, true, true, moved, &l, (10 + k, 0), at, room);
            assert_eq!(st.front.map(|(slot, _)| slot), Some(0), "shown on a pick frame");
            assert!(st.pick);
            assert_eq!(st.capture, None, "bin 30 is the pick's");
        }
        let st = step(&mut s, true, false, true, moved, &l, (13, 0), at, room);
        assert_eq!(st.capture.map(|(slot, face, _)| (slot, face)), Some((1, 1)));
        assert!(!st.pick);
    }

    #[test]
    fn a_pick_waits_for_a_probe_shadow_map_only_while_a_panorama_may_be_shown() {
        let at = Instant::now();
        let made = Made {
            centre: DVec3::ZERO,
            look: look(),
            at,
            stamp: 2,
            changes: 0,
        };
        let mut p = ScenePanorama::default();
        assert!(!p.pick_waits(true), "nothing shown: the main bins hold everything");
        p.schedule.back = Some(Back { slot: 0, made, faces: 5 });
        assert!(!p.pick_waits(true), "a capture under way is not shown yet");
        p.schedule.back = Some(Back { slot: 0, made, faces: 6 });
        assert!(p.pick_waits(true), "complete: shown at the next step");
        p.schedule.back = None;
        p.schedule.front = Some((0, made));
        assert!(p.pick_waits(true));
        assert!(!p.pick_waits(false), "bin 30 free");
    }

    #[test]
    fn nothing_starts_without_room_nor_off_the_world_view() {
        let mut s = Schedule::default();
        let at = Instant::now();
        let eye = DVec3::ZERO;
        let l = look();
        step(&mut s, true, false, true, eye, &l, (1, 0), at, |_| false);
        assert_eq!(step(&mut s, true, false, true, eye, &l, (2, 0), at, |_| false), Step::default());
        assert_eq!(step(&mut s, false, false, true, eye, &l, (3, 0), at, |_| true), Step::default());
    }

    /// A frame with the camera at the origin looking along +x, nothing but
    /// the main view drawn.
    fn frame(objects: u32) -> CullFrame {
        let view = glam::camera::rh::view::look_at_mat4(Vec3::ZERO, Vec3::X, Vec3::Z);
        let proj = glam::camera::rh::proj::directx::perspective_infinite_reverse(1.0, 1.5, 0.1);
        let g = GpuCullView {
            first_person: None,
            planes: view_planes(proj * view),
            eye: Vec3::ZERO,
            draw_distance: 1024.0,
            shadow_distance: 64.0,
            reflection_distance: 32.0,
            pixel_scale: 1000.0 / 0.5,
            shadows: false,
            reflections: false,
        };
        let all = [crate::shadow_cull::CASTER_PLANES_ALL; 3];
        let mut f = CullFrame::new(
            &g,
            &[Mat4::IDENTITY; 3],
            &all,
            0.0,
            [Mat4::IDENTITY; 3],
            [None; crate::gpu_cull::SPOT_SHADOWS],
            None,
            0,
        );
        f.sizes[1] = objects;
        f
    }

    /// Objects at these places, radius 4 m, stamp 3; one opaque face each.
    fn scene(at: &[Vec3]) -> (Vec<CullObject>, Vec<CullFace>) {
        let objects = at
            .iter()
            .map(|c| CullObject {
                sphere: [c.x, c.y, c.z, 4.0],
                flags: object_flags::ACTIVE,
                stamp: 3,
                ..CullObject::NONE
            })
            .collect();
        let faces = (0..at.len())
            .map(|i| CullFace {
                object: i as u32,
                bits: pass::OPAQUE | crate::gpu_cull::FACE_FIRST,
                index_count: 36,
                ..CullFace::NONE
            })
            .collect();
        (objects, faces)
    }

    fn bins(f: &CullFrame, objects: &[CullObject], faces: &[CullFace], i: usize) -> u32 {
        face_bins(f, faces, objects, &world_spheres(objects, &[], &[], &[]), i).0
    }

    #[test]
    fn the_panorama_shown_takes_the_far_static_objects_out_of_the_main_view() {
        // near, far, far and changed since the capture, far but kept
        // (blended faces, the selection), an avatar's attachment
        let at = [Vec3::new(100.0, 0.0, 0.0), Vec3::new(400.0, 0.0, 0.0)];
        let (mut objects, faces) = scene(&[at[0], at[1], at[1], at[1], at[1]]);
        objects[2].stamp = 9;
        objects[3].flags |= object_flags::NO_IMPOSTOR;
        objects[4].flags |= object_flags::NO_PROBE;
        let off = frame(objects.len() as u32);
        for i in 0..objects.len() {
            assert_eq!(bins(&off, &objects, &faces, i), 1, "without a panorama everything is drawn ({i})");
        }
        let mut on = off;
        on.panorama[0] = [0.0, 0.0, 0.0, 256.0];
        on.panorama[2][0] = 1024.0;
        on.panorama_stamps[0] = 5;
        let drawn: Vec<u32> = (0..objects.len()).map(|i| bins(&on, &objects, &faces, i)).collect();
        assert_eq!(drawn, [1, 0, 1, 1, 1]);
        // the shell is measured from the capture point, not the camera
        on.panorama[0] = [300.0, 0.0, 0.0, 256.0];
        assert_eq!(bins(&on, &objects, &faces, 1), 1);
        assert!(face_bins(&on, &faces, &objects, &world_spheres(&objects, &[], &[], &[]), 1).1);
    }

    #[test]
    fn the_face_captured_takes_what_its_panorama_will_stand_in_for() {
        // in front (+x), behind, too small for either view, beyond the
        // camera's draw distance but within the capture's, beyond both
        let (mut objects, faces) = scene(&[
            Vec3::new(400.0, 0.0, 0.0),
            Vec3::new(-400.0, 0.0, 0.0),
            Vec3::new(500.0, 30.0, 0.0),
            Vec3::new(1100.0, 0.0, 0.0),
            Vec3::new(1500.0, 0.0, 0.0),
        ]);
        objects[2].sphere[3] = 0.05;
        let mut f = frame(objects.len() as u32);
        let centre = Vec3::new(200.0, 0.0, 0.0);
        f.panorama[1] = centre.extend(256.0).to_array();
        f.panorama[2][1] = 1024.0;
        f.panorama_planes = view_planes(face_proj() * crate::probes::face_view(centre, 0)).map(|v| v.to_array());
        f.panorama_stamps[1] = 5;
        let pano = 1 << BIN_PROBE_SHADOW;
        let drawn: Vec<u32> = (0..objects.len()).map(|i| bins(&f, &objects, &faces, i)).collect();
        // (the first is 200 m from the capture point: not far enough)
        assert_eq!(drawn, [1, 0, 0, pano, 0]);
        f.panorama[1][0] = 0.0;
        f.panorama_planes = view_planes(face_proj() * crate::probes::face_view(Vec3::ZERO, 0)).map(|v| v.to_array());
        let drawn: Vec<u32> = (0..objects.len()).map(|i| bins(&f, &objects, &faces, i)).collect();
        assert_eq!(drawn, [1 | pano, 0, 0, 0, 0]);
        // a probe shadow map this frame has the bin
        f.sizes[3] |= frame_flags::PROBE_SHADOW;
        assert_eq!(bins(&f, &objects, &faces, 0) & pano, 0);
        f.sizes[3] &= !frame_flags::PROBE_SHADOW;
        // changed since the capture began, an avatar, a blended face
        objects[0].stamp = 5;
        assert_eq!(bins(&f, &objects, &faces, 0), 1);
        objects[0].stamp = 3;
        objects[0].flags |= object_flags::RIGGED;
        assert_eq!(bins(&f, &objects, &faces, 0) & pano, 0);
        objects[0].flags = object_flags::ACTIVE;
        let mut blended = faces.clone();
        blended[0].bits = pass::BLEND | crate::gpu_cull::FACE_FIRST;
        assert_eq!(
            face_bins(&f, &blended, &objects, &world_spheres(&objects, &[], &[], &[]), 0).0 & pano,
            0
        );
    }

    #[test]
    fn the_drawn_pick_finds_the_far_objects_of_the_panorama_shown() {
        // held under the pixel, held beside it, held but too small for the
        // main view, near, far but changed since the capture, far but kept
        let (mut objects, faces) = scene(&[
            Vec3::new(400.0, 0.0, 0.0),
            Vec3::new(400.0, 120.0, 0.0),
            Vec3::new(500.0, 0.0, 0.0),
            Vec3::new(100.0, 0.0, 0.0),
            Vec3::new(400.0, 0.0, 0.0),
            Vec3::new(400.0, 0.0, 0.0),
        ]);
        objects[2].sphere[3] = 0.05;
        objects[4].stamp = 9;
        objects[5].flags |= object_flags::NO_IMPOSTOR;
        let mut f = frame(objects.len() as u32);
        f.panorama[0] = [0.0, 0.0, 0.0, 256.0];
        f.panorama[2][0] = 1024.0;
        f.panorama_stamps[0] = 5;
        // the pick's narrowed view: along +x, a hundredth of a radian wide
        let view = glam::camera::rh::view::look_at_mat4(Vec3::ZERO, Vec3::X, Vec3::Z);
        let pick = glam::camera::rh::proj::directx::perspective_infinite_reverse(0.01, 1.0, 0.1) * view;
        f.panorama_planes = view_planes(pick).map(|v| v.to_array());
        let pano = 1 << BIN_PROBE_SHADOW;
        // without the pick's flag the planes are a capture's: nothing
        let drawn: Vec<u32> = (0..objects.len()).map(|i| bins(&f, &objects, &faces, i)).collect();
        assert_eq!(drawn, [0, 0, 0, 1, 1, 1]);
        f.panorama[2][2] = 1.0;
        let drawn: Vec<u32> = (0..objects.len()).map(|i| bins(&f, &objects, &faces, i)).collect();
        // the held one under the pixel in bin 30, still out of the main view
        assert_eq!(drawn, [pano, 0, 0, 1, 1, 1]);
        // the whole frame's state, as the renderer writes it
        let (far_front, in_bin) = face_panorama(&f, &objects[0], objects.len() as u32, Vec3::new(400.0, 0.0, 0.0), 4.0);
        assert!(far_front && in_bin);
        // a probe shadow map has bin 30 (the pick waits for the next frame)
        f.sizes[3] |= frame_flags::PROBE_SHADOW;
        assert_eq!(bins(&f, &objects, &faces, 0), 0);
        f.sizes[3] &= !frame_flags::PROBE_SHADOW;
        // past the camera's draw distance the main view would not draw it
        f.eye[3] = 380.0;
        assert_eq!(bins(&f, &objects, &faces, 0), 0);
        f.eye[3] = 1024.0;
        // a blended face (never held: kept near by the viewer) stays out
        let mut blended = faces.clone();
        blended[0].bits = pass::BLEND | crate::gpu_cull::FACE_FIRST;
        assert_eq!(
            face_bins(&f, &blended, &objects, &world_spheres(&objects, &[], &[], &[]), 0).0 & pano,
            0
        );
    }

    #[test]
    fn the_shader_is_valid() {
        let src = include_str!("shaders/scene_panorama.wgsl");
        let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));
        let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty());
        if let Err(e) = v.validate(&module) {
            panic!("{}", e.emit_to_string(src));
        }
        // the uniforms the shader reads are the ones written here
        assert_eq!(size_of::<PanoU>(), 160);
        assert!(
            src.contains(
                "struct Pano {\n    vp: mat4x4<f32>,\n    inv_vp: mat4x4<f32>,\n    centre: vec4<f32>,\n    screen: vec4<f32>,\n}"
            )
        );
    }

    /// CPU mirror of scene_panorama.wgsl `look_depth` (`EDGE` included).
    fn look_depth(z: [f32; 4], bilinear: [f32; 4]) -> f32 {
        const EDGE: f32 = 1.05;
        let dot = |a: [f32; 4], b: [f32; 4]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let nearest = z.iter().copied().fold(f32::MIN, f32::max);
        let farthest = z.iter().copied().fold(f32::MAX, f32::min);
        if farthest > 0.0 && nearest <= farthest * EDGE {
            return dot(bilinear, z);
        }
        let w: [f32; 4] = std::array::from_fn(|i| if z[i] > 0.0 { bilinear[i] } else { 0.0 });
        let cover: f32 = w.iter().sum();
        if cover <= 0.0 {
            return 0.0;
        }
        let nearer = |zi: f32| {
            if zi > 0.0 {
                (0..4).filter(|&j| z[j] >= zi).map(|j| w[j]).sum()
            } else {
                0.0
            }
        };
        let m = z.iter().copied().filter(|&zi| nearer(zi) >= 0.5 * cover).fold(0.0, f32::max);
        let ws: [f32; 4] = std::array::from_fn(|i| {
            if z[i] > 0.0 && z[i].max(m) <= z[i].min(m) * EDGE {
                w[i]
            } else {
                0.0
            }
        });
        dot(ws, z) / ws.iter().sum::<f32>().max(1e-6)
    }

    fn bilinear(wx: f32, wy: f32) -> [f32; 4] {
        [(1.0 - wx) * (1.0 - wy), wx * (1.0 - wy), (1.0 - wx) * wy, wx * wy]
    }

    #[track_caller]
    fn close(got: f32, want: f32, why: &str) {
        assert!((got - want).abs() < 1e-7, "{got} instead of {want}: {why}");
    }

    #[test]
    fn a_far_edge_lies_where_the_weights_cross_one_half() {
        // one surface (a slanted wall within EDGE): interpolated, as before
        let wall = [0.0100, 0.0102, 0.0101, 0.0103];
        let w = bilinear(0.3, 0.6);
        close(
            look_depth(wall, w),
            wall.iter().zip(w).map(|(z, w)| z * w).sum::<f32>(),
            "one surface",
        );
        // a near object in the left column, a far one in the right: the
        // depth switches where the near side's weight crosses one half,
        // not as soon as one near texel is in the 2 x 2
        let (near, far) = (0.02, 0.005);
        let edge = [near, far, near, far];
        close(look_depth(edge, bilinear(0.45, 0.5)), near, "the near side holds 55 %");
        close(look_depth(edge, bilinear(0.55, 0.5)), far, "the near side holds 45 %");
        close(look_depth(edge, bilinear(0.5, 0.5)), near, "a tie goes to the near side");
        // a corner: the near texel alone holds under half
        close(look_depth([near, far, far, far], bilinear(0.2, 0.2)), near, "the corner holds 64 %");
        close(look_depth([near, far, far, far], bilinear(0.6, 0.6)), far, "the corner holds 16 %");
        // the near side's own texels are interpolated among themselves
        let pair = [0.0200, far, 0.0202, far];
        let got = look_depth(pair, bilinear(0.2, 0.5));
        assert!((got - 0.0201).abs() < 1e-6, "{got}");
    }

    #[test]
    fn the_sky_only_lowers_the_coverage() {
        // an object against the sky keeps its own depth (whether the pixel
        // is drawn is the coverage's call, in fs_depth and fs_colour)
        let z = [0.02, 0.0, 0.02, 0.0];
        close(look_depth(z, bilinear(0.9, 0.5)), 0.02, "mostly sky");
        close(look_depth([0.0; 4], bilinear(0.5, 0.5)), 0.0, "all sky");
        // in front of a far object with the sky beside: the half rule
        // over the covered texels only
        let z = [0.02, 0.005, 0.0, 0.0];
        close(look_depth(z, bilinear(0.4, 0.5)), 0.02, "60 % of the covered weight");
        close(look_depth(z, bilinear(0.6, 0.5)), 0.005, "40 % of the covered weight");
    }

    #[test]
    fn the_shader_has_the_mirrored_edge_rule() {
        let src = include_str!("shaders/scene_panorama.wgsl");
        assert!(src.contains("const EDGE: f32 = 1.05;"));
        assert!(src.contains("fn look_depth(z: vec4<f32>, bilinear: vec4<f32>) -> f32 {"));
        assert!(src.contains("let d = look_depth(z, bilinear);"));
        assert!(src.contains("let half = 0.5 * cover;"));
        assert!(src.contains("let same = (z > vec4<f32>(0.0)) & (max(z, vec4<f32>(m)) <= min(z, vec4<f32>(m)) * EDGE);"));
    }

    #[test]
    fn the_culling_shader_has_the_panorama() {
        let src = include_str!("shaders/cull.wgsl");
        assert!(src.contains(&format!("const OBJ_NO_IMPOSTOR: u32 = {}u;", object_flags::NO_IMPOSTOR)));
        assert!(src.contains("panorama: array<vec4<f32>, 3>,"));
        assert!(src.contains("panorama_planes: array<vec4<f32>, 6>,"));
        assert!(src.contains("panorama_stamps: vec4<u32>,"));
        assert!(src.contains("fn in_panorama(p: vec4<f32>, reach: f32, stamp: u32, object_stamp: u32, c: vec3<f32>, r: f32) -> bool {"));
        // the drawn pick's part of bin 30 (face_panorama)
        assert!(src.contains("let picked = far_front"));
        assert!(src.contains("&& frame.panorama[2].z > 0.0"));
        assert!(src.contains("&& d - r <= frame.eye.w"));
        assert!(src.contains("let pano = pano_face || picked;"));
    }

    #[test]
    fn the_pick_shaders_draw_bin_30_with_the_sides_of_the_main_view() {
        // bin 30 mixes one- and two-sided faces: the pick pipelines of the
        // panorama's faces cull nothing and leave the back of a one-sided
        // face out themselves, as the main view's back-face culling does
        let src = include_str!("shaders/object.wgsl");
        assert!(src.contains("override PICK_HELD: bool = false;"));
        let back = "if (PICK_HELD && !front && (rec.flags.x & FLAG_DOUBLE_SIDED) == 0u) {";
        assert_eq!(src.matches(back).count(), 2);
        assert!(src.contains("if (PICK_HELD && (rec.flags.x & FLAG_ALPHA_MASK) == 0u) {"));
    }
}
