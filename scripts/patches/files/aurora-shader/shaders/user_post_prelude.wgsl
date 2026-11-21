// aurora-local-patch: aurora-shader-host-prelude
// aurora-shader contract v1: the host's prelude, put before every effect.
// Frozen. A change of a binding, of `Frame` or of a helper's meaning is
// contract v2: a new prelude, effects saying `api = 2`.
//
// Image: gamma-encoded colour (0..1) after tone mapping and the scene's
// MSAA/TAA resolve, before SMAA/FXAA (they smooth the chain's output),
// the photo look, the selection outline and the interface.
// Depth: reversed-Z with an infinite far plane. Raw depth 1 at the near
// plane, 0 at infinity: the sky (nothing drawn) is d == 0. The distance
// along the view axis is `frame.near / d` metres, exact under TAA jitter,
// photo tiles and VR eyes (they only move x and y).
// Pixels: @builtin(position) and textureLoad are y down, uv (0,0) is the top
// left corner; NDC y is up. View space: x right, y up, the camera looks
// down -z, normals point towards the camera (+z).

struct Frame {
    // this image's projection, unjittered (tile- and eye-specific)
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    // world to view; its rotation is exact, its translation may be
    // relative to the eye (AURORA_EYE_RELATIVE)
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    // w, h, 1/w, 1/h of this image (the world's size at the render scale)
    size: vec4<f32>,
    // xy: under TAA, where a colour pixel's depth and normal are, in pixels
    // (y down); zw: the same in uv. 0 without TAA
    jitter: vec4<f32>,
    // the part of the whole picture this image is: x, y, w, h (0, 0, 1, 1
    // except for the tiles of a large photo)
    tile: vec4<f32>,
    // seconds (the scene's clock)
    time: f32,
    // seconds since this effect chain's previous frame, 0..0.25
    delta: f32,
    frame: u32,
    // FLAG_*
    flags: u32,
    // near plane distance, metres
    near: f32,
    // the viewer's draw distance, metres
    draw_distance: f32,
    // image pixels per viewfinder pixel (1 live, more for a large photo)
    pixel_scale: f32,
    // 1
    api: u32,
}

// The normals come from the meshes (else rebuilt from the depth).
const FLAG_GEOMETRY_NORMALS: u32 = 1u;
const FLAG_TAA: u32 = 2u;
const FLAG_PHOTO: u32 = 4u;
const FLAG_VR_EYE: u32 = 8u;
// `state` was just zeroed (first frame of this effect version).
const FLAG_STATE_RESET: u32 = 16u;

// What fs_main receives: the pixel centre (y down) and its uv.
struct FsIn {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// The previous step's image (the frame for the first effect); filterable.
@group(0) @binding(0) var color: texture_2d<f32>;
// The scene depth, same size as `color`. textureLoad only (raw_depth).
@group(0) @binding(1) var depth: texture_depth_2d;
// View-space normal in xyz; w = 1 from the meshes, 0.5 rebuilt from the
// depth, 0 sky. Only filled when the effect uses it (view_normal).
@group(0) @binding(2) var normals: texture_2d<f32>;
@group(0) @binding(3) var linear_sampler: sampler;
@group(0) @binding(4) var point_sampler: sampler;
@group(0) @binding(5) var<uniform> frame: Frame;
// 4 KiB of this effect's own, kept from frame to frame and zeroed when the
// effect is (re)loaded: written by `prepare`, read by `fs_main`.
@group(0) @binding(6) var<storage, read_write> state: array<vec4<f32>, 256>;

fn image_size() -> vec2<i32> {
    return vec2<i32>(frame.size.xy);
}

// The pixel of a fragment (y down).
fn pixel_of(in: FsIn) -> vec2<i32> {
    return vec2<i32>(in.pos.xy);
}

// Under TAA the depth and normals are drawn half a pixel off the colour:
// the pixel holding a colour pixel's depth (use it for edges).
fn depth_pixel(px: vec2<i32>) -> vec2<i32> {
    return vec2<i32>(floor(vec2<f32>(px) + 0.5 + frame.jitter.xy));
}

// Raw depth at a pixel, clamped to the image. 0 = sky.
fn raw_depth(px: vec2<i32>) -> f32 {
    return textureLoad(depth, clamp(px, vec2<i32>(0), image_size() - vec2<i32>(1)), 0);
}

fn is_sky(d: f32) -> bool {
    return d <= 0.0;
}

// Distance along the view axis, metres (a very large number for the sky).
fn view_z(d: f32) -> f32 {
    return frame.near / max(d, 1e-20);
}

// View-space position of uv at raw depth d (not for the sky: check is_sky
// first). Its length is the true distance from the eye.
fn view_pos(uv: vec2<f32>, d: f32) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let p = frame.inv_proj * vec4<f32>(ndc, d, 1.0);
    return p.xyz / p.w;
}

// The normal at a pixel (see `normals`), clamped to the image.
fn view_normal(px: vec2<i32>) -> vec4<f32> {
    return textureLoad(normals, clamp(px, vec2<i32>(0), image_size() - vec2<i32>(1)), 0);
}

// uv in the whole picture (differs from uv only in a photo's tile).
fn whole_uv(uv: vec2<f32>) -> vec2<f32> {
    return frame.tile.xy + uv * frame.tile.zw;
}

// The whole picture's width in this image's pixels: sizes given as a % of
// the picture's width stay the same share of it at any render scale,
// photo size or tile.
fn ref_width() -> f32 {
    return frame.size.x / frame.tile.z;
}

fn pct_to_px(pct: f32) -> f32 {
    return pct * 0.01 * ref_width();
}

// The view axis in whole-picture uv: the image centre except for an
// off-centre projection (a VR eye, a photo's tile).
fn optical_centre() -> vec2<f32> {
    let p = frame.proj[2].xy;
    return whole_uv(vec2<f32>((1.0 - p.x) * 0.5, (1.0 + p.y) * 0.5));
}

// Rec. 709 luma of a colour.
fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}
// ---- end of the aurora-shader v1 prelude; the effect follows
