// aurora-local-patch: aurora-shader-host-shaders
// aurora-shader host passes: the full-screen triangle every effect is drawn
// with, the effects' depth with the water in it, the normals the effects
// read (`normals` of the prelude), the copy into an sRGB frame, and the AA
// finish (FXAA on the chain's output, or its temporal resolve).

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// One triangle over the whole image; uv (0,0) top left.
@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> VsOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var out: VsOut;
    out.pos = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    out.uv = uv;
    return out;
}

// The prelude's Frame, same layout (only some fields are read here).
struct Frame {
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    size: vec4<f32>,
    jitter: vec4<f32>,
    tile: vec4<f32>,
    time: f32,
    delta: f32,
    frame: u32,
    flags: u32,
    near: f32,
    draw_distance: f32,
    pixel_scale: f32,
    api: u32,
}

const FLAG_GEOMETRY_NORMALS: u32 = 1u;

// the effects' depth (the scene's, or fs_depth's with the water in it)
@group(0) @binding(0) var depth: texture_depth_2d;
// GTAO's mesh normals (gtao.rs NORMAL_FORMAT: xy the octahedral world
// normal), or a 1 x 1 stand-in without them
@group(0) @binding(1) var mesh_normals: texture_2d<f32>;
@group(0) @binding(2) var<uniform> frame: Frame;
// The scene depth (renderer.rs depth_ss) without the water: for fs_depth,
// and for fs_normals to tell the water pixels (the same view as `depth`
// when the frame has no water). Bindings 7-9: apart from the others of
// this one module.
@group(0) @binding(7) var scene_depth: texture_depth_2d;

// ---- the effects' depth: the scene depth with the nearest visible water
// surface in it, as reversed-Z. The water pass writes no depth (it reads
// depth_ss for what is under it); its interface (env.wgsl
// water_interface_depth) is 1 / (1 + view distance), 0 where there is none,
// and the mask is 0 where a water-exclusion face (a hull) is nearer.
@group(0) @binding(8) var water_interface: texture_depth_2d;
@group(0) @binding(9) var water_mask: texture_2d<f32>;

@fragment
fn fs_depth(@builtin(position) pos: vec4<f32>) -> @builtin(frag_depth) f32 {
    let px = vec2<i32>(pos.xy);
    let d = textureLoad(scene_depth, px, 0);
    // the water images are the world's size, smaller than the chain's under
    // an FSR upscale: sampled by size ratio, as the mesh normals are
    let full = vec2<i32>(textureDimensions(water_interface));
    let wp = min(vec2<i32>(vec2<f32>(px) * vec2<f32>(full) * frame.size.zw), full - vec2<i32>(1));
    let w = textureLoad(water_interface, wp, 0);
    if (w <= 0.0 || textureLoad(water_mask, wp, 0).r < 0.5) {
        return d;
    }
    let dist = 1.0 / w - 1.0;
    // reversed-Z infinite: d = near / distance; 1 inside the near plane
    let water = min(frame.near / max(dist, 1e-6), 1.0);
    // the nearer of the two: land in front of the water keeps its depth
    return max(d, water);
}

// The octahedral normal of the G-buffer (gtao.wgsl oct_decode).
fn oct_decode(e_in: vec2<f32>) -> vec3<f32> {
    let e = e_in * 2.0 - vec2<f32>(1.0);
    var n = vec3<f32>(e, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) {
        let s = select(vec2<f32>(-1.0), vec2<f32>(1.0), n.xy >= vec2<f32>(0.0));
        n = vec3<f32>((vec2<f32>(1.0) - abs(n.yx)) * s, n.z);
    }
    return normalize(n);
}

// View-space position of pixel px (w = 1), or 0 for the sky.
fn position_at(px: vec2<i32>) -> vec4<f32> {
    let size = vec2<i32>(frame.size.xy);
    let p = clamp(px, vec2<i32>(0), size - vec2<i32>(1));
    let d = textureLoad(depth, p, 0);
    if (d <= 0.0) {
        return vec4<f32>(0.0);
    }
    let uv = (vec2<f32>(p) + 0.5) * frame.size.zw;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let v = frame.inv_proj * vec4<f32>(ndc, d, 1.0);
    return vec4<f32>(v.xyz / v.w, 1.0);
}

// XeGTAO_CalculateEdges (gtao.wgsl calculate_edges) on view distances.
fn calculate_edges(c: f32, l: f32, r: f32, t: f32, b: f32) -> vec4<f32> {
    var e = vec4<f32>(l, r, t, b) - vec4<f32>(c);
    let slope_lr = (e.y - e.x) * 0.5;
    let slope_tb = (e.w - e.z) * 0.5;
    let adjusted = e + vec4<f32>(slope_lr, -slope_lr, slope_tb, -slope_tb);
    e = min(abs(e), abs(adjusted));
    return saturate(vec4<f32>(1.25) - e / (c * 0.011));
}

// A neighbour on the sky continues the slope of the opposite one, so it
// neither tilts the normal nor gives a zero-length direction.
fn mend(n: vec4<f32>, opposite: vec4<f32>, c: vec3<f32>) -> vec3<f32> {
    return select(2.0 * c - opposite.xyz, n.xyz, n.w > 0.0);
}

@fragment
fn fs_normals(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<i32>(pos.xy);
    let centre = position_at(px);
    if (centre.w == 0.0) {
        return vec4<f32>(0.0, 0.0, 1.0, 0.0);
    }
    // on a water pixel the mesh normal is the seabed's (or nothing on the
    // open ocean): the surface's is rebuilt from the depth below
    let water = textureLoad(scene_depth, px, 0) != textureLoad(depth, px, 0);
    // sampled by size ratio, as gtao.wgsl does, should the sizes differ
    let full = vec2<i32>(textureDimensions(mesh_normals));
    let src = min(vec2<i32>(vec2<f32>(px) * vec2<f32>(full) * frame.size.zw), full - vec2<i32>(1));
    let mesh = textureLoad(mesh_normals, src, 0);
    // the prepass leaves its target cleared (all 0) where it drew nothing
    // though the depth has a surface: the scene impostors' panorama, whose
    // depth goes in after it. A cleared texel would decode as a normal
    // pointing straight down: the normal is rebuilt from the depth there.
    let drawn = any(mesh != vec4<f32>(0.0));
    if (!water && drawn && (frame.flags & FLAG_GEOMETRY_NORMALS) != 0u) {
        let world = oct_decode(mesh.xy);
        let rot = mat3x3<f32>(frame.view[0].xyz, frame.view[1].xyz, frame.view[2].xyz);
        return vec4<f32>(normalize(rot * world), 1.0);
    }
    // XeGTAO_CalculateNormal (gtao.wgsl calculate_normal): the four
    // neighbours, each pair counted as much as both lie on the surface
    let l0 = position_at(px + vec2<i32>(-1, 0));
    let r0 = position_at(px + vec2<i32>(1, 0));
    let t0 = position_at(px + vec2<i32>(0, -1));
    let b0 = position_at(px + vec2<i32>(0, 1));
    if ((l0.w == 0.0 && r0.w == 0.0) || (t0.w == 0.0 && b0.w == 0.0)) {
        return vec4<f32>(0.0, 0.0, 1.0, 0.5);
    }
    let c = centre.xyz;
    let l = mend(l0, r0, c);
    let r = mend(r0, l0, c);
    let t = mend(t0, b0, c);
    let b = mend(b0, t0, c);
    let edges = calculate_edges(-c.z, -l.z, -r.z, -t.z, -b.z);
    let accepted = saturate(
        vec4<f32>(edges.x * edges.z, edges.z * edges.y, edges.y * edges.w, edges.w * edges.x) + vec4<f32>(0.01)
    );
    let dl = normalize(l - c);
    let dr = normalize(r - c);
    let dt = normalize(t - c);
    let db = normalize(b - c);
    var n = accepted.x * cross(dl, dt) + accepted.y * cross(dt, dr) + accepted.z * cross(dr, db)
        + accepted.w * cross(db, dl);
    n = normalize(n);
    if (dot(n, -c) < 0.0) {
        n = -n;
    }
    return vec4<f32>(n, 0.5);
}

// sRGB frames only: the last effect's gamma-encoded image, decoded, so the
// sRGB target encodes it back to the same bytes.
@group(0) @binding(3) var blit_src: texture_2d<f32>;

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

@fragment
fn fs_blit(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let c = textureLoad(blit_src, vec2<i32>(pos.xy), 0);
    return vec4<f32>(srgb_to_linear(c.rgb), c.a);
}

// ---- the AA finish: the viewer's spatial anti-aliasing on the chain's
// output, for the AA settings whose own pass cannot run after the chain.
// FXAA mode: the post pass skips its FXAA while the chain draws, this pass
// does it on the chain's result instead (whole image). TAA and MSAA: the
// scene is resolved already, so the pass is masked to what the chain itself
// made sharp (ink, fringes), and leaves the resolved scene alone.
// Bindings apart from the blit's and the normals' (one module).
@group(0) @binding(4) var finish_src: texture_2d<f32>;
@group(0) @binding(5) var finish_input: texture_2d<f32>;
@group(0) @binding(6) var finish_sampler: sampler;

// Range of (output - input) over 3 x 3, in 8-bit gamma units, under which
// the chain changed the image smoothly (haze, grading: no FXAA) and over
// which it drew an edge (FXAA in full).
const MASK_LO: f32 = 0.02;
const MASK_HI: f32 = 0.06;

// post.wgsl luma() reads sqrt(linear), about the gamma-encoded value: here
// the image is gamma-encoded already.
fn fxaa_luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

fn fxaa_tap(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(finish_src, finish_sampler, uv, 0.0).rgb;
}

// Port of post.wgsl fxaa() (Lottes, "FXAA 2" PC variant), same thresholds
// and taps, on the gamma-encoded image.
fn fxaa(uv: vec2<f32>, texel: vec2<f32>, rgb_m: vec3<f32>) -> vec3<f32> {
    let l_nw = fxaa_luma(fxaa_tap(uv + vec2<f32>(-1.0, -1.0) * texel));
    let l_ne = fxaa_luma(fxaa_tap(uv + vec2<f32>(1.0, -1.0) * texel));
    let l_sw = fxaa_luma(fxaa_tap(uv + vec2<f32>(-1.0, 1.0) * texel));
    let l_se = fxaa_luma(fxaa_tap(uv + vec2<f32>(1.0, 1.0) * texel));
    let l_m = fxaa_luma(rgb_m);
    let l_min = min(l_m, min(min(l_nw, l_ne), min(l_sw, l_se)));
    let l_max = max(l_m, max(max(l_nw, l_ne), max(l_sw, l_se)));
    if (l_max - l_min < max(0.0312, l_max * 0.125)) {
        return rgb_m;
    }
    var dir = vec2<f32>(-((l_nw + l_ne) - (l_sw + l_se)), (l_nw + l_sw) - (l_ne + l_se));
    let reduce = max((l_nw + l_ne + l_sw + l_se) * (0.25 / 8.0), 1.0 / 128.0);
    let rcp_min = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
    dir = clamp(dir * rcp_min, vec2<f32>(-8.0), vec2<f32>(8.0)) * texel;
    let rgb_a = 0.5 * (fxaa_tap(uv + dir * (1.0 / 3.0 - 0.5)) + fxaa_tap(uv + dir * (2.0 / 3.0 - 0.5)));
    let rgb_b = rgb_a * 0.5 + 0.25 * (fxaa_tap(uv + dir * -0.5) + fxaa_tap(uv + dir * 0.5));
    let l_b = fxaa_luma(rgb_b);
    if (l_b < l_min || l_b > l_max) {
        return rgb_a;
    }
    return rgb_b;
}

// How much the chain drew an edge around px: the range of its change
// (output - input) over 3 x 3, its largest channel (the fringes of a
// chromatic aberration move the channels apart).
fn chain_edge(px: vec2<i32>) -> f32 {
    let last = vec2<i32>(textureDimensions(finish_src)) - vec2<i32>(1);
    var lo = vec3<f32>(1.0);
    var hi = vec3<f32>(-1.0);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let p = clamp(px + vec2<i32>(x, y), vec2<i32>(0), last);
            let d = textureLoad(finish_src, p, 0).rgb - textureLoad(finish_input, p, 0).rgb;
            lo = min(lo, d);
            hi = max(hi, d);
        }
    }
    let r = hi - lo;
    return smoothstep(MASK_LO, MASK_HI, max(r.x, max(r.y, r.z)));
}

// The finished pixel, gamma-encoded; alpha (a cut-out photo's) is the
// centre's, as SMAA keeps it.
fn finish(pos: vec4<f32>, masked: bool) -> vec4<f32> {
    let px = vec2<i32>(pos.xy);
    let c = textureLoad(finish_src, px, 0);
    var mask = 1.0;
    if (masked) {
        mask = chain_edge(px);
        if (mask <= 0.0) {
            return c;
        }
    }
    let texel = 1.0 / vec2<f32>(textureDimensions(finish_src));
    let soft = fxaa(pos.xy * texel, texel, c.rgb);
    return vec4<f32>(mix(c.rgb, soft, mask), c.a);
}

@fragment
fn fs_fxaa(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return finish(pos, false);
}

@fragment
fn fs_fxaa_masked(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return finish(pos, true);
}

// sRGB frames: decoded, so the target encodes it back (replaces the blit).
@fragment
fn fs_fxaa_srgb(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let c = finish(pos, false);
    return vec4<f32>(srgb_to_linear(c.rgb), c.a);
}

@fragment
fn fs_fxaa_masked_srgb(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let c = finish(pos, true);
    return vec4<f32>(srgb_to_linear(c.rgb), c.a);
}

// ---- the temporal finish (ChainAa::Temporal): what the chain changed
// (its output minus its input), resolved over the frames like the scene's
// TAA (taa.wgsl): the last frame's result reprojected through each pixel's
// motion, else the camera through the depth, clamped to the current 3 x 3
// neighbourhood, blended with the current change. Under TAA and FSR the
// effects read a jittered depth after the scene's history resolved its own
// jitter, so their lines move by up to half a pixel each frame; averaged,
// they sit still at their true place, as sharp as the scene's edges.
// Bindings apart from the others of this one module.
struct Resolve {
    // this frame's unjittered clip space (raw depth z) to the last one's
    reproject: mat4x4<f32>,
    // xy: where a pixel's depth and motion are, in pixels (the TAA jitter)
    jitter: vec4<f32>,
    // x: the current frame's weight; y: 1 to start over; z: 1 when the
    // motion vectors are bound
    params: vec4<f32>,
}

// the last frame's resolved change (signed)
@group(0) @binding(10) var resolve_history: texture_2d<f32>;
// the depth prepass's target: zw the screen motion since the last frame,
// in uv and without the jitter, 0 where it drew nothing (object.wgsl
// depth_normal); the world's size; a 1 x 1 stand-in without it
@group(0) @binding(11) var resolve_motion: texture_2d<f32>;
// the effects' depth, at the chain's size
@group(0) @binding(12) var resolve_depth: texture_depth_2d;
@group(0) @binding(13) var<uniform> resolve: Resolve;

struct Resolved {
    @location(0) colour: vec4<f32>,
    @location(1) history: vec4<f32>,
}

fn resolved(pos: vec4<f32>) -> Resolved {
    let px = vec2<i32>(pos.xy);
    let size = vec2<i32>(textureDimensions(finish_src));
    let last = size - vec2<i32>(1);
    let c = textureLoad(finish_src, px, 0);
    let input = textureLoad(finish_input, px, 0).rgb;
    let change = c.rgb - input;
    // the range of the change around the pixel: the history may move
    // within it (a line half a pixel off) but no further (what moved away,
    // or appeared)
    var lo = change;
    var hi = change;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let p = clamp(px + vec2<i32>(x, y), vec2<i32>(0), last);
            let d = textureLoad(finish_src, p, 0).rgb - textureLoad(finish_input, p, 0).rgb;
            lo = min(lo, d);
            hi = max(hi, d);
        }
    }
    var o: Resolved;
    o.colour = c;
    o.history = vec4<f32>(change, 1.0);
    if (resolve.params.y > 0.5) {
        return o;
    }
    let uv = (vec2<f32>(px) + 0.5) / vec2<f32>(size);
    let dp = clamp(vec2<i32>(floor(vec2<f32>(px) + 0.5 + resolve.jitter.xy)), vec2<i32>(0), last);
    // reverse-Z: 0 is infinitely far
    let d = textureLoad(resolve_depth, dp, 0);
    var moved = vec2<f32>(0.0);
    if (resolve.params.z > 0.5 && d > 0.0) {
        // the motion is the world's size (smaller under an FSR upscale)
        let full = vec2<i32>(textureDimensions(resolve_motion));
        let src = min(vec2<i32>(vec2<f32>(dp) * vec2<f32>(full) / vec2<f32>(size)), full - vec2<i32>(1));
        moved = textureLoad(resolve_motion, src, 0).zw;
    }
    var puv: vec2<f32>;
    if (any(moved != vec2<f32>(0.0))) {
        // what the prepass drew moved by its own motion, the camera's included
        puv = uv - moved;
    } else {
        // the sky, what the prepass did not draw, what did not move: the
        // camera alone, through the depth
        let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
        let p = resolve.reproject * vec4<f32>(ndc, max(d, 1e-7), 1.0);
        if (p.w <= 0.0) {
            return o;
        }
        let pn = p.xy / p.w;
        puv = vec2<f32>(pn.x * 0.5 + 0.5, 0.5 - pn.y * 0.5);
    }
    if (any(puv < vec2<f32>(0.0)) || any(puv > vec2<f32>(1.0))) {
        return o;
    }
    let h = clamp(textureSampleLevel(resolve_history, finish_sampler, puv, 0.0).rgb, lo, hi);
    let blended = mix(h, change, resolve.params.x);
    o.history = vec4<f32>(blended, 1.0);
    o.colour = vec4<f32>(clamp(input + blended, vec3<f32>(0.0), vec3<f32>(1.0)), c.a);
    return o;
}

@fragment
fn fs_resolve(@builtin(position) pos: vec4<f32>) -> Resolved {
    return resolved(pos);
}

// sRGB frames: decoded, so the target encodes it back (replaces the blit).
@fragment
fn fs_resolve_srgb(@builtin(position) pos: vec4<f32>) -> Resolved {
    var o = resolved(pos);
    o.colour = vec4<f32>(srgb_to_linear(o.colour.rgb), o.colour.a);
    return o;
}
