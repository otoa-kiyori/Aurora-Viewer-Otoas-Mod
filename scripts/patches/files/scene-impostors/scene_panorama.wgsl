// aurora-local-patch: scene-impostors-shader
// Scene impostors (scene_panorama.rs): the far-field panorama drawn into the
// main view with its depth. A pixel's ray from the camera meets the
// panorama's surface where a fixed point puts it: the direction from the
// capture point gives a depth, the depth a point, the point (brought onto
// the ray) a new direction. The camera stays within a tenth of the impostor
// distance of the capture point and the panorama holds nothing nearer than
// that distance, so each step divides the error by ten at the very least
// (by a hundred until a refresh is due).

struct Pano {
    vp: mat4x4<f32>,
    inv_vp: mat4x4<f32>,
    centre: vec4<f32>,
    screen: vec4<f32>,
}

// vp: the main view's projection by its rotation alone (camera-relative
// points to clip space, as the scene's vertices are drawn), and inv_vp its
// inverse; centre: xyz the capture point relative to the camera, w the
// captures' near plane; screen: x, y the scene's size in pixels, z the
// faces' size, w the coverage a pixel needs to be drawn
@group(0) @binding(0) var<uniform> pano: Pano;
// the six faces in probes::face_view's order (Vulkan's cube map faces)
@group(0) @binding(1) var colour_faces: texture_2d_array<f32>;
@group(0) @binding(2) var depth_faces: texture_depth_2d_array;

// fixed-point steps after the first guess
const STEPS: i32 = 3;
// fs_depth's depth against fs_colour's (reversed depth: smaller is farther)
const DEPTH_MARGIN: f32 = 1.0 - 1.0 / 65536.0;
// across one surface the four depths around a direction differ by less than
// this ratio (a wall seen 87° off its normal from 300 m, 1920 px faces);
// beyond it they straddle an edge
const EDGE: f32 = 1.05;

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}

// The panorama in a direction from the capture point.
struct Look {
    // distance to its surface (0: nothing there)
    dist: f32,
    // coverage of the 2 x 2 texels around the direction
    cover: f32,
    face: i32,
    // the four texels and their weights, covered ones only
    lo: vec2<i32>,
    hi: vec2<i32>,
    weights: vec4<f32>,
}

// The face a unit direction falls in and where on it (texel coordinates of
// its centre, minus a half: the texel to the lower left of the direction
// and the weights of the bilinear filter), and the largest component.
fn face_of(u: vec3<f32>) -> vec4<f32> {
    let a = abs(u);
    var face = 0.0;
    var st = vec2<f32>(0.0);
    var major = 1.0;
    if (a.x >= a.y && a.x >= a.z) {
        major = a.x;
        if (u.x > 0.0) {
            st = vec2<f32>(-u.z, -u.y);
        } else {
            face = 1.0;
            st = vec2<f32>(u.z, -u.y);
        }
    } else if (a.y >= a.z) {
        major = a.y;
        if (u.y > 0.0) {
            face = 2.0;
            st = vec2<f32>(u.x, u.z);
        } else {
            face = 3.0;
            st = vec2<f32>(u.x, -u.z);
        }
    } else {
        major = a.z;
        if (u.z > 0.0) {
            face = 4.0;
            st = vec2<f32>(u.x, -u.y);
        } else {
            face = 5.0;
            st = vec2<f32>(-u.x, -u.y);
        }
    }
    let texel = (st / max(major, 1e-6) * 0.5 + 0.5) * pano.screen.z - 0.5;
    return vec4<f32>(texel, face, major);
}

fn look(u: vec3<f32>) -> Look {
    let f = face_of(u);
    let last = i32(pano.screen.z) - 1;
    let base = floor(f.xy);
    let w = f.xy - base;
    let i = vec2<i32>(base);
    var l: Look;
    l.face = i32(f.z);
    l.lo = clamp(i, vec2<i32>(0), vec2<i32>(last));
    l.hi = clamp(i + vec2<i32>(1), vec2<i32>(0), vec2<i32>(last));
    let z = vec4<f32>(
        textureLoad(depth_faces, l.lo, l.face, 0),
        textureLoad(depth_faces, vec2<i32>(l.hi.x, l.lo.y), l.face, 0),
        textureLoad(depth_faces, vec2<i32>(l.lo.x, l.hi.y), l.face, 0),
        textureLoad(depth_faces, l.hi, l.face, 0),
    );
    let bilinear = vec4<f32>((1.0 - w.x) * (1.0 - w.y), w.x * (1.0 - w.y), (1.0 - w.x) * w.y, w.x * w.y);
    l.weights = bilinear * select(vec4<f32>(0.0), vec4<f32>(1.0), z > vec4<f32>(0.0));
    l.cover = l.weights.x + l.weights.y + l.weights.z + l.weights.w;
    let d = look_depth(z, bilinear);
    l.dist = select(0.0, pano.centre.w / (d * f.w), d > 0.0);
    return l;
}

// The weight of the covered texels at least as near as zi (reverse Z: the
// largest depth is the nearest), when zi itself is covered; else 0.
fn nearer_weight(z: vec4<f32>, w: vec4<f32>, zi: f32) -> f32 {
    return select(0.0, dot(w, select(vec4<f32>(0.0), vec4<f32>(1.0), z >= vec4<f32>(zi))), zi > 0.0);
}

// The panorama's depth of a direction from the 2 x 2 texels around it and
// their bilinear weights (0: nothing there). Across one surface,
// interpolated: the reverse depth of a plane is linear across a face, exact
// for it. Across an edge, the surface holding at least half of the covered
// texels' weight, interpolated over its own texels: the edge lies where the
// weights cross one half and slides with the direction, as the filtered
// colour's edge does (fs_colour), where taking the nearest texel moved it in
// steps of a texel and widened the near side by up to one (about 1.6 px of
// a 4K view). Only the covered texels count: the sky in the 2 x 2 lowers
// the coverage, which decides whether the pixel is drawn at all, and both
// fs_depth and fs_colour trace through this same rule.
// scene_panorama.rs tests::look_depth mirrors it.
fn look_depth(z: vec4<f32>, bilinear: vec4<f32>) -> f32 {
    let nearest = max(max(z.x, z.y), max(z.z, z.w));
    let farthest = min(min(z.x, z.y), min(z.z, z.w));
    if (farthest > 0.0 && nearest <= farthest * EDGE) {
        return dot(bilinear, z);
    }
    let w = bilinear * select(vec4<f32>(0.0), vec4<f32>(1.0), z > vec4<f32>(0.0));
    let cover = w.x + w.y + w.z + w.w;
    if (cover <= 0.0) {
        return 0.0;
    }
    // the median surface, nearest first: the nearest texel that, with what
    // is nearer, holds at least half of the covered weight
    let half = 0.5 * cover;
    var m = 0.0;
    m = select(m, max(m, z.x), nearer_weight(z, w, z.x) >= half);
    m = select(m, max(m, z.y), nearer_weight(z, w, z.y) >= half);
    m = select(m, max(m, z.z), nearer_weight(z, w, z.z) >= half);
    m = select(m, max(m, z.w), nearer_weight(z, w, z.w) >= half);
    // its surface: the covered texels within EDGE of it
    let same = (z > vec4<f32>(0.0)) & (max(z, vec4<f32>(m)) <= min(z, vec4<f32>(m)) * EDGE);
    let ws = w * select(vec4<f32>(0.0), vec4<f32>(1.0), same);
    return dot(ws, z) / max(ws.x + ws.y + ws.z + ws.w, 1e-6);
}

// Unit direction of the pixel's ray, from the camera.
fn pixel_ray(frag: vec2<f32>) -> vec3<f32> {
    let ndc = vec2<f32>(frag.x / pano.screen.x * 2.0 - 1.0, 1.0 - frag.y / pano.screen.y * 2.0);
    let p = pano.inv_vp * vec4<f32>(ndc, 1.0, 1.0);
    return normalize(p.xyz / p.w);
}

struct Hit {
    // along the ray (<= 0: nothing)
    t: f32,
    l: Look,
}

fn march(d: vec3<f32>) -> Hit {
    let c = pano.centre.xyz;
    // first guess: the direction seen from the capture point
    var u = d;
    var l = look(u);
    var h: Hit;
    h.t = -1.0;
    h.l = l;
    if (l.dist <= 0.0) {
        return h;
    }
    for (var k = 0; k < STEPS; k++) {
        let t = dot(c + u * l.dist, d);
        let v = normalize(d * t - c);
        let next = look(v);
        if (next.dist <= 0.0) {
            break;
        }
        u = v;
        l = next;
    }
    h.t = dot(c + u * l.dist, d);
    h.l = l;
    return h;
}

// The colour of the covered texels around the direction, filtered.
fn colour_of(l: Look) -> vec3<f32> {
    let c = textureLoad(colour_faces, l.lo, l.face, 0).rgb * l.weights.x
        + textureLoad(colour_faces, vec2<i32>(l.hi.x, l.lo.y), l.face, 0).rgb * l.weights.y
        + textureLoad(colour_faces, vec2<i32>(l.lo.x, l.hi.y), l.face, 0).rgb * l.weights.z
        + textureLoad(colour_faces, l.hi, l.face, 0).rgb * l.weights.w;
    return c / max(l.cover, 1e-4);
}

fn depth_at(d: vec3<f32>, t: f32) -> f32 {
    let clip = pano.vp * vec4<f32>(d * t, 1.0);
    return clamp(clip.z / clip.w, 0.0, 1.0);
}

// The depth alone, into the single-sample depth after the prepass, for the
// passes that read it before the scene pass (scene_panorama::encode_depth).
// A hair farther than fs_colour's (1/65536 of it: 2 mm at 150 m): without
// MSAA the scene pass starts from this depth, and its composite, the same
// trace compiled into another pipeline, must still pass GreaterEqual.
@fragment
fn fs_depth(@builtin(position) pos: vec4<f32>) -> @builtin(frag_depth) f32 {
    let d = pixel_ray(pos.xy);
    let h = march(d);
    if (h.t <= 0.0 || h.l.cover < 0.5) {
        discard;
    }
    return depth_at(d, h.t) * DEPTH_MARGIN;
}

struct Out {
    @builtin(frag_depth) depth: f32,
    // alpha: the coverage (alpha to coverage under MSAA; never written)
    @location(0) colour: vec4<f32>,
}

@fragment
fn fs_colour(@builtin(position) pos: vec4<f32>) -> Out {
    let d = pixel_ray(pos.xy);
    let h = march(d);
    if (h.t <= 0.0 || h.l.cover < pano.screen.w) {
        discard;
    }
    var o: Out;
    o.depth = depth_at(d, h.t);
    o.colour = vec4<f32>(colour_of(h.l), clamp(h.l.cover, 0.0, 1.0));
    return o;
}
