// aurora-local-patch: mirror-pass-shader
// The mirror pass's occlusion test (mirror_pass.rs), after the Hi-Z pyramid
// of the main view's phase 1 depth (occlusion.rs): the depth of what was
// drawn last frame, drawn again from this frame's camera, so surely there.
// When even the nearest point of the volume where mirror_tap gives the
// mirror image (the probe's box in front of mirror_tap's cut) lies behind
// the farthest depth over that volume's rect, no surface on the screen can
// show the image: the pass's draws get no instance (its CPU-built lists,
// its GPU bin's count, its sky), in this same frame. The sky's draw is the
// verdict read back. One workgroup: the rect test is a few texel loads, the
// draws are cleared by its 64 lanes.

struct Test {
    // the volume's rect on the depth (uv, y down: lo.xy, hi.xy) through the
    // main view's camera, jittered as the depth
    rect: vec4<f32>,
    // the depth's size in pixels
    screen: vec2<f32>,
    // the volume's nearest depth (reverse-Z: the largest)
    nearest: f32,
    // the GPU bin's word in `counts`, or NO_BIN without the GPU culling
    bin: u32,
    // the pass's CPU-built lists: (first command, count) pairs, five of them
    ranges: array<vec4<u32>, 3>,
}

@group(0) @binding(0) var<uniform> test: Test;
@group(0) @binding(1) var hiz: texture_2d<f32>;
// the frame's CPU-built draws (DrawIndexedIndirect, five words each)
@group(0) @binding(2) var<storage, read_write> draws: array<u32>;
// the GPU bins' draw counts (gpu_cull.rs), or a stand-in
@group(0) @binding(3) var<storage, read_write> counts: array<u32>;
// the sky's draw: vertex count, instance count, first vertex, first instance
@group(0) @binding(4) var<storage, read_write> sky: array<u32, 4>;

const NO_BIN: u32 = 0xffffffffu;

// hiz_sphere_visible's rule (hiz_test.wgsl) for a rect: one pixel of margin
// (TAA jitter, rasterization rules), the level whose texels make the rect
// span 2 x 2 at most, the farthest depth of those texels; hidden when even
// the nearest point is behind it.
fn hidden() -> bool {
    let screen = test.screen;
    let pmin = clamp(test.rect.xy * screen - vec2<f32>(1.0), vec2<f32>(0.0), screen - vec2<f32>(1.0));
    let pmax = clamp(test.rect.zw * screen + vec2<f32>(1.0), vec2<f32>(0.0), screen - vec2<f32>(1.0));
    let size = max(pmax.x - pmin.x, pmax.y - pmin.y);
    var level = 0u;
    if (size > 2.0) {
        level = u32(ceil(log2(size))) - 1u;
    }
    level = min(level, textureNumLevels(hiz) - 1u);
    let texel = f32(1u << (level + 1u));
    let dims = vec2<i32>(textureDimensions(hiz, i32(level))) - vec2<i32>(1);
    let t0 = min(vec2<i32>(floor(pmin / texel)), dims);
    let t1 = min(vec2<i32>(floor(pmax / texel)), dims);
    var farthest = 1.0;
    for (var y = t0.y; y <= t1.y; y++) {
        for (var x = t0.x; x <= t1.x; x++) {
            farthest = min(farthest, textureLoad(hiz, vec2<i32>(x, y), i32(level)).r);
        }
    }
    return test.nearest < farthest;
}

@compute @workgroup_size(64)
fn cs_mirror_test(@builtin(local_invocation_index) lane: u32) {
    let gone = hidden();
    if (lane == 0u) {
        sky[0] = 3u;
        sky[1] = select(1u, 0u, gone);
        sky[2] = 0u;
        sky[3] = 0u;
        if (gone && test.bin != NO_BIN && test.bin < arrayLength(&counts)) {
            counts[test.bin] = 0u;
        }
    }
    if (!gone) {
        return;
    }
    let commands = arrayLength(&draws) / 5u;
    for (var k = 0u; k < 5u; k++) {
        let pair = test.ranges[k / 2u];
        let odd = (k & 1u) != 0u;
        let first = select(pair.x, pair.z, odd);
        let count = select(pair.y, pair.w, odd);
        for (var i = lane; i < count; i += 64u) {
            let c = first + i;
            if (c < commands) {
                draws[c * 5u + 1u] = 0u;
            }
        }
    }
}
