// aurora-local-patch: aurora-shader-host-prelude-v2
// aurora-shader contract v2: what it adds to the v1 prelude. Before an
// effect whose header says `api = 2` the host puts the v1 prelude,
// unchanged, a newline, then this; an `api = 1` effect gets the v1 prelude
// alone, as before. Frozen with v2: a change is contract v3.
//
// All of v1 holds, but `frame.api` is 2 and `frame.flags` may hold
// FLAG_ALPHA_DEPTH.

// `alpha_depth` tells this frame's blended faces apart from the scene.
const FLAG_ALPHA_DEPTH: u32 = 32u;

// The nearest blended face from 88 % opaque, or from half opaque where it
// lies on a surface (a layer on a skin, worn alpha; a fainter pane of glass
// away from any surface is not in it; no particles, no water), raw
// reversed-Z like `depth`, same size
// and jitter, the scene's depth where there is none. Ordinary blended faces
// write no scene depth, so they have no line of their own in `depth`: this
// is the only place they show. Yes or no, not how much. With
// FLAG_ALPHA_DEPTH only (the viewer's ambient occlusion on, or FSR without
// upscaling); without it, it is `depth` itself: nothing is in front.
@group(0) @binding(7) var alpha_depth: texture_depth_2d;

// Raw depth of the nearest blended face at a pixel, clamped to the image;
// 0 (nothing, like the sky) without FLAG_ALPHA_DEPTH.
fn raw_alpha_depth(px: vec2<i32>) -> f32 {
    if ((frame.flags & FLAG_ALPHA_DEPTH) == 0u) {
        return 0.0;
    }
    return textureLoad(alpha_depth, clamp(px, vec2<i32>(0), image_size() - vec2<i32>(1)), 0);
}

// 1 when a blended face at a pixel is nearer than the view distance `z`
// (metres, `view_z`) by more than `tolerance` of it (0.03: 3 %), else 0. A
// face that also writes the scene depth (rigged hair) is at its own `z`:
// within the tolerance, so not in front of itself.
fn alpha_in_front(px: vec2<i32>, z: f32, tolerance: f32) -> f32 {
    let a = raw_alpha_depth(px);
    return select(0.0, 1.0, !is_sky(a) && view_z(a) < z * (1.0 - tolerance));
}
// ---- end of the aurora-shader v2 prelude; the effect follows
