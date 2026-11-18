// aurora-local-patch: region-crossing-origin-shader
// The render origin moved by whole regions (origin_shift.rs): the records
// placed in the world that the frame did not write move by `delta`, their
// translation only (model column 3, the record's fourth vector).
//
// data[0..2]: delta x, y (f32 bits); data[2]: record slots; data[3]: the
// record's stride in vec4s; then one bit per slot, set for those to move.

@group(0) @binding(0) var<storage, read_write> records: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> data: array<u32>;

const HEADER: u32 = 4u;
// workgroups per row of the dispatch
const ROW: u32 = 65535u;

@compute @workgroup_size(64)
fn shift(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * ROW * 64u;
    if (i >= data[2]) {
        return;
    }
    if ((data[HEADER + (i >> 5u)] & (1u << (i & 31u))) == 0u) {
        return;
    }
    let k = i * data[3] + 3u;
    if (k >= arrayLength(&records)) {
        return;
    }
    let t = records[k];
    records[k] = vec4<f32>(t.x + bitcast<f32>(data[0]), t.y + bitcast<f32>(data[1]), t.z, t.w);
}
