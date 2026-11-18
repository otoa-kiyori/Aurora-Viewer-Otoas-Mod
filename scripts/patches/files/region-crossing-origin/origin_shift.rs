// aurora-local-patch: region-crossing-origin-module
//! A region crossing moves the render origin by whole regions: the records
//! follow on the GPU, nothing is uploaded again.
//!
//! The scene's coordinates start at the main region's corner, as the Legacy
//! Viewer's agent frame (LLWorld::updateAgentOffset, then
//! LLPipeline::shiftObjects moving every drawable). At a crossing every
//! object placed in the world is `delta` away from where its record has it,
//! the same `delta` for all (a whole number of regions on x and y). Writing
//! every record again sent 85 to 140 MB in one frame on a ride across regions, past the
//! journal staging's 32 MB (write_staging.rs), and stalled 50 to 226 ms.
//!
//! Here the scene moves what rests in the world as a placement would have
//! moved it (aurora-viewer scene/origin_shift.rs), without placing it again:
//! `RecordStore::shift_world` moves those records' CPU mirror at once, and
//! one compute pass (origin_shift.wgsl), submitted after the frame's writes,
//! moves the GPU's copies of those the frame did not write itself (those
//! already carry the moved mirror). What is anchored to an avatar's joints
//! or skinned (placed from a palette) does not move with the origin: it is
//! left alone and returned to the caller. A moved record that is placed
//! again anyway, where its new model is its moved one up to the rounding of
//! the other origin (`same_place`), is not marked for upload.
//!
//! Precision: the same coordinates as before, so the same f32 precision. A
//! moved translation is rounded once more, to half a unit in the last place
//! of the largest value it held, about 0.03 mm at 500 m: it does not grow
//! with more crossings (adding whole regions back and forth lands on the
//! coarser grid once), and a record placed again is exact.
//!
//! `AURORA_ORIGIN_SHIFT=0` writes every record again, as before (A/B).

use crate::types::{DrawRecord, flags};
use std::ops::Range;

/// Whether the origin moves are shifted on the GPU (`AURORA_ORIGIN_SHIFT`,
/// on unless `0`).
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AURORA_ORIGIN_SHIFT").ok().is_none_or(|v| v.trim() != "0"))
}

/// The record is placed in the world by its model: not anchored to a joint,
/// not skinned (both placed from a palette the scene writes each frame).
pub fn world_placed(r: &DrawRecord) -> bool {
    r.flags[0] & (flags::ANCHORED | flags::SKINNED) == 0
}

/// The model moved with the origin (what the GPU pass does: one f32 add).
pub fn shift_model(model: &mut [[f32; 4]; 4], delta: [f32; 2]) {
    model[3][0] += delta[0];
    model[3][1] += delta[1];
}

/// How far two translations may be and still be the same place: rounding
/// only, 16 units in the last place of the larger plus the move's `reach`
/// (a moved value was rounded to the grid of what it held before the move:
/// a prim brought near the origin from far keeps that coarser rounding).
/// 1.9 mm at 1 km.
fn rounding(a: f32, b: f32, reach: f32) -> f32 {
    f32::EPSILON * 16.0 * (a.abs().max(b.abs()) + reach.abs()).max(1.0)
}

/// `new` (the scene's model in the new origin) is `old` (the record's model
/// moved by `delta`) up to rounding: the same rotation, scale and height
/// exactly, x and y within [`rounding`].
pub fn same_place(old: &[[f32; 4]; 4], new: &[[f32; 4]; 4], delta: [f32; 2]) -> bool {
    old[..3] == new[..3]
        && old[3][2..] == new[3][2..]
        && (0..2).all(|k| (old[3][k] - new[3][k]).abs() <= rounding(old[3][k], new[3][k], delta[k]))
}

/// A set of record ids.
#[derive(Default, Debug, Clone)]
struct Bits(Vec<u32>);

impl Bits {
    fn insert(&mut self, i: usize) {
        let w = i / 32;
        if self.0.len() <= w {
            self.0.resize(w + 1, 0);
        }
        self.0[w] |= 1 << (i % 32);
    }

    fn contains(&self, i: usize) -> bool {
        self.0.get(i / 32).is_some_and(|w| w & (1 << (i % 32)) != 0)
    }

    fn remove(&mut self, i: usize) {
        if let Some(w) = self.0.get_mut(i / 32) {
            *w &= !(1 << (i % 32));
        }
    }

    fn remove_range(&mut self, r: Range<usize>) {
        for i in r {
            self.remove(i);
        }
    }

    fn clear(&mut self) {
        self.0.clear();
    }
}

/// The origin move waiting for the next flush.
#[derive(Default, Debug)]
pub struct Pending {
    delta: [f32; 2],
    /// Records whose mirror moved: the GPU pass moves them too.
    shifted: Bits,
    /// Moved records not placed again since: the next placement within
    /// rounding of the moved model is not uploaded.
    settling: Bits,
    count: usize,
    active: bool,
}

/// What the GPU pass moves: `delta`, and the mask of the record ids to move
/// (bit `i % 32` of word `i / 32`) below `slots`.
#[derive(Debug, PartialEq)]
pub struct Pass {
    pub delta: [f32; 2],
    pub slots: u32,
    pub mask: Vec<u32>,
}

impl Pending {
    /// An origin move waits for the GPU pass.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// A move of the origin by `delta` begins; false when one is already
    /// waiting (the caller then writes them all again).
    pub fn begin(&mut self, delta: [f32; 2]) -> bool {
        if self.active {
            return false;
        }
        self.delta = delta;
        self.active = true;
        self.count = 0;
        true
    }

    /// Record `id`'s mirror moved.
    pub fn add(&mut self, id: u32) {
        self.shifted.insert(id as usize);
        self.settling.insert(id as usize);
        self.count += 1;
    }

    /// The records moved, as a mask (bit `i % 32` of word `i / 32`), all
    /// at once (`shift_blocks`).
    pub fn add_mask(&mut self, mask: Vec<u32>) {
        self.count += mask.iter().map(|w| w.count_ones() as usize).sum::<usize>();
        self.settling = Bits(mask.clone());
        self.shifted = Bits(mask);
    }

    /// Whether record `id` moved with the origin under way.
    pub fn moved(&self, id: u32) -> bool {
        self.active && self.shifted.contains(id as usize)
    }

    /// Records moved by this origin move.
    pub fn count(&self) -> usize {
        self.count
    }

    /// The ids moved, for the fallback that writes them again.
    pub fn take_ids(&mut self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.count);
        for (w, bits) in self.shifted.0.iter().enumerate() {
            let mut b = *bits;
            while b != 0 {
                let k = b.trailing_zeros();
                out.push(w as u32 * 32 + k);
                b &= b - 1;
            }
        }
        self.reset();
        out
    }

    /// Record `id` is placed at `new` while its mirror holds `old`: true
    /// when that is its moved place up to rounding (not uploaded). Once per
    /// record and move: the next placement compares as usual.
    pub fn settles(&mut self, id: u32, old: &[[f32; 4]; 4], new: &[[f32; 4]; 4]) -> bool {
        if !self.active || !self.settling.contains(id as usize) {
            return false;
        }
        self.settling.remove(id as usize);
        same_place(old, new, self.delta)
    }

    /// The GPU pass of the flush: the records moved below `slots`, but those
    /// the flush writes itself (`written`: they carry the moved mirror).
    /// None when nothing is left to move. Ends the move.
    pub fn finish(&mut self, written: &[Range<usize>], slots: usize) -> Option<Pass> {
        if !self.active {
            return None;
        }
        let mut mask = std::mem::take(&mut self.shifted);
        for r in written {
            mask.remove_range(r.start..r.end.min(slots));
        }
        let words = slots.div_ceil(32);
        mask.0.resize(words, 0);
        if let Some(last) = mask.0.last_mut()
            && !slots.is_multiple_of(32)
        {
            *last &= (1u32 << (slots % 32)) - 1;
        }
        let delta = self.delta;
        self.reset();
        mask.0.iter().any(|w| *w != 0).then_some(Pass {
            delta,
            slots: slots as u32,
            mask: mask.0,
        })
    }

    fn reset(&mut self) {
        self.shifted.clear();
        self.settling.clear();
        self.count = 0;
        self.active = false;
    }
}

/// The ids of `ids` below `slots`, as a mask.
pub fn mask_of(ids: impl IntoIterator<Item = u32>, slots: usize) -> Vec<u32> {
    let mut mask = vec![0u32; slots.div_ceil(32)];
    for id in ids {
        if let Some(w) = mask.get_mut(id as usize / 32)
            && (id as usize) < slots
        {
            *w |= 1 << (id % 32);
        }
    }
    mask
}

/// Records per thread at least, below which one thread moves them all.
const PER_THREAD: usize = 32_768;

/// Move the mirror's records whose bit is set in `mask` (blocks of `block`
/// records each, a multiple of 32): those placed in the world move by
/// `delta`, the others lose their bit and are returned. A few threads share
/// the blocks: the mirror of a busy scene is over 100 MB, and the walk is
/// bound by memory (3 ms instead of 10 on 384,000 records).
pub fn shift_blocks(blocks: &mut [Vec<DrawRecord>], mask: &mut [u32], block: usize, delta: [f32; 2]) -> Vec<u32> {
    debug_assert!(block.is_multiple_of(32));
    let words = block / 32;
    let total: usize = blocks.iter().map(Vec::len).sum();
    let threads = (total / PER_THREAD).clamp(1, std::thread::available_parallelism().map_or(1, |n| n.get() / 2).clamp(1, 6));
    let work = |first: usize, blocks: &mut [Vec<DrawRecord>], mask: &mut [u32]| {
        let mut refused = Vec::new();
        for (b, recs) in blocks.iter_mut().enumerate() {
            let Some(bits) = mask.get_mut(b * words..((b + 1) * words).min(mask.len())) else {
                continue;
            };
            for (k, r) in recs.iter_mut().enumerate() {
                let (w, bit) = (k / 32, 1u32 << (k % 32));
                if bits.get(w).is_some_and(|v| v & bit != 0) {
                    if world_placed(r) {
                        shift_model(&mut r.model, delta);
                    } else {
                        bits[w] &= !bit;
                        refused.push(((first + b) * block + k) as u32);
                    }
                }
            }
        }
        refused
    };
    if threads == 1 {
        return work(0, blocks, mask);
    }
    let per = blocks.len().div_ceil(threads).max(1);
    let mut mask_rest = mask;
    std::thread::scope(|scope| {
        let mut parts = Vec::new();
        for (n, chunk) in blocks.chunks_mut(per).enumerate() {
            let take = (chunk.len() * words).min(mask_rest.len());
            let (mine, rest) = std::mem::take(&mut mask_rest).split_at_mut(take);
            mask_rest = rest;
            parts.push(scope.spawn(move || work(n * per, chunk, mine)));
        }
        parts.into_iter().flat_map(|h| h.join().unwrap_or_default()).collect()
    })
}

/// Words before the mask in the pass's buffer: delta x and y (f32 bits),
/// the slot count, the records' stride in vec4s.
const HEADER: usize = 4;
/// Workgroup size of origin_shift.wgsl.
const GROUP: u32 = 64;
/// Workgroups per row of the dispatch (the shader's `ROW`).
const ROW: u32 = 65535;

/// The dispatch covering `slots` records: (x, y) workgroups.
fn dispatch(slots: u32) -> (u32, u32) {
    let groups = slots.div_ceil(GROUP);
    if groups == 0 {
        return (0, 0);
    }
    (groups.min(ROW), groups.div_ceil(ROW))
}

/// The pass's buffer contents.
fn words(pass: &Pass) -> Vec<u32> {
    let stride = (std::mem::size_of::<DrawRecord>() / 16) as u32;
    let mut w = Vec::with_capacity(HEADER + pass.mask.len());
    w.extend_from_slice(&[pass.delta[0].to_bits(), pass.delta[1].to_bits(), pass.slots, stride]);
    w.extend_from_slice(&pass.mask);
    w
}

/// The compute pass that moves the records on the GPU.
pub struct ShiftPass {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    data: Option<wgpu::Buffer>,
    /// Prepared by the flush, encoded after the frame's other writes.
    ready: Option<(wgpu::BindGroup, (u32, u32))>,
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl ShiftPass {
    pub fn new(device: &wgpu::Device) -> Self {
        use crate::pipeline_cache::CachedPipelines;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("origin shift"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/origin_shift.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("origin shift"),
            entries: &[storage_entry(0, false), storage_entry(1, true)],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("origin shift"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline_cached(&wgpu::ComputePipelineDescriptor {
            label: Some("origin shift"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("shift"),
            compilation_options: Default::default(),
            cache: None,
        });
        ShiftPass {
            pipeline,
            layout,
            data: None,
            ready: None,
        }
    }

    /// Write the pass's data in the frame's journal (after the records'
    /// writes) and bind it to `records`.
    pub fn prepare(&mut self, device: &wgpu::Device, writes: &mut crate::writes::GpuWrites, records: &wgpu::Buffer, pass: &Pass) {
        let data = words(pass);
        let bytes = (data.len() * 4) as u64;
        if self.data.as_ref().is_none_or(|b| b.size() < bytes) {
            self.data = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("origin shift"),
                size: bytes.next_power_of_two().max(4096),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let Some(buffer) = &self.data else {
            return;
        };
        writes.write_buffer(buffer, 0, bytemuck::cast_slice(&data));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("origin shift"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: records.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer.as_entire_binding(),
                },
            ],
        });
        self.ready = Some((bind_group, dispatch(pass.slots)));
    }

    /// The prepared pass, to submit after the frame's writes (main_thread.rs).
    pub fn take(&mut self, device: &wgpu::Device) -> Option<wgpu::CommandEncoder> {
        let (bind_group, (x, y)) = self.ready.take()?;
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("origin shift"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("origin shift"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        Some(enc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4, Quat, Vec3};

    fn model(pos: Vec3, rot: Quat, scale: Vec3) -> [[f32; 4]; 4] {
        Mat4::from_scale_rotation_translation(scale, rot, pos).to_cols_array_2d()
    }

    /// A prim of region `region` (in regions from the origin's), at
    /// `local` in it, placed as the scene does: the region's offset plus
    /// its position, a child through its root's matrix.
    fn placed(region: (i32, i32), local: Vec3, child: Option<Vec3>) -> [[f32; 4]; 4] {
        let offset = Vec3::new(region.0 as f32 * 256.0, region.1 as f32 * 256.0, 0.0);
        let rot = Quat::from_rotation_z(0.7) * Quat::from_rotation_x(0.2);
        let root = Mat4::from_scale_rotation_translation(Vec3::ONE, rot, offset + local);
        match child {
            None => (root * Mat4::from_scale(Vec3::new(3.0, 0.5, 2.0))).to_cols_array_2d(),
            Some(c) => (root * Mat4::from_scale_rotation_translation(Vec3::splat(0.3), Quat::from_rotation_y(0.4), c)).to_cols_array_2d(),
        }
    }

    #[test]
    fn a_moved_record_is_where_the_new_origin_places_it() {
        // the origin moves one region east, then two north and back: every
        // prim around, roots and children, is placed by the scene in the
        // new origin where its moved record already is, up to rounding
        let moves = [(1, 0), (0, 2), (-1, -2), (3, -1)];
        let mut worst: f32 = 0.0;
        for k in 0..400u32 {
            let f = |s: u32| ((k.wrapping_mul(2654435761).wrapping_add(s * 40503)) % 25600) as f32 / 100.0;
            let local = Vec3::new(f(1), f(2), 20.0 + f(3));
            let child = (k % 2 == 0).then(|| Vec3::new(f(4) / 20.0, -f(5) / 30.0, f(6) / 40.0));
            let mut region = ((k % 5) as i32 - 2, (k / 5 % 5) as i32 - 2);
            let mut record = placed(region, local, child);
            for (dx, dy) in moves {
                // the origin moves by (dx, dy) regions: the prim is that far
                // the other way
                let before = placed(region, local, child);
                region = (region.0 - dx, region.1 - dy);
                let delta = [-dx as f32 * 256.0, -dy as f32 * 256.0];
                shift_model(&mut record, delta);
                let fresh = placed(region, local, child);
                // placed, then moved once: settles without an upload (a
                // record moved several times keeps the rounding of a value
                // it held before; placed again it may be uploaded, 304 bytes)
                let mut once = before;
                shift_model(&mut once, delta);
                assert!(same_place(&once, &fresh, delta), "prim {k} in {region:?}: {once:?} / {fresh:?}");
                for a in 0..2 {
                    worst = worst.max((record[3][a] - fresh[3][a]).abs());
                }
            }
        }
        // at most a few units in the last place of 1.5 km: invisible
        assert!(worst < 5e-4, "{worst}");
    }

    #[test]
    fn a_real_move_is_not_rounding() {
        let rot = Quat::from_rotation_z(0.3);
        let at = model(Vec3::new(300.0, 40.0, 25.0), rot, Vec3::ONE);
        assert!(same_place(&at, &at, [0.0; 2]));
        // a centimetre, a turn, another height, another size: uploaded
        assert!(!same_place(&at, &model(Vec3::new(300.01, 40.0, 25.0), rot, Vec3::ONE), [0.0; 2]));
        assert!(!same_place(&at, &model(Vec3::new(300.0, 40.0, 25.0001), rot, Vec3::ONE), [0.0; 2]));
        assert!(!same_place(
            &at,
            &model(Vec3::new(300.0, 40.0, 25.0), Quat::from_rotation_z(0.30001), Vec3::ONE),
            [0.0; 2]
        ));
        assert!(!same_place(
            &at,
            &model(Vec3::new(300.0, 40.0, 25.0), rot, Vec3::splat(1.0001)),
            [0.0; 2]
        ));
        // far away the rounding is larger, a real move still is not
        let far = model(Vec3::new(1500.0, -900.0, 25.0), rot, Vec3::ONE);
        assert!(!same_place(
            &far,
            &model(Vec3::new(1500.005, -900.0, 25.0), rot, Vec3::ONE),
            [256.0, 0.0]
        ));
    }

    #[test]
    fn only_what_the_model_places_in_the_world_moves() {
        let mut r = DrawRecord::default();
        assert!(world_placed(&r));
        r.set_anchor(Some(3));
        assert!(!world_placed(&r), "worn: placed from its joint");
        let mut s = DrawRecord::default();
        s.flags[0] |= flags::SKINNED;
        assert!(!world_placed(&s), "skinned: placed from its palette");
    }

    #[test]
    fn the_pass_moves_what_the_flush_did_not_write() {
        let mut p = Pending::default();
        assert!(p.finish(&[], 100).is_none());
        assert!(p.begin([256.0, 0.0]));
        for id in [0, 5, 31, 32, 40, 99, 150] {
            p.add(id);
        }
        assert_eq!(p.count(), 7);
        // a second move before the flush: refused (all written again)
        assert!(!p.begin([0.0, 256.0]));
        // the flush wrote 30..41 itself; 150 is past the buffer
        let pass = p.finish(std::slice::from_ref(&(30..41)), 100).expect("a pass");
        assert_eq!(pass.delta, [256.0, 0.0]);
        assert_eq!(pass.slots, 100);
        assert_eq!(pass.mask.len(), 4);
        let moved: Vec<u32> = (0..100).filter(|i| pass.mask[*i as usize / 32] & (1 << (i % 32)) != 0).collect();
        assert_eq!(moved, [0, 5, 99]);
        // ended: the next one may begin
        assert!(!p.is_active());
        assert!(p.begin([0.0, -256.0]));
        p.add(7);
        assert!(p.finish(std::slice::from_ref(&(0..10)), 10).is_none(), "all written: no pass");
    }

    #[test]
    fn the_blocks_move_what_is_placed_in_the_world() {
        let block = 64;
        let mut anchored = DrawRecord::default();
        anchored.set_anchor(Some(2));
        let mut blocks: Vec<Vec<DrawRecord>> = (0..40).map(|_| vec![DrawRecord::default(); block]).collect();
        blocks[3][5] = anchored;
        blocks.push(vec![DrawRecord::default(); 10]);
        let slots = 40 * block + 10;
        // every other record, the anchored one and the last ones included
        let ids: Vec<u32> = (0..slots as u32)
            .filter(|i| i % 2 == 1)
            .chain([3 * 64 + 5, slots as u32 - 2])
            .collect();
        let mut mask = mask_of(ids.iter().copied().chain([slots as u32 + 5]), slots);
        let refused = shift_blocks(&mut blocks, &mut mask, block, [256.0, -512.0]);
        assert_eq!(refused, [3 * 64 + 5], "the anchored record is told");
        for (b, recs) in blocks.iter().enumerate() {
            for (k, r) in recs.iter().enumerate() {
                let i = b * block + k;
                let moved = (i % 2 == 1 || i == slots - 2) && i != 3 * 64 + 5;
                assert_eq!(r.model[3][..2] == [256.0, -512.0], moved, "record {i}");
                assert_eq!(mask[i / 32] & (1 << (i % 32)) != 0, moved, "bit {i}");
            }
        }
        // the anchored record kept its model, past the slots nothing is set
        assert_eq!(blocks[3][5].model[3][..2], [0.0, 0.0]);
        let mut p = Pending::default();
        p.begin([256.0, -512.0]);
        p.add_mask(mask);
        assert_eq!(p.count(), slots / 2 - 1 + 1);
        assert!(p.moved(1) && !p.moved(2) && !p.moved(3 * 64 + 5));
    }

    #[test]
    fn a_record_settles_once_per_move() {
        let at = model(Vec3::new(10.0, 20.0, 30.0), Quat::IDENTITY, Vec3::ONE);
        let near = model(Vec3::new(10.0 + 1e-6, 20.0, 30.0), Quat::IDENTITY, Vec3::ONE);
        let moved = model(Vec3::new(11.0, 20.0, 30.0), Quat::IDENTITY, Vec3::ONE);
        let mut p = Pending::default();
        assert!(!p.settles(1, &at, &near), "no move under way");
        p.begin([256.0, 0.0]);
        p.add(1);
        p.add(2);
        assert!(!p.settles(3, &at, &near), "not moved");
        assert!(!p.settles(2, &at, &moved), "really moved");
        assert!(!p.settles(2, &at, &near), "compared once");
        assert!(p.settles(1, &at, &near));
        assert!(!p.settles(1, &near, &near), "once");
        // the fallback gets every id moved
        assert_eq!(p.take_ids(), [1, 2]);
        assert!(!p.is_active());
    }

    #[test]
    fn the_dispatch_covers_every_slot() {
        assert_eq!(dispatch(0), (0, 0));
        assert_eq!(dispatch(1), (1, 1));
        assert_eq!(dispatch(64), (1, 1));
        assert_eq!(dispatch(65), (2, 1));
        let (x, y) = dispatch(10_000_000);
        assert_eq!(x, ROW);
        assert!(x as u64 * y as u64 * GROUP as u64 >= 10_000_000);
        let src = include_str!("shaders/origin_shift.wgsl");
        assert!(src.contains(&format!("@workgroup_size({GROUP})")));
        assert!(src.contains(&format!("const ROW: u32 = {ROW}u;")));
        assert!(src.contains(&format!("const HEADER: u32 = {HEADER}u;")));
    }

    #[test]
    fn the_shader_is_valid() {
        let src = include_str!("shaders/origin_shift.wgsl");
        let module = match naga::front::wgsl::parse_str(src) {
            Ok(m) => m,
            Err(e) => panic!("{}", e.emit_to_string(src)),
        };
        let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
        if let Err(e) = v.validate(&module) {
            panic!("{}", e.emit_to_string(src));
        }
    }

    #[test]
    fn the_header_tells_the_shader_the_record_layout() {
        let pass = Pass {
            delta: [-256.0, 512.0],
            slots: 3,
            mask: vec![0b101],
        };
        let w = words(&pass);
        assert_eq!(f32::from_bits(w[0]), -256.0);
        assert_eq!(f32::from_bits(w[1]), 512.0);
        assert_eq!(w[2], 3);
        // the translation is the record's fourth vector (model column 3)
        assert_eq!(w[3] as usize * 16, std::mem::size_of::<DrawRecord>());
        assert_eq!(std::mem::offset_of!(DrawRecord, model), 0);
        assert_eq!(w[HEADER], 0b101);
    }
}
