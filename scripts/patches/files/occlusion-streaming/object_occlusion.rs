// aurora-local-patch: occl-stream-render-module
//! Which objects the occlusion culling found hidden behind others, read back
//! for the streaming of their textures and meshes (local patch set:
//! occlusion streaming, the viewer's scene/occlusion_streaming.rs).
//!
//! The occlusion test of the main view's draw lists (cull.wgsl
//! `cs_occlude`) already tests one sphere per object for every face it
//! keeps. Its result is folded into two bits per object of the scene's
//! object table: tested (the object had a face in the main view's bins)
//! and seen (its sphere, slightly enlarged, passed the Hi-Z test). The
//! enlarged sphere (`INFLATE_SCALE`, `INFLATE_ADD`) makes an object that
//! is about to come out from behind a wall count as seen a little early;
//! the draws themselves keep the exact test.
//!
//! The bits are copied with the culling's counts into a ring of mapped
//! buffers and read back a few frames later, never waited for. A frame
//! whose main view is not the user's camera (a photo or a film frame at
//! its own size, a face of a 360° capture, an external output) gives none.
//!
//! The Legacy Viewer reads its occlusion queries back the same way, a frame
//! or more late (LLOcclusionCullingGroup::checkOcclusion,
//! indra/newview/llspatialpartition.cpp), and freezes the detail level of
//! what they hide (LLPipeline::updateCull and stateSort,
//! indra/newview/pipeline.cpp, only mark the visible groups' drawables).

use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// The sphere tested for "seen": the object's own, this much larger
/// (mirrored in cull.wgsl `cs_occlude`).
pub const INFLATE_SCALE: f32 = 1.25;
pub const INFLATE_ADD: f32 = 0.5;

/// Words of one half (tested or seen) for `objects` objects.
pub fn words_for(objects: u32) -> usize {
    objects.div_ceil(32) as usize
}

/// One frame's verdict on every object of the scene's object table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectOcclusion {
    /// The frame (`FramePacket::index`) it was tested in.
    pub index: u64,
    /// Objects of the table that frame.
    pub objects: u32,
    /// `words_for(objects)` words of tested bits, then as many of seen bits.
    words: Vec<u32>,
}

impl ObjectOcclusion {
    /// None when `words` does not hold both halves for `objects`.
    pub fn from_words(index: u64, objects: u32, words: Vec<u32>) -> Option<Self> {
        (words.len() == 2 * words_for(objects)).then_some(ObjectOcclusion { index, objects, words })
    }

    fn bit(&self, half: usize, i: u32) -> bool {
        i < self.objects && self.words[half + (i / 32) as usize] & (1 << (i % 32)) != 0
    }

    /// The object had a face in the main view's bins that frame.
    pub fn tested(&self, i: u32) -> bool {
        self.bit(0, i)
    }

    /// Its enlarged sphere passed the occlusion test that frame.
    pub fn seen(&self, i: u32) -> bool {
        self.bit(words_for(self.objects), i)
    }

    /// Every tested object, with whether it was seen.
    pub fn tested_objects(&self) -> impl Iterator<Item = (u32, bool)> + '_ {
        let half = words_for(self.objects);
        self.words[..half].iter().enumerate().flat_map(move |(w, &tested)| {
            let seen = self.words[half + w];
            (0..32u32)
                .filter(move |b| tested & (1 << b) != 0)
                .map(move |b| (w as u32 * 32 + b, seen & (1 << b) != 0))
        })
    }
}

struct Slot {
    buffer: wgpu::Buffer,
    ready: Arc<AtomicBool>,
    pending: bool,
    /// Frame and object count of the copy it holds.
    meta: (u64, u32),
}

/// The bits on the GPU (cull.wgsl binding 14) and their readbacks.
pub(crate) struct ObjectBits {
    /// Both halves for `cap` objects; a placeholder of one word while off
    /// (the shader then writes nothing: `arrayLength` is too short).
    buffer: wgpu::Buffer,
    cap: u32,
    slots: Vec<Slot>,
    /// Wanted at all (the viewer's setting), and this frame's index when
    /// its main view is the user's camera.
    on: Cell<bool>,
    frame: Cell<Option<u64>>,
    /// Objects written by this frame's occlusion pass (none yet: None).
    written: Cell<Option<u32>>,
    /// Readback slot, frame and objects of this frame's copy.
    copied: Cell<Option<(usize, u64, u32)>>,
    last: Option<Arc<ObjectOcclusion>>,
}

fn placeholder(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cull object bits (off)"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

impl ObjectBits {
    pub fn new(device: &wgpu::Device) -> Self {
        ObjectBits {
            buffer: placeholder(device),
            cap: 0,
            slots: Vec::new(),
            on: Cell::new(false),
            frame: Cell::new(None),
            written: Cell::new(None),
            copied: Cell::new(None),
            last: None,
        }
    }

    /// This frame: wanted or not, and its index when it may be read back.
    pub fn begin_frame(&self, on: bool, frame: Option<u64>) {
        self.on.set(on);
        self.frame.set(frame.filter(|_| on));
        self.written.set(None);
    }

    /// Room for `objects` objects while wanted, the placeholder otherwise
    /// (`GpuCull::prepare`, which sizes the spheres for as many).
    pub fn ensure(&mut self, device: &wgpu::Device, objects: u32) {
        if !self.on.get() {
            if self.cap != 0 {
                self.buffer = placeholder(device);
                self.cap = 0;
                self.slots.clear();
            }
            return;
        }
        if objects <= self.cap {
            return;
        }
        let bytes = (2 * words_for(objects) * 4) as u64;
        self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull object bits"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // (a readback in flight on the old ones is dropped)
        self.slots = (0..3)
            .map(|_| Slot {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("cull object bits readback"),
                    size: crate::readback::readback_size(bytes),
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                ready: Arc::new(AtomicBool::new(false)),
                pending: false,
                meta: (0, 0),
            })
            .collect();
        self.cap = objects;
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Before the occlusion pass: both halves of this frame's `objects`
    /// cleared, when the frame is read back.
    pub fn clear(&self, encoder: &mut wgpu::CommandEncoder, objects: u32) {
        if self.frame.get().is_none() || objects > self.cap || objects == 0 {
            return;
        }
        let bytes = (2 * words_for(objects) * 4) as u64;
        encoder.clear_buffer(&self.buffer, 0, Some(bytes));
        self.written.set(Some(objects));
    }

    /// After the occlusion pass: this frame's bits into a free readback.
    pub fn copy(&self, encoder: &mut wgpu::CommandEncoder) {
        let (Some(index), Some(objects)) = (self.frame.take(), self.written.take()) else {
            return;
        };
        if self.copied.get().is_some() {
            return;
        }
        if let Some(slot) = self.slots.iter().position(|s| !s.pending) {
            let bytes = (2 * words_for(objects) * 4) as u64;
            encoder.copy_buffer_to_buffer(&self.buffer, 0, &self.slots[slot].buffer, 0, bytes);
            self.copied.set(Some((slot, index, objects)));
        }
    }

    /// Map this frame's copy (after the submit).
    pub fn after_submit(&mut self) {
        let Some((slot, index, objects)) = self.copied.take() else {
            return;
        };
        let Some(s) = self.slots.get_mut(slot) else {
            return;
        };
        s.pending = true;
        s.meta = (index, objects);
        let ready = s.ready.clone();
        s.buffer.map_async(wgpu::MapMode::Read, .., move |r| {
            if r.is_ok() {
                ready.store(true, Ordering::Release);
            }
        });
    }

    /// Collect the finished readbacks (never waits): the newest is kept
    /// until taken.
    pub fn harvest(&mut self) {
        for s in &mut self.slots {
            if !(s.pending && s.ready.load(Ordering::Acquire)) {
                continue;
            }
            let (index, objects) = s.meta;
            let n = 2 * words_for(objects);
            if let Ok(view) = s.buffer.get_mapped_range(..)
                && view.len() >= n * 4
            {
                let words = view[..n * 4].as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
                if let Some(o) = ObjectOcclusion::from_words(index, objects, words)
                    && self.last.as_ref().is_none_or(|l| l.index < index)
                {
                    self.last = Some(Arc::new(o));
                }
            }
            s.buffer.unmap();
            s.pending = false;
            s.ready.store(false, Ordering::Release);
        }
    }

    /// The newest verdict read back since the last call.
    pub fn take(&mut self) -> Option<Arc<ObjectOcclusion>> {
        self.last.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bits_tell_tested_and_seen_objects() {
        // 40 objects: two words a half; 1 and 33 tested, 33 seen
        let words = vec![1 << 1, 1 << 1, 0, 1 << 1];
        let o = ObjectOcclusion::from_words(7, 40, words).expect("both halves");
        assert!(o.tested(1) && !o.seen(1));
        assert!(o.tested(33) && o.seen(33));
        assert!(!o.tested(0) && !o.tested(39));
        // past the table: never tested
        assert!(!o.tested(64) && !o.seen(64));
        assert_eq!(o.tested_objects().collect::<Vec<_>>(), vec![(1, false), (33, true)]);
    }

    #[test]
    fn a_readback_of_the_wrong_length_is_refused() {
        assert!(ObjectOcclusion::from_words(1, 40, vec![0; 3]).is_none());
        assert!(ObjectOcclusion::from_words(1, 0, Vec::new()).is_some());
        assert_eq!(words_for(32), 1);
        assert_eq!(words_for(33), 2);
    }

    /// The enlargement of the sphere is the one the shader applies.
    #[test]
    fn the_shader_enlarges_the_sphere_as_documented() {
        let src = include_str!("shaders/cull.wgsl");
        assert!(
            src.contains(&format!("s.w * {INFLATE_SCALE:?} + {INFLATE_ADD:?}")),
            "cs_occlude's enlarged sphere"
        );
    }
}
