// aurora-local-patch: occl-stream-viewer-module
//! Occlusion streaming (local patch set): objects that stay hidden behind
//! others load less.
//!
//! The GPU's occlusion test of the main view (aurora_render::object_occlusion)
//! comes back a few frames late with, for every object of the table, whether
//! it was tested (it had a face in the view) and whether it was seen. An
//! object tested and not seen for `delay` (2 s by default) counts as hidden:
//! its textures are noted as out of view (`TextureStreamer::note_usage_away`:
//! not loaded finer for it, lowered under memory pressure as out-of-view
//! textures are) and its mesh keeps the detail level it has. An object never
//! seen since its faces were built counts as hidden at once: nothing finer
//! was loaded for it yet, so there is nothing to flicker.
//!
//! Any frame that sees it reveals it at once: its textures take the level
//! their size on screen asks for in the same frame (`TextureStreamer::reveal`)
//! and its detail level is checked again. An object out of the view is not
//! tested and keeps its state (the out-of-view path handles it anyway).
//!
//! What stays as before: avatars and everything they wear, HUDs, selected,
//! picked or highlighted objects, the faces a shadow samples (alpha-tested
//! casters), what stands in front of the mirror drawn this frame; and
//! everything while the CPU lists draw the scene (no GPU occlusion test),
//! while a second camera shows the world, or while a photo wants every
//! texture at full resolution.
//!
//! The Legacy Viewer sizes textures by their pixel area from the camera
//! whatever hides them (LLViewerTextureList::updateImageDecodePriority,
//! indra/newview/llviewertexturelist.cpp; LLFace::calcPixelArea,
//! indra/newview/llface.cpp): treating hidden faces as out of view is
//! Aurora's own, stricter than it on downloads only. It does freeze the
//! detail level of what its occlusion hides: LLPipeline::updateCull and
//! stateSort (indra/newview/pipeline.cpp) mark only the visible groups'
//! drawables, and only those reach LLDrawable::updateDistance and
//! LLVOVolume::updateLOD.

use super::GeomKey;
use aurora_render::object_occlusion::ObjectOcclusion;
use std::sync::OnceLock;
use std::time::Instant;

/// The clock of the states (s since the first call).
pub fn seconds(now: Instant) -> f64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    now.saturating_duration_since(*EPOCH.get_or_init(|| now)).as_secs_f64()
}

/// How long an object must stay hidden before it loads less, by default and
/// at the ends of its slider (s).
pub const DEFAULT_DELAY_S: f32 = 2.0;
pub const MIN_DELAY_S: f32 = 0.5;
pub const MAX_DELAY_S: f32 = 10.0;

/// The delay of the settings, bounded (a value that is not a number gives
/// the default).
pub fn bounded_delay(s: f32) -> f32 {
    if s.is_finite() {
        s.clamp(MIN_DELAY_S, MAX_DELAY_S)
    } else {
        DEFAULT_DELAY_S
    }
}

#[derive(Clone, Copy, Default, Debug, PartialEq)]
struct Slot {
    /// Since when (s, the scene's clock) it has been tested and not seen;
    /// None while seen or never tested.
    since: Option<f64>,
    /// Seen at least once since its slot was given to it.
    ever_seen: bool,
}

#[derive(Default)]
pub struct OcclusionStreaming {
    on: bool,
    delay: f64,
    slots: Vec<Slot>,
    /// Slots with `since` set (the only ones that can be hidden).
    candidates: Vec<usize>,
    /// Frame of the last verdict taken.
    last_index: Option<u64>,
    /// Objects hidden until the last verdict and seen in it.
    revealed: Vec<usize>,
    /// Objects counted as hidden at the last verdict (statistics).
    pub hidden_count: usize,
    /// Statistics since the last profile line: verdicts taken, objects
    /// revealed; when it was written.
    verdicts: u32,
    revealed_total: u32,
    logged_at: f64,
}

impl OcclusionStreaming {
    /// The settings: on or off, and the delay (s). Turning it off reveals
    /// every hidden object.
    pub fn configure(&mut self, on: bool, delay_s: f32) {
        self.delay = f64::from(bounded_delay(delay_s));
        if self.on && !on {
            self.reset_all();
        }
        self.on = on;
    }

    pub fn is_on(&self) -> bool {
        self.on
    }

    /// Every state forgotten, nothing revealed (the scene is emptied).
    pub fn clear(&mut self) {
        self.slots.clear();
        self.candidates.clear();
        self.revealed.clear();
        self.last_index = None;
        self.hidden_count = 0;
    }

    /// With AURORA_PROFILE, a line a second: objects tested, hidden, past
    /// the delay, revealed.
    pub fn profile_line(&mut self, now: f64) -> Option<String> {
        static PROFILE: OnceLock<bool> = OnceLock::new();
        if !*PROFILE.get_or_init(|| std::env::var_os("AURORA_PROFILE").is_some()) || now - self.logged_at < 1.0 {
            return None;
        }
        self.logged_at = now;
        let line = format!(
            "occlusion streaming: {} verdicts, {} objects hidden behind others ({} counted as hidden), {} revealed",
            std::mem::take(&mut self.verdicts),
            self.candidates.len(),
            self.hidden_count,
            std::mem::take(&mut self.revealed_total),
        );
        Some(line)
    }

    /// Every state forgotten; what was hidden is revealed.
    pub fn reset_all(&mut self) {
        for &i in &self.candidates {
            if self.slots.get(i).is_some_and(|s| s.since.is_some()) {
                self.revealed.push(i);
            }
        }
        self.slots.clear();
        self.candidates.clear();
        self.hidden_count = 0;
    }

    /// Slot `idx` was released (its object removed, or the slot given to
    /// another object).
    pub fn forget(&mut self, idx: usize) {
        if let Some(s) = self.slots.get_mut(idx) {
            *s = Slot::default();
        }
    }

    /// Take the verdict `o` if it is new; `now`: the scene's clock (s).
    pub fn ingest(&mut self, o: Option<&ObjectOcclusion>, now: f64) {
        let Some(o) = o.filter(|_| self.on) else {
            return;
        };
        if self.last_index.is_some_and(|i| i >= o.index) {
            return;
        }
        self.last_index = Some(o.index);
        self.verdicts += 1;
        if self.slots.len() < o.objects as usize {
            self.slots.resize(o.objects as usize, Slot::default());
        }
        for (i, seen) in o.tested_objects() {
            let i = i as usize;
            let was_hidden = self.hidden_at(i, now);
            let s = &mut self.slots[i];
            if seen {
                if was_hidden {
                    self.revealed.push(i);
                    self.revealed_total += 1;
                }
                s.since = None;
                s.ever_seen = true;
            } else if s.since.is_none() {
                s.since = Some(now);
                self.candidates.push(i);
            }
        }
        let slots = &self.slots;
        self.candidates.retain(|&i| slots.get(i).is_some_and(|s| s.since.is_some()));
        self.candidates.sort_unstable();
        self.candidates.dedup();
        self.hidden_count = self.candidates.iter().filter(|&&i| self.hidden_at(i, now)).count();
    }

    fn hidden_at(&self, idx: usize, now: f64) -> bool {
        self.slots
            .get(idx)
            .and_then(|s| s.since.map(|since| !s.ever_seen || now - since >= self.delay))
            .unwrap_or(false)
    }

    /// Object `idx` counts as hidden behind others now.
    pub fn hidden(&self, idx: usize, now: f64) -> bool {
        self.on && self.hidden_at(idx, now)
    }

    /// The objects revealed since the last call.
    pub fn take_revealed(&mut self) -> Vec<usize> {
        std::mem::take(&mut self.revealed)
    }
}

/// An object whose textures may follow its occlusion: not an avatar nor
/// worn, not selected, picked or highlighted, not in front of the mirror
/// drawn this frame (`kept` gathers those).
pub fn streams_less(avatar_or_worn: bool, kept: bool) -> bool {
    !avatar_or_worn && !kept
}

/// A face keeps its size on screen though its object is hidden: the shadow
/// pass samples its texture (an alpha-tested caster), and a shadow can fall
/// in view from behind a wall. `alpha`: the face is masked or blended (its
/// record has alpha flags); `blend`: blended; `blend_casts`: blended faces
/// cast too (aurora_render::shadow_cull::blend_casts).
pub fn face_keeps_size(casts: bool, alpha: bool, blend: bool, blend_casts: bool) -> bool {
    casts && alpha && (!blend || blend_casts)
}

/// The detail level a hidden object keeps: the one of the mesh it shows,
/// when it still shows the mesh it is asked for at that level (`wanted_at`);
/// otherwise the level computed from its distance. Only meshes (what is
/// downloaded); prims, sculpts and trees follow their distance as before.
pub fn frozen_lod(hidden: bool, bound: Option<GeomKey>, computed: u8, wanted_at: impl Fn(u8) -> Option<GeomKey>) -> u8 {
    match bound {
        Some(k @ GeomKey::Mesh { lod, .. }) if hidden && wanted_at(lod) == Some(k) => lod,
        _ => computed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    /// A verdict of frame `index` on `objects` objects: `tested` and `seen`
    /// list object numbers.
    fn verdict(index: u64, objects: u32, tested: &[u32], seen: &[u32]) -> ObjectOcclusion {
        let n = objects.div_ceil(32) as usize;
        let mut words = vec![0u32; 2 * n];
        for &i in tested {
            words[(i / 32) as usize] |= 1 << (i % 32);
        }
        for &i in seen {
            words[n + (i / 32) as usize] |= 1 << (i % 32);
        }
        ObjectOcclusion::from_words(index, objects, words).expect("both halves")
    }

    fn on() -> OcclusionStreaming {
        let mut s = OcclusionStreaming::default();
        s.configure(true, 2.0);
        s
    }

    #[test]
    fn a_seen_object_is_hidden_only_after_the_delay() {
        let mut s = on();
        s.ingest(Some(&verdict(1, 4, &[1], &[1])), 0.0);
        s.ingest(Some(&verdict(2, 4, &[1], &[])), 1.0);
        assert!(!s.hidden(1, 1.0));
        assert!(!s.hidden(1, 2.9));
        assert!(s.hidden(1, 3.0));
        assert!(s.take_revealed().is_empty());
    }

    #[test]
    fn an_object_never_seen_is_hidden_at_once() {
        let mut s = on();
        s.ingest(Some(&verdict(1, 4, &[2], &[])), 5.0);
        assert!(s.hidden(2, 5.0));
        assert_eq!(s.hidden_count, 1);
        // never tested: as before
        assert!(!s.hidden(3, 5.0));
    }

    #[test]
    fn one_frame_that_sees_it_reveals_it_once() {
        let mut s = on();
        s.ingest(Some(&verdict(1, 4, &[0], &[])), 0.0);
        assert!(s.hidden(0, 0.0));
        s.ingest(Some(&verdict(2, 4, &[0], &[0])), 0.1);
        assert!(!s.hidden(0, 0.1));
        assert_eq!(s.take_revealed(), vec![0]);
        assert!(s.take_revealed().is_empty());
        // seen now: hidden again only after the whole delay
        s.ingest(Some(&verdict(3, 4, &[0], &[])), 0.2);
        assert!(!s.hidden(0, 2.1));
        assert!(s.hidden(0, 2.2));
    }

    #[test]
    fn out_of_the_view_an_object_keeps_its_state() {
        let mut s = on();
        s.ingest(Some(&verdict(1, 4, &[1], &[1])), 0.0);
        s.ingest(Some(&verdict(2, 4, &[1], &[])), 0.0);
        // not tested (the camera turned away): still counting
        s.ingest(Some(&verdict(3, 4, &[], &[])), 1.0);
        assert!(s.hidden(1, 2.0));
    }

    #[test]
    fn hidden_and_seen_in_turn_never_counts_as_hidden() {
        let mut s = on();
        s.ingest(Some(&verdict(0, 1, &[0], &[0])), 0.0);
        for k in 1..100u64 {
            let t = k as f64 * 0.1;
            let seen: &[u32] = if k % 3 == 0 { &[0] } else { &[] };
            s.ingest(Some(&verdict(k, 1, &[0], seen)), t);
            assert!(!s.hidden(0, t), "frame {k}");
        }
    }

    #[test]
    fn a_released_slot_starts_afresh() {
        let mut s = on();
        s.ingest(Some(&verdict(1, 4, &[3], &[])), 0.0);
        assert!(s.hidden(3, 0.0));
        s.forget(3);
        assert!(!s.hidden(3, 0.0));
        // a new object there, never seen: hidden at its first verdict
        s.ingest(Some(&verdict(2, 4, &[3], &[])), 0.1);
        assert!(s.hidden(3, 0.1));
    }

    #[test]
    fn an_older_or_repeated_verdict_is_ignored() {
        let mut s = on();
        s.ingest(Some(&verdict(5, 4, &[1], &[1])), 0.0);
        s.ingest(Some(&verdict(4, 4, &[1], &[])), 0.0);
        s.ingest(Some(&verdict(5, 4, &[1], &[])), 0.0);
        assert!(!s.hidden(1, 100.0));
    }

    #[test]
    fn off_nothing_is_hidden_and_turning_off_reveals_all() {
        let mut s = OcclusionStreaming::default();
        s.ingest(Some(&verdict(1, 4, &[1], &[])), 0.0);
        assert!(!s.hidden(1, 0.0));
        let mut s = on();
        s.ingest(Some(&verdict(1, 4, &[1, 2], &[])), 0.0);
        s.configure(false, 2.0);
        assert!(!s.hidden(1, 0.0));
        let mut revealed = s.take_revealed();
        revealed.sort_unstable();
        assert_eq!(revealed, vec![1, 2]);
    }

    #[test]
    fn the_delay_is_bounded() {
        assert_eq!(bounded_delay(0.0), MIN_DELAY_S);
        assert_eq!(bounded_delay(60.0), MAX_DELAY_S);
        assert_eq!(bounded_delay(f32::NAN), DEFAULT_DELAY_S);
        assert_eq!(bounded_delay(3.5), 3.5);
    }

    #[test]
    fn avatars_worn_and_kept_objects_never_stream_less() {
        assert!(streams_less(false, false));
        assert!(!streams_less(true, false));
        assert!(!streams_less(false, true));
    }

    #[test]
    fn the_faces_a_shadow_samples_keep_their_size() {
        // masked caster
        assert!(face_keeps_size(true, true, false, false));
        // blended caster, only when blended faces cast
        assert!(face_keeps_size(true, true, true, true));
        assert!(!face_keeps_size(true, true, true, false));
        // opaque, or not casting
        assert!(!face_keeps_size(true, false, false, true));
        assert!(!face_keeps_size(false, true, false, true));
    }

    #[test]
    fn a_hidden_mesh_keeps_its_level_while_it_shows_the_same_mesh() {
        let id = Uuid::from_u128(7);
        let mesh = |lod| Some(GeomKey::Mesh { id, lod });
        // hidden, bound at 1, asked for 3 from its distance: stays at 1
        assert_eq!(frozen_lod(true, mesh(1), 3, mesh), 1);
        // and at 3 when asked for 0: no lower level downloaded either
        assert_eq!(frozen_lod(true, mesh(3), 0, mesh), 3);
        // seen: its distance decides
        assert_eq!(frozen_lod(false, mesh(1), 3, mesh), 3);
        // another mesh asked for (the object changed): its distance decides
        let other = |lod| {
            Some(GeomKey::Mesh {
                id: Uuid::from_u128(8),
                lod,
            })
        };
        assert_eq!(frozen_lod(true, mesh(1), 3, other), 3);
        // a prim follows its distance
        let prim = |lod| Some(GeomKey::Prim { hash: 1, lod });
        assert_eq!(frozen_lod(true, prim(1), 3, prim), 3);
        // nothing bound yet: loaded at the level its distance asks for
        assert_eq!(frozen_lod(true, None, 2, mesh), 2);
    }
}
