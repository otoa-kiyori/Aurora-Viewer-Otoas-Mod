// aurora-local-patch: minimize-stall-module
//! Whether a frame the render thread keeps is held by the GPU or by the
//! compositor.
//!
//! The stall notice (aurora-viewer app/gpu_stall.rs) shows when the render
//! thread has kept a frame for seconds: a graphics card that stopped
//! answering. But a window that is not shown keeps it too. On Wayland a
//! minimized window, or one on another virtual desktop, gets no frame
//! callbacks, and with a synchronized present mode (Fifo, FifoRelaxed) the
//! driver's `vkQueuePresentKHR` waits for the next one: NVIDIA's waits in
//! `wl_display_dispatch_queue` for as long as the window stays hidden (the
//! render thread's stack of a "GPU stall" session, 2026-11-08). Nothing is
//! wrong with the card, and the notice told the user it does not answer.
//!
//! So the render thread notes when it enters and leaves `present`, and
//! whether the GPU had finished every frame before the presented one by
//! then (it waits a little for it, see `Backend::earlier_frames_done`).
//! A frame kept inside `present` with the GPU caught up, while the window
//! is not in front, is the compositor's: the drawing is paused, the world,
//! the network and the sound go on, and the stall clock does not run. The
//! Legacy Viewer does the same with a hidden or minimized window: `display`
//! draws nothing then (indra/newview/llviewerdisplay.cpp, "Don't draw if
//! the window is hidden or minimized"), and `LLAppViewer::mainLoop` keeps
//! the world running, yielding `BackgroundYieldTime` each frame
//! (indra/newview/llappviewer.cpp).
//!
//! A real hang still counts: a GPU that has not finished the frame before
//! (the present waits for our own work), any other wait of the render
//! thread (acquire, fences, uploads), and a present that keeps the frame
//! while the window is in front, counted from the moment it came back.

use parking_lot::Mutex;
use std::time::{Duration, Instant};

/// Inside `present` this long, the GPU caught up and the window not in
/// front: the compositor is not taking images. A shown window's present
/// returns within a refresh or two; a hidden one's waits for ever.
pub(crate) const COMPOSITOR_WAIT: Duration = Duration::from_millis(250);

#[derive(Clone, Copy, Debug)]
struct Presenting {
    since: Instant,
    /// Every frame before the one presented was finished on the GPU when
    /// the present began.
    gpu_done: bool,
}

/// What the render thread is doing with the swapchain, read by the main
/// thread. Two short locks per frame on the render thread.
#[derive(Default)]
pub(crate) struct PresentWatch(Mutex<Option<Presenting>>);

impl PresentWatch {
    /// The render thread calls `present`.
    pub fn enter(&self, gpu_done: bool) {
        *self.0.lock() = Some(Presenting {
            since: Instant::now(),
            gpu_done,
        });
    }

    /// `present` returned.
    pub fn leave(&self) {
        *self.0.lock() = None;
    }

    /// How long the render thread has been inside `present` with the GPU
    /// caught up; None when it is not in `present`, or the GPU was behind.
    pub fn compositor_wait(&self) -> Option<Duration> {
        self.0.lock().filter(|p| p.gpu_done).map(|p| p.since.elapsed())
    }
}

/// The render thread waits inside `present` with the GPU caught up, for
/// longer than a shown window's present takes: most likely a hidden window.
/// The main thread stops waiting for the frame then, and handles its events
/// (the focus leaving comes with the minimize) before it decides.
pub(crate) fn present_waits(compositor_wait: Option<Duration>) -> bool {
    compositor_wait.is_some_and(|waited| waited >= COMPOSITOR_WAIT)
}

/// Whether the frame kept is the compositor's (no GPU stall): inside
/// `present` long enough with the GPU caught up, and the window not in
/// front (minimized, hidden, behind or unfocused).
pub(crate) fn compositor_holds(compositor_wait: Option<Duration>, in_front: bool) -> bool {
    !in_front && present_waits(compositor_wait)
}

/// The part of a frame held `held` that counts as a GPU stall: when the
/// compositor held it until `excused_ago` (the window was hidden), only the
/// time since. A frame kept on after the window came back is counted from
/// then, not from when it was handed over.
pub(crate) fn counted_stall(held: Duration, excused_ago: Option<Duration>) -> Duration {
    excused_ago.map_or(held, |ago| held.min(ago))
}

#[cfg(test)]
mod tests {
    use super::*;

    const LONG: Duration = Duration::from_secs(30);

    #[test]
    fn a_hidden_window_s_present_with_the_gpu_done_is_the_compositor_s() {
        // minimized / on another desktop: the present waits, nothing is wrong
        assert!(compositor_holds(Some(LONG), false));
        assert!(compositor_holds(Some(COMPOSITOR_WAIT), false));
    }

    #[test]
    fn a_long_present_stops_the_main_thread_s_wait_in_front_or_not() {
        assert!(present_waits(Some(COMPOSITOR_WAIT)));
        assert!(!present_waits(Some(Duration::from_millis(16))));
        assert!(!present_waits(None));
    }

    #[test]
    fn a_short_present_is_not_yet_a_hold() {
        assert!(!compositor_holds(Some(Duration::from_millis(16)), false));
        assert!(!compositor_holds(Some(COMPOSITOR_WAIT - Duration::from_millis(1)), false));
    }

    #[test]
    fn a_window_in_front_or_another_wait_is_still_a_stall() {
        // in front: the compositor takes images, a present that waits is a hang
        assert!(!compositor_holds(Some(LONG), true));
        // not in present (acquire, a fence, uploads), or the GPU was behind
        // when it began (the present waits for our own work): `None`
        assert!(!compositor_holds(None, false));
        assert!(!compositor_holds(None, true));
    }

    #[test]
    fn the_watch_only_excuses_a_present_with_the_gpu_caught_up() {
        let watch = PresentWatch::default();
        assert_eq!(watch.compositor_wait(), None, "not presenting");
        watch.enter(false);
        assert_eq!(watch.compositor_wait(), None, "the GPU was behind");
        watch.leave();
        watch.enter(true);
        assert!(watch.compositor_wait().is_some());
        watch.leave();
        assert_eq!(watch.compositor_wait(), None, "present returned");
    }

    #[test]
    fn the_watch_measures_from_the_present_s_start() {
        let watch = PresentWatch::default();
        watch.enter(true);
        std::thread::sleep(Duration::from_millis(20));
        let waited = watch.compositor_wait().expect("in present");
        assert!(waited >= Duration::from_millis(20), "{waited:?}");
    }

    #[test]
    fn the_stall_counts_only_once_the_window_is_back() {
        // never hidden: the whole hold
        assert_eq!(counted_stall(LONG, None), LONG);
        // hidden for most of it, back 3 s ago: 3 s
        assert_eq!(counted_stall(LONG, Some(Duration::from_secs(3))), Duration::from_secs(3));
        // excused longer ago than the hold began (an earlier hold): the hold
        assert_eq!(counted_stall(Duration::from_secs(2), Some(LONG)), Duration::from_secs(2));
    }
}
