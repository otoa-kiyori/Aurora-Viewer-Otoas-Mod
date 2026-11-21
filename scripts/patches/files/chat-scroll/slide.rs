//! The lines of a conversation slide up to a new message instead of jumping
//! there (Preferences › Chat › Conversations › Scroll speed). Aurora's own:
//! the Legacy Viewer moves its text at once.
//!
//! `View::show` places the rows against the end of a conversation it
//! follows. When lines come in, the rows in sight would move up by their
//! height in one frame; while a slide runs they are drawn `offset` lower
//! instead, an offset that eases out (`emath::easing::cubic_out`, as the
//! notification center's cards) to 0 over the Scroll speed. A message that
//! comes in during a slide adds its height to what is left and starts the
//! time again from there: the lines in sight never step back nor jump, and
//! they reach the end within the Scroll speed of the latest message.
//!
//! Only a view that follows the end slides: one scrolled back (a menu open,
//! older lines read) does not move, and every jump asked for (find, the
//! latest messages, another conversation, a window opened, a resize) stays
//! at once. Reduced motion (egui's `animation_time` at 0) jumps too.

// aurora-local-patch: chat-scroll-slide

/// A slide in progress (the default: none).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Slide {
    /// The offset when it (re)started, in points.
    from: f32,
    /// When it (re)started, egui's input time in seconds.
    start: f64,
    /// How long it lasts from `start`, in seconds.
    seconds: f32,
}

/// One frame of a conversation, as its slide sees it.
#[derive(Clone, Copy, Debug)]
pub(super) struct Frame {
    /// egui's input time, in seconds.
    pub now: f64,
    /// The Scroll speed, in seconds (0: at once; also with reduced motion).
    pub seconds: f32,
    /// The view was shown at the pass before this one too.
    pub shown_before: bool,
    /// Its rows stand against the end of the conversation this frame.
    pub at_end: bool,
    /// They stood there at the last show.
    pub was_at_end: bool,
    /// How far the rows in sight move up because lines came in since the
    /// last show (`rise`).
    pub arrived: f32,
    /// The viewport's height: never more than this is left to slide.
    pub limit: f32,
}

/// What a frame does with the slide.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Step {
    /// How much lower than their place at the end the rows are drawn.
    pub offset: f32,
    /// What was left of a slide when the reader left the end during it:
    /// the view keeps the rows where they were drawn (its correction of
    /// the offset), so that leaving does not jump either.
    pub dropped: f32,
}

impl Slide {
    /// How much lower the rows are drawn at `now`: 0 once settled.
    pub fn offset(&self, now: f64) -> f32 {
        if !(self.from > 0.0 && self.seconds > 0.0) || now >= self.start + f64::from(self.seconds) {
            return 0.0;
        }
        let t = ((now - self.start) / f64::from(self.seconds)).clamp(0.0, 1.0) as f32;
        (self.from * (1.0 - egui::emath::easing::cubic_out(t))).max(0.0)
    }

    /// The slide of frame `f`.
    pub fn step(&mut self, f: Frame) -> Step {
        let following = f.shown_before && f.at_end && f.was_at_end;
        if !following || f.seconds <= 0.0 {
            let left = f.shown_before && f.was_at_end && !f.at_end;
            let dropped = if left { self.offset(f.now) } else { 0.0 };
            *self = Slide::default();
            return Step { offset: 0.0, dropped };
        }
        if f.arrived > 0.0 {
            // from where the rows are drawn now: what they had left, plus
            // the lines that just came in
            let from = (self.offset(f.now) + f.arrived).min(f.limit.max(0.0));
            *self = Slide {
                from,
                start: f.now,
                seconds: f.seconds,
            };
        }
        Step {
            offset: self.offset(f.now),
            dropped: 0.0,
        }
    }
}

/// How far the rows in sight move up when `appended` points of new rows end
/// a conversation now `total` high, in a viewport `view_height` high: all
/// of it once the conversation fills the viewport; in a shorter one,
/// nothing when its rows start from the top (the classic lines), all of it
/// when they stand against the bottom (Aurora's workspace).
pub(super) fn rise(total: f32, appended: f32, view_height: f32, bottom_aligned: bool) -> f32 {
    let end = |total: f32| if bottom_aligned { total } else { total.max(view_height) };
    if appended > 0.0 {
        (end(total) - end(total - appended)).max(0.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMIT: f32 = 300.0;

    fn frame(now: f64, seconds: f32, arrived: f32) -> Frame {
        Frame {
            now,
            seconds,
            shown_before: true,
            at_end: true,
            was_at_end: true,
            arrived,
            limit: LIMIT,
        }
    }

    /// Frames of 1/60 s, `frames` of them, with `arrivals` (frame,
    /// height): where an old line in sight is drawn, up from its place
    /// before the first arrival, and the offsets.
    fn run(seconds: f32, arrivals: &[(usize, f32)], frames: usize) -> (Vec<f32>, Vec<f32>) {
        let mut slide = Slide::default();
        let (mut risen, mut places, mut offsets) = (0.0, Vec::new(), Vec::new());
        for n in 0..frames {
            let arrived: f32 = arrivals.iter().filter(|(at, _)| *at == n).map(|(_, h)| h).sum();
            risen += arrived;
            let step = slide.step(frame(n as f64 / 60.0, seconds, arrived));
            assert_eq!(step.dropped, 0.0);
            places.push(risen - step.offset);
            offsets.push(step.offset);
        }
        (places, offsets)
    }

    #[test]
    fn no_scroll_speed_jumps_at_once() {
        let mut slide = Slide::default();
        let step = slide.step(frame(1.0, 0.0, 40.0));
        assert_eq!(step, Step::default());
        assert_eq!(slide, Slide::default());
        let (places, offsets) = run(0.0, &[(30, 20.0)], 60);
        assert!(offsets.iter().all(|&o| o == 0.0));
        assert_eq!(places[29], 0.0);
        assert_eq!(places[30], 20.0, "the whole height on the arrival's frame");
    }

    #[test]
    fn a_slide_reaches_the_end_exactly_at_its_duration() {
        for seconds in [0.05, 0.3, 0.5, 1.0] {
            let mut slide = Slide::default();
            let start = 2.0;
            assert_eq!(slide.step(frame(start, seconds, 24.0)).offset, 24.0, "drawn where it was");
            let end = start + f64::from(seconds);
            assert!(slide.offset(end - f64::from(seconds) * 0.05) > 0.0, "{seconds}");
            assert_eq!(slide.offset(end), 0.0, "{seconds}");
            assert_eq!(slide.offset(end + 5.0), 0.0, "{seconds}");
            assert!(
                slide.offset(start + f64::from(seconds) / 2.0) < 12.0,
                "eased out: most of the way in the first half"
            );
        }
    }

    #[test]
    fn a_slide_only_moves_forward_and_never_overshoots() {
        let (places, offsets) = run(0.5, &[(6, 20.0)], 60);
        for pair in places.windows(2) {
            assert!(pair[1] >= pair[0], "{places:?}");
        }
        assert!(offsets.iter().all(|&o| (0.0..=20.0).contains(&o)), "{offsets:?}");
        assert!(places.iter().all(|&p| p <= 20.0), "never past the end: {places:?}");
        assert_eq!(*places.last().unwrap(), 20.0);
        // intermediate positions, not a jump
        assert!(places.iter().any(|&p| p > 2.0 && p < 18.0));
    }

    #[test]
    fn messages_during_a_slide_retarget_it_without_a_step_back() {
        // three messages close together, then one more near the end
        let arrivals = [(6, 20.0), (12, 36.0), (15, 20.0), (33, 20.0)];
        let (places, offsets) = run(0.5, &arrivals, 120);
        for pair in places.windows(2) {
            assert!(pair[1] >= pair[0], "{places:?}");
        }
        // a retarget continues from where the rows are drawn: on the
        // arrival's frame the old lines move a frame's worth, not the
        // message's height
        for &(at, h) in &arrivals {
            let moved = places[at] - places[at - 1];
            assert!((0.0..h / 4.0).contains(&moved), "frame {at}: {moved} {places:?}");
        }
        // settled within the Scroll speed (30 frames) of the latest
        // message, at the end
        assert!(offsets[33 + 28] > 0.0, "{offsets:?}");
        assert!(offsets[33 + 31..].iter().all(|&o| o == 0.0), "{offsets:?}");
        assert_eq!(*places.last().unwrap(), 96.0);
    }

    #[test]
    fn a_long_arrival_slides_a_viewport_at_most() {
        let mut slide = Slide::default();
        assert_eq!(slide.step(frame(0.0, 0.3, 5000.0)).offset, LIMIT);
        assert_eq!(slide.step(frame(0.01, 0.3, 200.0)).offset, LIMIT);
    }

    #[test]
    fn a_view_away_from_the_end_does_not_slide() {
        let mut slide = Slide::default();
        // scrolled back: lines come in, nothing moves
        let away = Frame {
            at_end: false,
            was_at_end: false,
            ..frame(1.0, 0.5, 40.0)
        };
        assert_eq!(slide.step(away), Step::default());
        assert_eq!(slide, Slide::default());
        // back at the end by a jump: at once
        let back = Frame {
            was_at_end: false,
            ..frame(1.1, 0.5, 40.0)
        };
        assert_eq!(slide.step(back), Step::default());
        // shown again after another conversation (not the pass before)
        let again = Frame {
            shown_before: false,
            ..frame(1.2, 0.5, 40.0)
        };
        assert_eq!(slide.step(again), Step::default());
    }

    #[test]
    fn leaving_the_end_during_a_slide_keeps_the_rows_where_they_are() {
        let mut slide = Slide::default();
        slide.step(frame(0.0, 0.5, 30.0));
        let drawn = slide.step(frame(0.1, 0.5, 0.0)).offset;
        assert!(drawn > 0.0 && drawn < 30.0);
        // the wheel took the view off the end at the next frame
        let left = Frame {
            at_end: false,
            ..frame(0.1, 0.5, 0.0)
        };
        assert_eq!(
            slide.step(left),
            Step {
                offset: 0.0,
                dropped: drawn
            }
        );
        assert_eq!(slide, Slide::default(), "stopped");
        // a menu opened over the end (paused): the same
        let mut slide = Slide::default();
        slide.step(frame(0.0, 0.5, 30.0));
        let paused = Frame {
            at_end: false,
            ..frame(0.0, 0.5, 0.0)
        };
        assert_eq!(slide.step(paused).dropped, 30.0);
    }

    #[test]
    fn reduced_motion_stops_a_slide_at_once() {
        let mut slide = Slide::default();
        assert!(slide.step(frame(0.0, 0.5, 30.0)).offset > 0.0);
        // `View::show` gives 0 s while egui's animation_time is 0
        assert_eq!(slide.step(frame(0.05, 0.0, 0.0)), Step::default());
        assert_eq!(slide.step(frame(0.06, 0.0, 30.0)), Step::default());
    }

    #[test]
    fn the_rows_in_sight_rise_by_what_their_layout_moves() {
        // a conversation that fills the viewport: the new row's height
        assert_eq!(rise(1000.0, 40.0, 300.0, false), 40.0);
        assert_eq!(rise(1000.0, 40.0, 300.0, true), 40.0);
        // short classic lines start from the top: a new one below moves nothing
        assert_eq!(rise(200.0, 40.0, 300.0, false), 0.0);
        // the one that makes it overflow moves them by the overflow
        assert_eq!(rise(320.0, 40.0, 300.0, false), 20.0);
        // the workspace stands against the bottom: always the whole row
        assert_eq!(rise(200.0, 40.0, 300.0, true), 40.0);
        assert_eq!(rise(1000.0, 0.0, 300.0, true), 0.0);
        assert_eq!(rise(1000.0, -5.0, 300.0, true), 0.0);
    }
}
