// aurora-local-patch: director-live-frames-module
//! The Director's live film, several frames in flight (local patch set,
//! 113-director-live-frames): the pure rules. `app/director.rs` and
//! `app/director_render.rs` own the film; the readback is aurora-render's
//! film_readback.rs, which answers in the order the frames were asked.

use std::ops::RangeInclusive;

/// The live film frames the user may have in flight at once (the Output
/// section's "Frames fetched at once", `recorder::Options::in_flight`): 1
/// is upstream's one at a time. film_readback.rs keeps up to `POOL` (12)
/// buffers for reuse: the most of this range plus an offline render's two.
pub const IN_FLIGHT_RANGE: RangeInclusive<u32> = 1..=10;

/// The default: a frame's pixels come back about three render frames after
/// it is drawn, so three in flight let the film take a view every render
/// frame; more only help when the card answers later than that.
pub const IN_FLIGHT_DEFAULT: u32 = 3;

/// A value of the settings file, brought into `IN_FLIGHT_RANGE`.
pub fn in_flight(value: u32) -> u32 {
    value.clamp(*IN_FLIGHT_RANGE.start(), *IN_FLIGHT_RANGE.end())
}

/// Render frames a live film that ends waits, at most, for its frames still
/// in flight (they are back in about three, later on a slow card).
pub const DRAIN_FRAMES: u32 = 8;

/// Whether this render frame asks for another view of the live film.
///
/// `wanted`: the frames the film should hold by now (`Recording::period`);
/// `written`: the frames handed to the writer, repeats included;
/// `last_asked`: the `wanted` of the newest view still in flight (each view
/// is written for the periods from the one before it up to its own);
/// `asked`: the views in flight; `renderer`: the renderer's film frames in
/// flight (another film request counts against the bound too); `bound`:
/// the user's frames in flight (`in_flight`). A view is asked only for a
/// period no view written or in flight covers yet, so no period gets two
/// views and none is asked twice.
pub fn ask(wanted: u64, written: u64, last_asked: Option<u64>, asked: usize, renderer: u64, bound: u32) -> bool {
    let flying = renderer.max(u64::try_from(asked).unwrap_or(u64::MAX));
    let covered = last_asked.map_or(written, |last| last.max(written));
    flying < u64::from(in_flight(bound)) && wanted > covered
}

/// What a live film that is asked to end does this render frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtStop {
    /// Its frames in flight come back first: it ends at a later frame,
    /// asking for none.
    Wait,
    /// It ends now; what is still in flight is dropped.
    End,
}

/// A film with `asked` views in flight, whose writer `failed` or not, that
/// has waited `waited` render frames since it was asked to end.
pub fn at_stop(asked: usize, failed: bool, waited: u32) -> AtStop {
    if asked == 0 || failed || waited >= DRAIN_FRAMES {
        AtStop::End
    } else {
        AtStop::Wait
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: u32 = IN_FLIGHT_DEFAULT;

    #[test]
    fn a_view_is_asked_for_a_period_nothing_covers_yet() {
        // nothing in flight: upstream's rule, a frame owed
        assert!(ask(1, 0, None, 0, 0, D));
        assert!(!ask(5, 5, None, 0, 0, D));
        assert!(ask(6, 5, None, 0, 0, D));
        // one in flight for period 6: not again in period 6, yes in 7
        assert!(!ask(6, 5, Some(6), 1, 1, D));
        assert!(ask(7, 5, Some(6), 1, 1, D));
        // a stalled film (written behind): the newest view in flight counts
        assert!(!ask(9, 2, Some(9), 2, 2, D));
        assert!(ask(10, 2, Some(9), 2, 2, D));
        // a view in flight that a write has passed (a capped repeat) is no
        // reason to ask twice for the written period
        assert!(!ask(8, 8, Some(4), 1, 1, D));
        assert!(ask(9, 8, Some(4), 1, 1, D));
    }

    #[test]
    fn no_more_than_the_setting_are_in_flight() {
        // 1: upstream, one at a time
        assert!(ask(10, 5, None, 0, 0, 1));
        assert!(!ask(10, 5, Some(9), 1, 1, 1));
        // the default
        assert!(ask(10, 5, Some(9), 2, 2, D));
        assert!(!ask(10, 5, Some(9), 3, 3, D));
        // another request of the renderer counts too
        assert!(!ask(10, 5, Some(9), 2, 3, D));
        assert!(!ask(10, 5, None, usize::MAX, 0, D));
        // 10
        assert!(ask(20, 5, Some(19), 9, 9, 10));
        assert!(!ask(20, 5, Some(19), 10, 10, 10));
        // out of range values act as the nearest end
        assert!(!ask(10, 5, Some(9), 1, 1, 0));
        assert!(!ask(30, 5, Some(29), 10, 10, 99));
    }

    #[test]
    fn the_setting_is_brought_into_its_range() {
        assert_eq!(in_flight(0), 1);
        assert_eq!(in_flight(1), 1);
        assert_eq!(in_flight(D), 3);
        assert_eq!(in_flight(10), 10);
        assert_eq!(in_flight(11), 10);
        assert_eq!(in_flight(u32::MAX), 10);
    }

    /// Views asked and received over one second of a 24 fps film drawn at
    /// 30 render frames a second, each view back `lag` render frames after
    /// it is asked: (different views written, frames written).
    fn film(bound: u32, lag: u32) -> (u64, u64) {
        let fps = 24.0;
        let mut written = 0u64;
        let mut views = 0u64;
        let mut flight: std::collections::VecDeque<(u32, u64)> = Default::default();
        for frame in 0..30u32 {
            let t = f64::from(frame) / 30.0;
            while flight.front().is_some_and(|(at, _)| frame >= at + lag) {
                let Some((_, period)) = flight.pop_front() else { break };
                let repeat = period.saturating_sub(written);
                assert!(repeat >= 1, "a view with no period of its own");
                written += repeat;
                views += 1;
            }
            let wanted = (t * fps).floor() as u64 + 1;
            let last = flight.back().map(|(_, p)| *p);
            if ask(wanted, written, last, flight.len(), flight.len() as u64, bound) {
                flight.push_back((frame, wanted));
            }
            assert!(flight.len() as u64 <= u64::from(in_flight(bound)));
        }
        (views, written)
    }

    #[test]
    fn more_in_flight_give_more_different_views() {
        // back in three frames: one at a time gets a view every third
        // render frame, the film repeats the others
        let (one, _) = film(1, 3);
        assert!(one <= 10, "{one}");
        // the default takes a view for every period the second has
        let (three, written) = film(D, 3);
        assert_eq!(three, written);
        assert!(three >= 21, "{three}");
        // ten is no worse, and keeps up with a card that answers later
        assert!(film(10, 3).0 >= three);
        assert!(film(10, 6).0 > film(D, 6).0);
    }

    #[test]
    fn the_setting_is_clamped_on_load_and_defaults_when_missing() {
        for (saved, loaded) in [(0, 1), (1, 1), (7, 7), (10, 10), (11, 10), (u32::MAX, 10)] {
            let mut settings = crate::settings::Settings::default();
            settings.director_record.in_flight = saved;
            assert_eq!(settings.sanitized().director_record.in_flight, loaded);
        }
        let old: crate::recorder::Options = serde_json::from_str(r#"{"fps": 24}"#).expect("options without the field");
        assert_eq!(old.in_flight, IN_FLIGHT_DEFAULT);
        assert_eq!(crate::recorder::Options::default().in_flight, 3);
    }

    #[test]
    fn an_ended_film_waits_a_little_for_its_frames_in_flight() {
        assert_eq!(at_stop(2, false, 0), AtStop::Wait);
        assert_eq!(at_stop(1, false, DRAIN_FRAMES - 1), AtStop::Wait);
        assert_eq!(at_stop(0, false, 0), AtStop::End);
        // a failed writer takes nothing more
        assert_eq!(at_stop(3, true, 0), AtStop::End);
        // the wait is bounded
        assert_eq!(at_stop(1, false, DRAIN_FRAMES), AtStop::End);
    }
}
