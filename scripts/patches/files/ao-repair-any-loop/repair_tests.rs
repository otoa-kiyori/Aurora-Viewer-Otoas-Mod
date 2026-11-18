// aurora-local-patch: ao-repair-any-loop-tests
//! The AO's repair of an animation the simulator's list lost while the AO
//! still plays it (Runtime::reconcile_loops): a worn script stops the AO's
//! sit, and the AO starts it again at most every 2 s. Any loop is repaired,
//! one-frame poses included (Runtime::repairs); never a one-shot.

use super::runtime::Runtime;
use super::tests::{fixture, frame, step};
use super::*;
use crate::world::PlayingAnimation;
use std::time::{Duration, Instant};

fn clip(looping: bool, duration: f32) -> aurora_assets::Animation {
    aurora_assets::Animation {
        base_priority: 4,
        duration,
        emote_name: String::new(),
        loop_in: 0.0,
        loop_out: duration,
        looping,
        ends_after: None,
        ease_in: 0.0,
        ease_out: 0.0,
        hand_pose: 0,
        joints: Vec::new(),
    }
}

#[test]
fn a_lost_ao_sit_is_started_again_two_seconds_after_its_last_start() {
    let (_, set) = fixture();
    let sit = step(&set, State::Sitting, 0);
    let now = Instant::now();
    let at = |ms: u64| now + Duration::from_millis(ms);
    let stock = PlayingAnimation::from_signal(State::Sitting.motion(), 1, now, None);
    let seated = |raw: &[PlayingAnimation], t: Instant, r: &mut Runtime| {
        let mut f = frame(raw);
        f.on_object = true;
        r.update(Some(&set), f, t, |_| true, |_| 0)
    };
    let mut r = Runtime::default();
    let (_, cmd) = seated(&[stock], now, &mut r);
    assert!(cmd.contains(&(sit, true)), "the AO's sit starts");
    let echo = [
        stock,
        PlayingAnimation::from_signal(SIT_GENERIC, 2, at(100), None),
        PlayingAnimation::from_signal(sit, 3, at(100), None),
    ];
    assert!(seated(&echo, at(100), &mut r).1.is_empty());
    // a worn script stops the AO's sit: the list loses it, the AO keeps it
    let lost = [echo[0], echo[1]];
    let (pose, cmd) = seated(&lost, at(1500), &mut r);
    assert!(cmd.is_empty(), "the AO sends nothing of its own");
    assert!(pose.iter().any(|a| a.id == sit), "and still draws its sit");
    assert!(r.reconcile_loops(&lost, at(1600), |_| true).is_empty(), "too soon after its start");
    assert_eq!(
        r.reconcile_loops(&lost, at(2000), |_| true),
        vec![(sit, true)],
        "2 s after its last start"
    );
    assert!(r.reconcile_loops(&lost, at(2100), |_| true).is_empty(), "not again at once");
    // the cover, lost too, waits for the same 2 s from its own start
    let bare = [echo[0]];
    let one_frame = clip(true, 0.0);
    let looping = |id: &Uuid| *id == SIT_GENERIC && Runtime::repairs(&one_frame);
    assert_eq!(r.reconcile_loops(&bare, at(2200), looping), vec![(SIT_GENERIC, true)]);
}

#[test]
fn any_loop_is_repaired_one_frame_poses_included_never_a_one_shot() {
    let one_frame = clip(true, 0.0);
    assert!(one_frame.loop_range().is_none(), "the old test missed it");
    assert!(Runtime::repairs(&one_frame));
    assert!(Runtime::repairs(&clip(true, 2.0)));
    assert!(!Runtime::repairs(&clip(false, 2.0)), "a one-shot that ended is not replayed");
    assert!(!Runtime::repairs(&clip(false, 0.0)));
}
