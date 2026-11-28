// aurora-local-patch: ao-sit-cover-order-tests
//! The sit's cover (ANIM_AGENT_SIT_GENERIC) comes before the AO's sit: drawn
//! and sent first, and loaded with the set, so the AO's sit is the newest
//! motion of the two and wins the joints of its priority, as with
//! AOEngine::override and LLVOAvatar::startMotion in the Legacy Viewer.

use super::runtime::Runtime;
use super::tests::{fixture, frame, step};
use super::*;
use crate::world::PlayingAnimation;
use std::time::Instant;

#[test]
fn the_sit_cover_is_drawn_and_started_before_the_ao_sit() {
    let (_, set) = fixture();
    let sit = step(&set, State::Sitting, 0);
    let now = Instant::now();
    let raw = [PlayingAnimation::from_signal(State::Sitting.motion(), 1, now, None)];
    let mut f = frame(&raw);
    f.on_object = true;
    let mut r = Runtime::default();
    let (pose, cmd) = r.update(Some(&set), f, now, |_| true, |_| 0);
    assert_eq!(
        cmd,
        vec![(SIT_GENERIC, true), (sit, true)],
        "the cover's start first, the stock sit untouched"
    );
    assert!(!pose.iter().any(|a| a.id == State::Sitting.motion()), "the stock sit is not drawn");
    let at = |id: Uuid| pose.iter().position(|a| a.id == id);
    let (cover, ao) = (
        at(SIT_GENERIC).expect("the cover is drawn"),
        at(sit).expect("the AO's sit is drawn"),
    );
    assert!(cover < ao, "the cover before the AO's sit, which the blender takes as the newer");
}

#[test]
fn a_set_with_sitting_animations_prefetches_the_sit_cover_first() {
    let loads = |set: &Set| {
        let mut r = Runtime::default();
        let mut loaded = Vec::new();
        for _ in 0..256 {
            r.prefetch(Some(set), 1, |id| loaded.push(*id));
        }
        loaded
    };
    let (_, set) = fixture();
    let loaded = loads(&set);
    assert_eq!(loaded.first(), Some(&SIT_GENERIC), "loaded before any sit");
    assert_eq!(loaded.iter().filter(|id| **id == SIT_GENERIC).count(), 1, "once");
    assert!(loaded.contains(&step(&set, State::Sitting, 0)), "with the AO's sits");
    let mut standing_only = set.clone();
    standing_only.states.remove(&State::Sitting);
    assert!(!loads(&standing_only).contains(&SIT_GENERIC), "no sit, no cover");
}
