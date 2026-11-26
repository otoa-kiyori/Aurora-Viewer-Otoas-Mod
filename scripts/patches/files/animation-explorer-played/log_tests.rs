// aurora-local-patch: animation-explorer-played-log-tests
//! The Animation explorer's "Played" column, per row: a pair of (animation,
//! source) that leaves the simulator's list has its newest start stopped, and
//! only a start that is the newest of its pair and not stopped can be
//! playing. Older rows of the pair stay in the log, with their stop time.

use super::animation_log::AnimationLog;
use std::collections::HashSet;
use std::time::{Duration, Instant};
use uuid::Uuid;

fn log() -> AnimationLog {
    AnimationLog::default()
}

#[test]
fn three_starts_of_one_pair_leave_one_playing_row_and_two_stopped_ones() {
    let mut log = log();
    let (anim, source) = (Uuid::from_u128(1), Uuid::from_u128(2));
    let t0 = Instant::now();
    let at = |s: u64| t0 + Duration::from_secs(s);
    log.record(anim, source, false, at(0));
    log.stop(anim, source, at(3));
    log.record(anim, source, false, at(10));
    log.stop(anim, source, at(12));
    log.record(anim, source, false, at(20));
    let entries: Vec<_> = log.entries().copied().collect();
    assert_eq!(entries.len(), 3, "every start stays a row");
    assert_eq!(entries[0].stopped, Some(at(3)));
    assert_eq!(entries[0].played_for(), Some(Duration::from_secs(3)));
    assert_eq!(entries[1].stopped, Some(at(12)));
    assert_eq!(entries[1].played_for(), Some(Duration::from_secs(2)));
    assert_eq!(entries[2].stopped, None);
    assert_eq!(entries[2].played_for(), None);
    assert_eq!(
        log.live_ids(),
        HashSet::from([entries[2].id]),
        "only the last start is the running one"
    );
}

#[test]
fn a_pair_that_stopped_has_no_playing_row_and_keeps_its_first_stop_time() {
    let mut log = log();
    let (anim, source) = (Uuid::from_u128(1), Uuid::from_u128(2));
    let t0 = Instant::now();
    log.record(anim, source, false, t0);
    log.stop(anim, source, t0 + Duration::from_secs(4));
    let generation = log.generation;
    log.stop(anim, source, t0 + Duration::from_secs(9));
    assert!(log.live_ids().is_empty());
    assert_eq!(log.entries().next().and_then(|e| e.stopped), Some(t0 + Duration::from_secs(4)));
    assert_eq!(log.generation, generation, "a second stop changes nothing");
    log.stop(Uuid::from_u128(7), source, t0);
    assert_eq!(log.generation, generation, "a pair that was never logged changes nothing");
}

#[test]
fn two_sources_of_one_animation_are_independent() {
    let mut log = log();
    let anim = Uuid::from_u128(1);
    let (ball, hud) = (Uuid::from_u128(2), Uuid::from_u128(3));
    let t0 = Instant::now();
    log.record(anim, ball, false, t0);
    log.record(anim, hud, false, t0);
    let ids: Vec<_> = log.entries().map(|e| e.id).collect();
    assert_eq!(log.live_ids(), ids.iter().copied().collect::<HashSet<_>>());
    log.stop(anim, ball, t0 + Duration::from_secs(5));
    assert_eq!(log.live_ids(), HashSet::from([ids[1]]), "the hud still plays it");
    log.record(anim, ball, false, t0 + Duration::from_secs(8));
    let newest = log.entries().last().map(|e| e.id);
    assert_eq!(log.live_ids(), HashSet::from([ids[1], newest.expect("a row")]));
    assert_eq!(log.entries().next().and_then(|e| e.played_for()), Some(Duration::from_secs(5)));
    assert_eq!(log.entries().nth(1).and_then(|e| e.stopped), None);
}
