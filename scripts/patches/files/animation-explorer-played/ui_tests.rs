// aurora-local-patch: animation-explorer-played-ui-tests
use super::*;
use std::time::Duration;

fn row(playing: bool, started: Instant, stopped: Option<Instant>) -> Row {
    Row {
        id: 1,
        animation: Uuid::from_u128(1),
        source: Uuid::from_u128(2),
        name: "Pose ball".into(),
        own: false,
        playing,
        started,
        stopped,
        priority: None,
        animation_shown: false,
    }
}

#[test]
fn a_stopped_row_says_when_it_stopped_and_how_long_it_played() {
    let started = Instant::now();
    let r = row(false, started, Some(started + Duration::from_secs(3)));
    let text = played_text(&r, started + Duration::from_secs(15));
    assert_eq!(text, aurora_i18n::tr!("animation-explorer-stopped", ago = 12u64, played = 3u64));
    assert!(text.contains("12") && text.contains('3'), "{text}");
}

#[test]
fn the_playing_row_says_still_playing_and_an_unstamped_stopped_row_keeps_the_old_text() {
    let started = Instant::now();
    let now = started + Duration::from_secs(42);
    assert_eq!(
        played_text(&row(true, started, None), now),
        aurora_i18n::tr!("animation-explorer-still-playing")
    );
    assert_eq!(
        played_text(&row(false, started, None), now),
        aurora_i18n::tr!("animation-explorer-seconds-ago", seconds = 42u64)
    );
}

#[test]
fn the_tooltip_of_a_stopped_row_carries_the_whole_played_text() {
    let started = Instant::now();
    let now = started + Duration::from_secs(15);
    let stopped = row(false, started, Some(started + Duration::from_secs(3)));
    assert_eq!(stopped.tooltip_at(now), format!("Pose ball\n{}", played_text(&stopped, now)));
    assert_eq!(row(true, started, None).tooltip_at(now), "Pose ball");
}
