// aurora-local-patch: version-upstream-count-tests
#[path = "../build_support.rs"]
mod build_support;
#[path = "../upstream_count.rs"]
mod upstream_count;

use build_support::resolve;
use upstream_count::{local_commits, mark_local_build, upstream_ref_path, without_local_commits};

#[test]
fn a_plain_tag_loses_the_local_commit() {
    assert_eq!(without_local_commits("v1.2.3-10-gabcdef123", 1), "v1.2.3-9-gabcdef123");
    assert_eq!(without_local_commits("v1.2.3-10-gabcdef123", 0), "v1.2.3-10-gabcdef123");
}

#[test]
fn a_tag_with_dashes_counts_from_the_end() {
    assert_eq!(
        without_local_commits("v0.9.14-beta.20261127.2-10-g8682d3a1e", 1),
        "v0.9.14-beta.20261127.2-9-g8682d3a1e"
    );
}

#[test]
fn the_count_never_goes_below_zero() {
    assert_eq!(without_local_commits("v1.2.3-1-gabcdef123", 1), "v1.2.3-0-gabcdef123");
    assert_eq!(without_local_commits("v1.2.3-2-gabcdef123", 5), "v1.2.3-0-gabcdef123");
}

#[test]
fn text_that_is_not_a_long_description_is_left_alone() {
    for text in [
        "",
        "v1.2.3",
        "v1.2.3-gabcdef123",
        "v1.2.3-x-gabcdef123",
        "v1.2.3-4-abcdef123",
        "v1.2.3-4-gzz",
        "-4-gabc",
    ] {
        assert_eq!(without_local_commits(text, 1), text);
    }
}

#[test]
fn the_local_count_comes_from_rev_list_or_is_zero() {
    assert_eq!(local_commits(Some("1")), 1);
    assert_eq!(local_commits(Some(" 3\n")), 3);
    assert_eq!(local_commits(None), 0);
    assert_eq!(local_commits(Some("fatal")), 0);
}

#[test]
fn the_displayed_count_is_upstream_s() {
    let reduced = without_local_commits("v0.9.14-beta.20261127.2-10-g8682d3a1e", 1);
    let version = resolve(None, Some(&reduced), 4242, Some("8682d3a1e")).unwrap();
    assert_eq!(version.display, "0.9.14-beta.20261127.2+9.g8682d3a1e");
    let shown = mark_local_build(version.display.clone(), Some(&reduced), 1, Some("8682d3a1e"));
    assert_eq!(shown, version.display);
}

#[test]
fn a_local_commit_on_a_tag_is_not_the_official_release() {
    let reduced = without_local_commits("v1.2.3-1-gabcdef123", 1);
    let version = resolve(None, Some(&reduced), 7, Some("abcdef123")).unwrap();
    assert_eq!(version.display, "1.2.3"); // what the resolver alone prints
    assert_eq!(
        mark_local_build(version.display, Some(&reduced), 1, Some("abcdef123")),
        "1.2.3+0.gabcdef123"
    );
    let pre = resolve(None, Some("v1.2.3-beta.1+build-0-gabcdef123"), 0, None).unwrap().display;
    let reduced = Some("v1.2.3-beta.1+build-0-gabcdef123");
    assert_eq!(
        mark_local_build(pre, reduced, 1, Some("abcdef123")),
        "1.2.3-beta.1+build.0.gabcdef123"
    );
}

#[test]
fn without_a_local_commit_nothing_is_marked() {
    let description = "v1.2.3-0-gabcdef123";
    assert_eq!(mark_local_build("1.2.3".into(), Some(description), 0, Some("abcdef123")), "1.2.3");
    assert_eq!(mark_local_build("1.2.3".into(), None, 1, Some("abcdef123")), "1.2.3");
    assert_eq!(mark_local_build("1.2.3".into(), Some(description), 1, None), "1.2.3");
}

#[test]
fn the_upstream_branch_is_watched_only_when_it_exists() {
    assert_eq!(upstream_ref_path(None), None);
    assert_eq!(upstream_ref_path(Some("/nonexistent-aurora-git-dir")), None);
}
