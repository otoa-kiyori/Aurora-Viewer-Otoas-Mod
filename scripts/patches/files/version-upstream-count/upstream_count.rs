//! The displayed version counts upstream's commits after the release tag,
//! not the local commits of this build on top of them (pure helpers, shared
//! by the build script and its integration test).
// aurora-local-patch: version-upstream-count-module

use std::path::{Path, PathBuf};

/// The remote-tracking branch whose commits are upstream's.
pub const UPSTREAM_REF: &str = "refs/remotes/origin/main";

/// The commits of this checkout that upstream does not have: the output of
/// `git rev-list --count refs/remotes/origin/main..HEAD`. A missing output
/// (no such branch, a source archive without Git) is 0, which leaves the
/// version as `git describe` gave it.
#[allow(dead_code)]
pub fn local_commits(rev_list_count: Option<&str>) -> u64 {
    rev_list_count.and_then(|text| text.trim().parse().ok()).unwrap_or(0)
}

/// Splits `<tag>-<distance>-g<hash>` from the right (a tag can contain
/// dashes itself) into the tag, the distance and the hash.
fn split_describe(description: &str) -> Option<(&str, u64, &str)> {
    let mut parts = description.rsplitn(3, '-');
    let hash = parts.next()?;
    let distance = parts.next()?.parse::<u64>().ok()?;
    let tag = parts.next()?;
    let digits = hash.strip_prefix('g')?;
    (!tag.is_empty() && !digits.is_empty() && digits.bytes().all(|c| c.is_ascii_hexdigit())).then_some((tag, distance, hash))
}

/// `git describe --long` text with the local commits taken off its count:
/// `v1.2.3-10-gabcdef123` with 1 local commit is `v1.2.3-9-gabcdef123`. The
/// count never goes below 0, and text that is not a long description is
/// returned as it came.
#[allow(dead_code)]
pub fn without_local_commits(description: &str, local: u64) -> String {
    match split_describe(description) {
        Some((tag, distance, hash)) => format!("{tag}-{}-{hash}", distance.saturating_sub(local)),
        None => description.to_owned(),
    }
}

/// With the local commits taken off, a description can say 0: the local
/// commit sits directly on the release tag, and the version resolver would
/// then print the bare release number, as if this were the official build.
/// The display keeps its provenance instead: `<version>+0.g<commit>`.
#[allow(dead_code)]
pub fn mark_local_build(display: String, reduced: Option<&str>, local: u64, commit: Option<&str>) -> String {
    let on_tag = reduced.and_then(split_describe).is_some_and(|(_, distance, _)| distance == 0);
    match commit {
        Some(commit) if local > 0 && on_tag => {
            let separator = if display.contains('+') { '.' } else { '+' };
            format!("{display}{separator}0.g{commit}")
        }
        _ => display,
    }
}

/// The loose file of the upstream branch, when it exists, from the output of
/// `git rev-parse --path-format=absolute --git-common-dir`: Cargo must run
/// the build script again when a fetch moves it.
#[allow(dead_code)]
pub fn upstream_ref_path(common_dir: Option<&str>) -> Option<PathBuf> {
    let path = Path::new(common_dir?).join(UPSTREAM_REF);
    path.exists().then_some(path)
}
