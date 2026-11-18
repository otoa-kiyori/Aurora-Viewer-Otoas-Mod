// aurora-local-patch: cef-site-profiles-module
//! Which Chromium profile a CEF browser gets, and what of it stays on disk
//! (local patch set). Two tiers:
//!
//! - **Kept**: the sites of Preferences › Network and cache › Keep cookies
//!   for these sites. A browser (in-world face, HUD, web window) whose
//!   first address is on one of them gets that site's own profile,
//!   `<cef>/cef-sites/<domain>/`, kept across restarts: its cookies and
//!   sign-ins survive, isolated from every other site. Nothing is pruned
//!   inside it: a site signs in through others (YouTube through
//!   google.com), and that has to keep working.
//! - **Normal**: every other browser gets a private profile
//!   (`private_profile.rs`, under `web-private/`), deleted once its plugin
//!   process has exited; whatever an exit or a crash cut short goes at the
//!   next start (`private_profile::sweep`).
//!
//! Why not a profile kept in memory only: MediaPluginCEF builds the profile
//! path itself. On `set_user_data_path` it sets `mRootCachePath` to
//! `<cache_path>cef_cache/<its pid>` and hands that to Dullahan as
//! `root_cache_path` at `init` (indra/media_plugins/cef/media_plugin_cef.cpp);
//! the pack's plugin carries the same `cef_cache` string. CEF's in-memory
//! mode needs an empty root path, which the plugin never sends: an empty
//! `cache_path` would only make the path relative to SLPlugin's working
//! folder (the media pack). So the normal tier deletes the profile instead.
//!
//! The pid in the path: the plugin names the profile after its own process,
//! whose id changes at every launch. The viewer starts SLPlugin itself and
//! knows that id before the plugin reads any message (they wait in
//! `MediaPlugin`'s queue until the plugin has loaded, `MediaPlugin::idle`),
//! so right after the launch the kept profile's folder is renamed
//! `cef_cache/<new pid>`, and the plugin opens it as its own.
//!
//! One browser per profile: Chromium cannot share a profile between
//! processes. A kept profile is held through a lock file (`in-use.lock`,
//! like the private profiles) from before the launch until its plugin
//! process has exited. A second browser of the same site meanwhile, in this
//! viewer or another, gets the normal tier, and the log says so. A profile
//! whose Chromium still runs (its `SingletonLock` names a live process: a
//! plugin of a viewer that has just quit) counts as in use too.
//!
//! The start's clean-up only reads `<cef>/cef_cache/` (`cef_cache.rs`) and
//! `web-private/`: `cef-sites/` is never part of it (tested below). Clearing
//! a site, or removing it from the list, moves its folder to
//! `cef-sites/.clearing-*` and deletes it in the background; the start
//! sweeps only those.

use aurora_media::{BrowserSettings, MediaPlugin};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Condvar, Mutex, RwLock};
use std::time::{Duration, Instant};

use super::private_profile::PrivateProfile;

/// The folder of the kept profiles, beside the plugin's `cef_cache`.
pub const SITES: &str = "cef-sites";
const LOCK_FILE: &str = "in-use.lock";
/// A site's folder being deleted (renamed first, so that a new browser of
/// that site starts on a fresh folder at once).
const CLEARING: &str = ".clearing-";
const CEF_PLUGIN: &str = "media_plugin_cef";
/// How long a plugin may take to exit once dropped (it has 14 s before it
/// is killed, `PluginProcess`), before its profile is let go anyway.
const EXIT_WAIT: Duration = Duration::from_secs(30);
/// The browser's helper processes (renderers, GPU) exit right after it.
const HELPERS_GRACE: Duration = Duration::from_millis(500);

/// One site of the list (saved in `settings.json`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct KeptSite {
    /// As [`normalize`] writes it: lower case, ASCII (punycode), no
    /// leading "www.".
    pub domain: String,
    /// On: the domain and all its subdomains. Off: the domain and its
    /// "www." only.
    pub subdomains: bool,
}

impl Default for KeptSite {
    fn default() -> Self {
        KeptSite {
            domain: String::new(),
            subdomains: true,
        }
    }
}

/// The list in use, read by every launch; set from the settings.
static LIST: RwLock<Vec<KeptSite>> = RwLock::new(Vec::new());
static NEXT: AtomicU32 = AtomicU32::new(0);
/// Profiles whose plugin is still exiting: (count, its change).
static EXITING: (Mutex<usize>, Condvar) = (Mutex::new(0), Condvar::new());

/// At start and after each change of the preferences. An entry that is not
/// a valid domain (a hand-edited file) is left out.
pub fn configure(sites: &[KeptSite]) {
    let mut clean: Vec<KeptSite> = Vec::with_capacity(sites.len());
    for site in sites {
        match normalize(&site.domain) {
            Some(domain) if !clean.iter().any(|s| s.domain == domain) => clean.push(KeptSite {
                domain,
                subdomains: site.subdomains,
            }),
            Some(_) => {}
            None => log::warn!("web profiles: an entry of the kept cookies list is not a domain, it is ignored"),
        }
    }
    if let Ok(mut list) = LIST.write() {
        *list = clean;
    }
}

/// The folder of the kept profiles in Aurora's cache.
pub fn sites_root() -> PathBuf {
    super::browser_cache(&crate::settings::cache_dir()).join(SITES)
}

/// What the user typed, as a domain the list can hold: lower case, no
/// scheme (http or https only), path, query or port, no leading "*." or
/// "www.", no trailing dot, an international name in its ASCII form
/// (punycode, as the browser writes it). None for anything else: an IP
/// address, a single label (`localhost`), a user name before "@", spaces.
pub fn normalize(input: &str) -> Option<String> {
    let mut text = input.trim().to_lowercase();
    if let Some((scheme, rest)) = text.split_once("://") {
        if scheme != "http" && scheme != "https" {
            return None;
        }
        text = rest.to_owned();
    }
    let host = text.split(['/', '?', '#', '\\']).next().unwrap_or("");
    // "secondlife.com@evil.example" goes to evil.example; IPv6 literals
    if host.contains(['@', '[', ']']) || host.chars().any(char::is_whitespace) {
        return None;
    }
    let host = match host.rsplit_once(':') {
        Some((name, port)) if !port.is_empty() && port.chars().all(|c| c.is_ascii_digit()) => name,
        Some(_) => return None,
        None => host,
    };
    let host = host.strip_prefix("*.").unwrap_or(host).trim_end_matches('.');
    if host.is_empty() {
        return None;
    }
    // the URL parser of the browser's side: IDNA to ASCII, IPv4 in any
    // notation recognised as such
    let parsed = reqwest::Url::parse(&format!("http://{host}/")).ok()?;
    let ascii = parsed.host_str()?.trim_end_matches('.');
    if ascii.parse::<std::net::Ipv4Addr>().is_ok() {
        return None;
    }
    let domain = ascii.strip_prefix("www.").unwrap_or(ascii);
    let labels: Vec<&str> = domain.split('.').collect();
    let valid = domain.len() <= 253
        && labels.len() >= 2
        && labels.iter().all(|l| {
            !l.is_empty()
                && l.len() <= 63
                && !l.starts_with('-')
                && !l.ends_with('-')
                && l.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
        })
        && !labels.last().is_some_and(|tld| tld.chars().all(|c| c.is_ascii_digit()));
    valid.then(|| domain.to_owned())
}

/// The host of a page address the way the browser reads it (lower case,
/// punycode, no trailing dot); None for another scheme than http(s), an IP
/// address or an address the browser would not open.
fn host_of(url: &str) -> Option<String> {
    let parsed = reqwest::Url::parse(url.trim()).ok()?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return None;
    }
    let host = parsed.host_str()?.trim_end_matches('.');
    if host.starts_with('[') || host.parse::<std::net::Ipv4Addr>().is_ok() {
        return None;
    }
    Some(host.to_ascii_lowercase())
}

/// Whether `site` covers `host`: the domain itself and its "www.", plus
/// every subdomain when the site's switch is on. A whole label must match:
/// `notsecondlife.com` and `secondlife.com.evil.example` are other sites.
pub fn covers(site: &KeptSite, host: &str) -> bool {
    let domain = site.domain.as_str();
    if domain.is_empty() {
        return false;
    }
    host == domain
        || host.strip_prefix("www.") == Some(domain)
        || (site.subdomains && host.strip_suffix(domain).is_some_and(|rest| rest.ends_with('.')))
}

/// The site of the list a page address belongs to: the most specific one
/// when several cover it (`id.secondlife.com` before `secondlife.com`).
pub fn site_for<'a>(sites: &'a [KeptSite], url: &str) -> Option<&'a KeptSite> {
    let host = host_of(url)?;
    sites.iter().filter(|s| covers(s, &host)).max_by_key(|s| s.domain.len())
}

/// The folder name of a site's profile: the domain itself, once checked to
/// be one plain name (never a path: no separator, no "." or "..", nothing
/// outside `a-z 0-9 . -`). None otherwise.
pub fn folder_name(domain: &str) -> Option<&str> {
    let plain = !domain.is_empty()
        && domain.len() <= 253
        && !domain.starts_with('.')
        && !domain.contains("..")
        && domain
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '.' || c == '-');
    let one = matches!(
        Path::new(domain).components().collect::<Vec<_>>().as_slice(),
        [std::path::Component::Normal(_)]
    );
    (plain && one).then_some(domain)
}

fn site_dir(root: &Path, domain: &str) -> Option<PathBuf> {
    folder_name(domain).map(|name| root.join(name))
}

#[cfg(unix)]
fn make_private_dir(dir: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new().recursive(true).mode(0o700).create(dir)
}

#[cfg(not(unix))]
fn make_private_dir(dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)
}

/// The profile folders the plugin made in a site's folder, newest first.
/// Links are never followed nor counted.
fn profiles_in(site: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(site.join(super::cef_cache::DIR)) else {
        return Vec::new();
    };
    let mut found: Vec<(std::time::SystemTime, PathBuf)> = entries
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .map(|e| {
            let modified = e.metadata().and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
            (modified, e.path())
        })
        .collect();
    found.sort_by_key(|f| std::cmp::Reverse(f.0));
    found.into_iter().map(|(_, p)| p).collect()
}

/// Whether the Chromium that last used a profile still runs: its
/// `SingletonLock` link reads `<host>-<pid>`.
fn chromium_runs(profile: &Path) -> bool {
    let Ok(target) = std::fs::read_link(profile.join("SingletonLock")) else {
        return false;
    };
    let pid = target
        .to_string_lossy()
        .rsplit_once('-')
        .and_then(|(_, pid)| pid.parse::<i64>().ok());
    pid.is_some_and(pid_runs)
}

#[cfg(unix)]
fn pid_runs(pid: i64) -> bool {
    libc::pid_t::try_from(pid).is_ok_and(super::process_runs)
}

#[cfg(not(unix))]
fn pid_runs(_pid: i64) -> bool {
    false
}

/// Why a site's profile cannot be used or cleared now.
#[derive(Debug)]
pub enum Busy {
    /// Another browser (of this viewer or another) has it.
    InUse,
    /// The domain cannot name a folder.
    BadName,
    Io(std::io::Error),
}

impl std::fmt::Display for Busy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Busy::InUse => f.write_str("in use"),
            Busy::BadName => f.write_str("not a folder name"),
            Busy::Io(e) => write!(f, "{e}"),
        }
    }
}

/// Take a site's lock: held while the lock file stays open.
fn lock_site(dir: &Path) -> Result<File, Busy> {
    make_private_dir(dir).map_err(Busy::Io)?;
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(dir.join(LOCK_FILE))
        .map_err(Busy::Io)?;
    if lock.try_lock().is_err() || profiles_in(dir).iter().any(|p| chromium_runs(p)) {
        return Err(Busy::InUse);
    }
    Ok(lock)
}

/// A kept site's profile, held for one browser.
#[derive(Debug)]
pub struct KeptProfile {
    dir: PathBuf,
    /// Held while open: the profile is ours (released by the drop).
    _lock: File,
}

impl KeptProfile {
    /// The profile of `domain` under `root`, created when it does not
    /// exist yet.
    pub fn acquire(root: &Path, domain: &str) -> Result<KeptProfile, Busy> {
        let dir = site_dir(root, domain).ok_or(Busy::BadName)?;
        let lock = lock_site(&dir)?;
        Ok(KeptProfile { dir, _lock: lock })
    }

    /// `base` with this profile as the plugin's cache path (the plugin
    /// appends `cef_cache/<pid>` to it, as it is); the log stays shared.
    pub fn settings(&self, base: &BrowserSettings) -> BrowserSettings {
        BrowserSettings {
            cache_path: format!("{}{}", self.dir.display(), std::path::MAIN_SEPARATOR),
            ..base.clone()
        }
    }

    /// Before the plugin reads its first message: the site's newest profile
    /// becomes `cef_cache/<pid>`, the plugin's own name for it. Older ones
    /// (a crash between two launches) are removed. Nothing to do on a first
    /// launch: the plugin creates it.
    pub fn bind(&self, pid: u32) -> std::io::Result<()> {
        let cache = self.dir.join(super::cef_cache::DIR);
        let target = cache.join(pid.to_string());
        let mut found = profiles_in(&self.dir).into_iter();
        let Some(newest) = found.next() else {
            return Ok(());
        };
        if newest != target {
            if target.exists() {
                // a folder of an earlier process of the same id, older
                std::fs::remove_dir_all(&target)?;
            }
            std::fs::rename(&newest, &target)?;
        }
        for older in found.filter(|p| *p != target) {
            if let Err(e) = std::fs::remove_dir_all(&older) {
                log::warn!("web profiles: an older copy of a kept profile was not removed ({e})");
            }
        }
        Ok(())
    }
}

/// The tier a browser of `url` gets: its site's kept profile when the site
/// is listed and its profile free, else a private profile under
/// `private_root` (None when even that fails: the shared cache, emptied at
/// the next start, as before this patch).
pub enum Tier {
    Kept(KeptProfile),
    Normal(Option<PrivateProfile>),
}

pub fn choose(sites: &[KeptSite], url: &str, sites_root: &Path, private_root: &Path) -> Tier {
    if let Some(site) = site_for(sites, url) {
        match KeptProfile::acquire(sites_root, &site.domain) {
            Ok(kept) => return Tier::Kept(kept),
            Err(Busy::InUse) => {
                log::info!("web profiles: the kept profile of a listed site is in use by another browser, this one keeps nothing")
            }
            Err(e) => log::warn!("web profiles: the kept profile of a listed site cannot be used ({e}), this one keeps nothing"),
        }
    }
    match PrivateProfile::create(private_root) {
        Ok(profile) => Tier::Normal(Some(profile)),
        Err(e) => {
            log::warn!("web profiles: no private profile ({e}), this browser uses the shared cache");
            Tier::Normal(None)
        }
    }
}

impl Tier {
    fn settings(&self, base: &BrowserSettings) -> BrowserSettings {
        match self {
            Tier::Kept(kept) => kept.settings(base),
            // the private profile, but the shared log as before
            Tier::Normal(Some(private)) => BrowserSettings {
                cef_log_file: base.cef_log_file.clone(),
                ..private.settings(base)
            },
            Tier::Normal(None) => base.clone(),
        }
    }
}

/// Holds a profile until the plugin process that uses it has exited, then
/// lets it go: a kept profile's lock is released, a private profile is
/// deleted (`PrivateProfile`'s drop).
struct Hold {
    tier: Option<Tier>,
    pid: Option<u32>,
}

impl Drop for Hold {
    fn drop(&mut self) {
        let Some(tier) = self.tier.take() else {
            return;
        };
        let pid = self.pid;
        if let Ok(mut n) = EXITING.0.lock() {
            *n += 1;
        }
        let waited = std::thread::Builder::new().name("aurora-web-profile".into()).spawn(move || {
            wait_exit(pid, EXIT_WAIT);
            drop(tier);
            exited();
        });
        if waited.is_err() {
            // no thread: let it go now (the start sweeps what stays)
            exited();
        }
    }
}

fn exited() {
    if let Ok(mut n) = EXITING.0.lock() {
        *n = n.saturating_sub(1);
        EXITING.1.notify_all();
    }
}

fn wait_exit(pid: Option<u32>, timeout: Duration) {
    let Some(pid) = pid else {
        return;
    };
    let deadline = Instant::now() + timeout;
    let mut was_running = false;
    while Instant::now() < deadline && pid_runs(i64::from(pid)) {
        was_running = true;
        std::thread::sleep(Duration::from_millis(200));
    }
    if was_running {
        std::thread::sleep(HELPERS_GRACE);
    }
}

/// At exit, once the browsers are dropped: wait (at most `timeout`) for
/// their plugins to exit, so that the private profiles are deleted now
/// and the kept ones are let go cleanly.
pub fn finish(timeout: Duration) {
    let deadline = Instant::now() + timeout;
    let Ok(mut n) = EXITING.0.lock() else {
        return;
    };
    while *n > 0 {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return;
        }
        match EXITING.1.wait_timeout(n, left) {
            Ok((next, _)) => n = next,
            Err(_) => return,
        }
    }
}

/// Launch a browser plugin on the profile its first address calls for;
/// `start` launches the plugin with the settings it is given. Another
/// plugin than CEF (video, music) is launched as it is.
pub fn launch(
    plugin: &str,
    first_url: &str,
    base: &BrowserSettings,
    start: impl FnOnce(&BrowserSettings) -> std::io::Result<MediaPlugin>,
) -> std::io::Result<MediaPlugin> {
    if plugin != CEF_PLUGIN {
        return start(base);
    }
    let sites = LIST.read().map(|l| l.clone()).unwrap_or_default();
    let private_root = crate::settings::cache_dir().join(super::private_profile::ROOT);
    let tier = choose(&sites, first_url, &sites_root(), &private_root);
    let mut plugin = start(&tier.settings(base))?;
    let pid = plugin.pid();
    if let Tier::Kept(kept) = &tier {
        // the plugin has read nothing yet (its messages are queued)
        match pid {
            Some(pid) => {
                if let Err(e) = kept.bind(pid) {
                    log::warn!("web profiles: the kept profile could not be handed to the browser ({e}), it starts empty");
                }
            }
            None => log::warn!("web profiles: no process id for the browser, its kept profile starts empty"),
        }
        log::info!("web profiles: a browser of a listed site uses its kept profile");
    }
    plugin.keep_until_dropped(Box::new(Hold { tier: Some(tier), pid }));
    Ok(plugin)
}

/// Whether a site's profile has something on disk (for Clear cookies).
pub fn has_profile(root: &Path, domain: &str) -> bool {
    site_dir(root, domain).is_some_and(|dir| dir.join(super::cef_cache::DIR).is_dir())
}

/// Delete everything a site's profile kept. Refused while a browser uses
/// it. The folder is renamed first (holding its lock), then deleted in the
/// background. False when there was nothing to delete.
pub fn clear(root: &Path, domain: &str) -> Result<bool, Busy> {
    let dir = site_dir(root, domain).ok_or(Busy::BadName)?;
    if !dir.is_dir() {
        return Ok(false);
    }
    let lock = lock_site(&dir)?;
    let trash = root.join(format!("{CLEARING}{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    std::fs::rename(&dir, &trash).map_err(Busy::Io)?;
    drop(lock);
    let removed = std::thread::Builder::new()
        .name("aurora-site-profile-clear".into())
        .spawn(move || remove_trash(&trash));
    if let Err(e) = removed {
        log::warn!("web profiles: a cleared profile is deleted at the next start ({e})");
    }
    Ok(true)
}

fn remove_trash(trash: &Path) {
    if let Err(e) = std::fs::remove_dir_all(trash)
        && e.kind() != std::io::ErrorKind::NotFound
    {
        log::warn!("web profiles: a cleared profile was not fully deleted ({e}), tried again at the next start");
    }
}

/// At start: finish the deletions an exit cut short. Only the folders
/// renamed by [`clear`]; never a kept profile.
pub fn sweep(root: &Path) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let named = entry.file_name().to_str().is_some_and(|n| n.starts_with(CLEARING));
        if named && entry.file_type().is_ok_and(|t| t.is_dir()) {
            remove_trash(&entry.path());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn site(domain: &str, subdomains: bool) -> KeptSite {
        KeptSite {
            domain: domain.into(),
            subdomains,
        }
    }

    fn root(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("aurora-site-profiles-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn typed_sites_become_plain_domains() {
        for (typed, domain) in [
            ("secondlife.com", "secondlife.com"),
            ("  SecondLife.COM ", "secondlife.com"),
            ("https://marketplace.secondlife.com/products?x=1", "marketplace.secondlife.com"),
            ("http://www.youtube.com:8080/watch#t", "youtube.com"),
            ("www.youtube.com", "youtube.com"),
            ("*.example.org", "example.org"),
            ("example.org.", "example.org"),
            ("bücher.example", "xn--bcher-kva.example"),
            ("xn--bcher-kva.example", "xn--bcher-kva.example"),
            ("a-b.co.uk", "a-b.co.uk"),
        ] {
            assert_eq!(normalize(typed).as_deref(), Some(domain), "{typed}");
        }
        for typed in [
            "",
            "   ",
            "localhost",
            "127.0.0.1",
            "http://192.168.1.10/",
            "0x7f.1",
            "[::1]",
            "http://[::1]:80/",
            "ftp://example.org",
            "javascript://example.org",
            "secondlife.com@evil.example",
            "exa mple.org",
            "example.org:port",
            "-bad.example",
            "bad-.example",
            "example.123",
            "..",
            "../etc",
            "a..b.example",
        ] {
            assert_eq!(normalize(typed), None, "{typed:?}");
        }
    }

    #[test]
    fn a_site_covers_its_own_hosts_only() {
        let all = site("secondlife.com", true);
        let exact = site("secondlife.com", false);
        for url in [
            "https://secondlife.com/",
            "https://www.secondlife.com/",
            "https://SECONDLIFE.com./x",
            "http://secondlife.com:8080/",
        ] {
            assert!(site_for(std::slice::from_ref(&all), url).is_some(), "{url}");
            assert!(site_for(std::slice::from_ref(&exact), url).is_some(), "{url}");
        }
        for url in ["https://marketplace.secondlife.com/", "https://id.secondlife.com/openid"] {
            assert!(site_for(std::slice::from_ref(&all), url).is_some(), "{url}");
            assert!(site_for(std::slice::from_ref(&exact), url).is_none(), "{url}");
        }
        for url in [
            "https://notsecondlife.com/",
            "https://secondlife.com.evil.example/",
            "https://evil.example/secondlife.com",
            "https://secondlife.com@evil.example/",
            "https://evil.example/?u=https://secondlife.com",
            "data:text/html,secondlife.com",
            "about:blank",
            "file:///secondlife.com",
            "https://127.0.0.1/",
            "https://[::1]/",
            "not a url",
            "",
        ] {
            assert!(site_for(std::slice::from_ref(&all), url).is_none(), "{url}");
        }
        // an international name: the browser's punycode against the list's
        let idn = KeptSite {
            domain: normalize("bücher.example").expect("idn"),
            subdomains: true,
        };
        assert!(site_for(std::slice::from_ref(&idn), "https://shop.bücher.example/").is_some());
        assert!(site_for(std::slice::from_ref(&idn), "https://xn--bcher-kva.example/").is_some());
        // an IP address never matches a domain of the list
        let ip_like = site("0.1", true);
        assert!(site_for(std::slice::from_ref(&ip_like), "http://127.0.0.1/").is_none());
        // the most specific of the sites that cover the host
        let both = [site("secondlife.com", true), site("id.secondlife.com", false)];
        assert_eq!(
            site_for(&both, "https://id.secondlife.com/").map(|s| s.domain.as_str()),
            Some("id.secondlife.com")
        );
        assert_eq!(
            site_for(&both, "https://marketplace.secondlife.com/").map(|s| s.domain.as_str()),
            Some("secondlife.com")
        );
        assert!(!covers(&site("", true), "example.org"));
    }

    #[test]
    fn a_folder_name_is_never_a_path() {
        for name in ["secondlife.com", "xn--bcher-kva.example", "a-b.co.uk"] {
            assert_eq!(folder_name(name), Some(name));
        }
        for name in [
            "",
            ".",
            "..",
            "../x",
            "a/b",
            "a\\b",
            "/etc",
            ".hidden",
            "a..b",
            "Upper.com",
            "a b.com",
            "c:",
            "a\0b",
        ] {
            assert_eq!(folder_name(name), None, "{name:?}");
        }
        let root = Path::new("/r");
        assert_eq!(site_dir(root, "example.org"), Some(root.join("example.org")));
        assert_eq!(site_dir(root, ".."), None);
    }

    #[test]
    fn configure_keeps_valid_unique_sites() {
        configure(&[
            site("WWW.Example.org", false),
            site("example.org", true),
            site("../etc", true),
            site("youtube.com", true),
        ]);
        let list = LIST.read().expect("list").clone();
        assert_eq!(list, vec![site("example.org", false), site("youtube.com", true)]);
        configure(&[]);
        assert!(LIST.read().expect("list").is_empty());
    }

    #[test]
    fn a_listed_site_gets_its_profile_and_a_second_browser_the_normal_tier() {
        let base = root("choose");
        let (sites_root, private_root) = (base.join(SITES), base.join("web-private"));
        let sites = [site("youtube.com", true)];
        let first = choose(&sites, "https://www.youtube.com/embed/x", &sites_root, &private_root);
        let Tier::Kept(kept) = &first else {
            panic!("the first browser gets the kept profile");
        };
        assert_eq!(kept.dir, sites_root.join("youtube.com"));
        let s = kept.settings(&BrowserSettings {
            cache_path: "/shared/cef/".into(),
            cef_log_file: "/shared/cef.log".into(),
            ..Default::default()
        });
        assert_eq!(Path::new(&s.cache_path), sites_root.join("youtube.com"));
        assert!(s.cache_path.ends_with(std::path::MAIN_SEPARATOR), "the plugin appends cef_cache");
        assert_eq!(s.cef_log_file, "/shared/cef.log");
        // the same site while the first holds it: private, deleted after use
        let second = choose(&sites, "https://m.youtube.com/", &sites_root, &private_root);
        let Tier::Normal(Some(private)) = &second else {
            panic!("a profile in use falls back to the normal tier");
        };
        assert!(private.dir().starts_with(&private_root));
        // another site: the normal tier
        assert!(matches!(
            choose(&sites, "https://example.org/", &sites_root, &private_root),
            Tier::Normal(Some(_))
        ));
        // once the first lets go, the next browser of the site gets it
        drop(first);
        assert!(matches!(
            choose(&sites, "https://youtube.com/", &sites_root, &private_root),
            Tier::Kept(_)
        ));
        drop(second);
        super::super::private_profile::finish(Duration::from_secs(10));
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn the_kept_profile_follows_the_plugin_id() {
        let base = root("bind");
        let kept = KeptProfile::acquire(&base, "example.org").expect("profile");
        // first launch: nothing to rename, the plugin creates it
        kept.bind(1111).expect("bind");
        let cache = base.join("example.org").join(super::super::cef_cache::DIR);
        assert!(!cache.join("1111").exists());
        // what that plugin left, then the next launch
        std::fs::create_dir_all(cache.join("1111").join("Default")).expect("profile");
        std::fs::write(cache.join("1111").join("Default").join("Cookies"), b"db").expect("file");
        kept.bind(2222).expect("bind");
        assert!(cache.join("2222").join("Default").join("Cookies").is_file());
        assert!(!cache.join("1111").exists());
        // an older copy beside the newest is removed; a link is never taken
        std::fs::create_dir_all(cache.join("old")).expect("old");
        std::fs::write(cache.join("old").join("x"), b"x").expect("file");
        let old = std::time::SystemTime::now() - Duration::from_secs(3600);
        File::open(cache.join("old")).and_then(|f| f.set_modified(old)).expect("mtime");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&base, cache.join("link")).expect("link");
        kept.bind(3333).expect("bind");
        assert!(cache.join("3333").join("Default").join("Cookies").is_file());
        assert!(!cache.join("old").exists() && !cache.join("2222").exists());
        #[cfg(unix)]
        assert!(cache.join("link").symlink_metadata().is_ok() && base.join("example.org").is_dir());
        drop(kept);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[cfg(unix)]
    #[test]
    fn a_profile_whose_chromium_still_runs_is_in_use() {
        let base = root("singleton");
        let profile = base.join("example.org").join(super::super::cef_cache::DIR).join("4444");
        std::fs::create_dir_all(&profile).expect("profile");
        // our own pid stands in for a Chromium still running
        std::os::unix::fs::symlink(format!("host-{}", std::process::id()), profile.join("SingletonLock")).expect("link");
        assert!(matches!(KeptProfile::acquire(&base, "example.org"), Err(Busy::InUse)));
        assert!(matches!(clear(&base, "example.org"), Err(Busy::InUse)));
        // a process far above any pid_max: gone
        std::fs::remove_file(profile.join("SingletonLock")).expect("unlink");
        std::os::unix::fs::symlink("host-2147483000", profile.join("SingletonLock")).expect("link");
        assert!(KeptProfile::acquire(&base, "example.org").is_ok());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn clearing_a_site_deletes_its_profile_unless_in_use() {
        let base = root("clear");
        assert!(matches!(clear(&base, "example.org"), Ok(false)), "nothing yet");
        assert!(matches!(clear(&base, "../x"), Err(Busy::BadName)));
        let kept = KeptProfile::acquire(&base, "example.org").expect("profile");
        let cookies = base.join("example.org").join(super::super::cef_cache::DIR).join("5555");
        std::fs::create_dir_all(&cookies).expect("profile");
        assert!(has_profile(&base, "example.org"));
        assert!(
            matches!(clear(&base, "example.org"), Err(Busy::InUse)),
            "refused while a browser has it"
        );
        assert!(cookies.is_dir());
        drop(kept);
        assert!(matches!(clear(&base, "example.org"), Ok(true)));
        assert!(!base.join("example.org").exists() && !has_profile(&base, "example.org"));
        // the background deletion, or the start's sweep after an exit
        let deadline = Instant::now() + Duration::from_secs(10);
        while std::fs::read_dir(&base).map(|d| d.count()).unwrap_or(0) > 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        let left = base.join(format!("{CLEARING}1-0"));
        std::fs::create_dir_all(left.join("cef_cache")).expect("left");
        let other = base.join("other.example");
        std::fs::create_dir_all(other.join("cef_cache")).expect("other");
        sweep(&base);
        assert!(!left.exists(), "an unfinished deletion is finished");
        assert!(other.join("cef_cache").is_dir(), "a kept profile is never swept");
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn the_start_clean_up_never_touches_the_kept_profiles() {
        // the browser cache as main.rs cleans it: cef_cache set aside and
        // purged, the misplaced folder of earlier builds removed
        let cache = root("startup");
        let cef = cache.join("cef");
        let kept = cef
            .join(SITES)
            .join("example.org")
            .join(super::super::cef_cache::DIR)
            .join("2147483001");
        std::fs::create_dir_all(kept.join("Default")).expect("kept");
        std::fs::write(kept.join("Default").join("Cookies"), b"db").expect("file");
        std::fs::create_dir_all(cef.join(super::super::cef_cache::DIR).join("2147483002")).expect("normal");
        super::super::cef_cache::set_aside(&cef);
        super::super::cef_cache::purge(&cef);
        #[cfg(unix)]
        super::super::remove_misplaced_cache(&cache);
        super::super::private_profile::sweep(&cache.join(super::super::private_profile::ROOT));
        sweep(&cef.join(SITES));
        assert!(kept.join("Default").join("Cookies").is_file());
        assert!(!cef.join(super::super::cef_cache::DIR).join("2147483002").exists());
        let _ = std::fs::remove_dir_all(&cache);
    }

    #[test]
    fn old_settings_load_and_new_ones_round_trip() {
        let s: KeptSite = serde_json::from_str(r#"{"domain":"example.org"}"#).expect("json");
        assert_eq!(s, site("example.org", true));
        let list = vec![site("youtube.com", true), site("example.org", false)];
        let text = serde_json::to_string(&list).expect("json");
        assert_eq!(serde_json::from_str::<Vec<KeptSite>>(&text).expect("json"), list);
    }
}
