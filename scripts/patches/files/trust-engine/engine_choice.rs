// aurora-local-patch: trust-engine-choice
//! Which engine shows a web page (local patch set, 77-trust-engine):
//! TRust (`aurora-trust-host`, aurora-media `trust_host.rs`) or Chromium
//! (SLPlugin + the CEF plugin of the pack), on the three surfaces that show
//! pages: in-world faces (and parcel media), HUDs and the viewer's own web
//! windows (`ui/web_view.rs`).
//!
//! - Preferences › Media › Web pages: "TRust only", "Chromium only", or
//!   "TRust, then Chromium" (the default): TRust first, Chromium when TRust
//!   cannot show the page.
//! - What TRust is given in "TRust, then Chromium": the addresses its host
//!   serves ([`TRUST_SCHEMES`]: web pages, inline pages and local files);
//!   anything else goes straight to Chromium, with no TRust attempt and no
//!   blank face. "TRust only" sends every page to TRust: one it cannot load
//!   fails at once (`engine_unsupported`), it never hangs.
//! - Only web pages are concerned: video and music types keep their player
//!   (GStreamer on Linux), as `plugin_for_mime` chose it.
//! - "Always use Chromium for these sites" (Preferences › Media): the
//!   listed sites (a domain, with or without its subdomains, as the kept
//!   cookies list of `site_profiles.rs`) go to Chromium in both TRust
//!   settings.
//! - The fall back is automatic when the host does not start, crashes,
//!   stops answering (the viewer's PluginFailed: no heartbeat for 15 s), or
//!   refuses the page itself (`media_browser:engine_unsupported`: a response
//!   that is no page for TRust, a failure of its script engine or its
//!   raster). Never on an HTTP, DNS or TLS error: Chromium would fail as
//!   well. It is logged; a web window also shows a one-line notice, a face
//!   or a HUD only logs.
//! - A site that fell back, or that the user opened with Chromium ("Open
//!   with Chromium" on a face's controls and a web window's engine line),
//!   stays with Chromium for the rest of the session (no flapping when a
//!   face reloads), until the page moves to another site or "Try with
//!   TRust" is pressed. Only the address carries over to Chromium: scroll,
//!   forms and history start again.
//! - Cookies: a TRust page keeps nothing (memory only), except on a site of
//!   the kept cookies list (patch 76): its own TRust profile,
//!   `<cache>/trust-sites/<domain>/`, separate from Chromium's, which the
//!   host opens for the page ([`kept_profile`]). A site taken off the list
//!   loses it at the next change of the list.
//! - "Chromium only" keeps nothing of TRust: no host is started, running
//!   ones are dropped at the next frame (another plugin name for the page),
//!   and the hosts' run folders and log are deleted.
//!
//! Not a Legacy Viewer feature: it has only CEF. The Chromium path is
//! upstream's, unchanged.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use super::site_profiles::{self, KeptSite};

pub use aurora_media::trust_host::{CEF_PLUGIN, PLUGIN as TRUST_PLUGIN, is_browser};

/// The engine setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WebEngine {
    TrustOnly,
    ChromiumOnly,
    #[default]
    TrustThenChromium,
}

impl WebEngine {
    #[cfg(test)]
    pub const ALL: [WebEngine; 3] = [WebEngine::TrustOnly, WebEngine::ChromiumOnly, WebEngine::TrustThenChromium];

    fn key(self) -> &'static str {
        match self {
            WebEngine::TrustOnly => "trust_only",
            WebEngine::ChromiumOnly => "chromium_only",
            WebEngine::TrustThenChromium => "trust_then_chromium",
        }
    }

    /// A saved value or the `AURORA_WEB_ENGINE` override (`trust`,
    /// `chromium`, `trust-then-chromium`, or the saved names).
    pub fn parse(text: &str) -> Option<WebEngine> {
        match text.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "trust_only" | "trust" => Some(WebEngine::TrustOnly),
            "chromium_only" | "chromium" | "cef" => Some(WebEngine::ChromiumOnly),
            "trust_then_chromium" => Some(WebEngine::TrustThenChromium),
            _ => None,
        }
    }
}

/// An unknown value (a later build's) loads as the default rather than
/// failing the whole media settings.
impl<'de> serde::Deserialize<'de> for WebEngine {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let text = String::deserialize(d)?;
        Ok(WebEngine::parse(&text).unwrap_or_default())
    }
}

/// Where a page is shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Surface {
    /// A face of an object, or the parcel's media.
    Face,
    Hud,
    /// One of the viewer's web windows (profiles, Marketplace, search, the
    /// web browser).
    Window,
}

impl Surface {
    fn name(self) -> &'static str {
        match self {
            Surface::Face => "face",
            Surface::Hud => "HUD",
            Surface::Window => "web window",
        }
    }
}

/// The sites that fell back to Chromium this session.
#[derive(Debug, Default, Clone)]
pub struct Memory {
    sites: HashSet<String>,
}

impl Memory {
    pub fn remember(&mut self, url: &str) {
        self.sites.insert(site(url));
    }

    pub fn forget(&mut self, url: &str) {
        self.sites.remove(&site(url));
    }

    pub fn fell_back(&self, url: &str) -> bool {
        self.sites.contains(&site(url))
    }
}

/// A page's site: its host (without `www.`), or for an address without one
/// (`data:`, `about:`) its scheme and a hash of the whole address, so one
/// failing inline page does not send every other one to Chromium.
pub fn site(url: &str) -> String {
    let trimmed = url.trim();
    let lower = trimmed.to_ascii_lowercase();
    if let Some((scheme, rest)) = lower.split_once("://") {
        let host = rest.split(['/', '?', '#']).next().unwrap_or("");
        let host = host.rsplit_once('@').map_or(host, |(_, h)| h);
        let host = host.split(':').next().unwrap_or(host);
        if !host.is_empty() {
            return host.strip_prefix("www.").unwrap_or(host).to_string();
        }
        return format!("{scheme}:");
    }
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    trimmed.hash(&mut h);
    let scheme = lower.split(':').next().unwrap_or("");
    format!("{scheme}:{:016x}", h.finish())
}

/// What the TRust host serves: web pages (its own fetch and TLS), inline
/// pages and local files.
pub const TRUST_SCHEMES: [&str; 5] = ["http", "https", "data", "about", "file"];

/// Whether TRust can show `url` today ([`TRUST_SCHEMES`]); spaces and
/// control characters around it and the case do not count, as the browser
/// reads it.
pub fn trust_serves(url: &str) -> bool {
    let url = url.trim_matches(|c: char| c <= ' ');
    let Some((scheme, _)) = url.split_once(':') else {
        return false;
    };
    TRUST_SCHEMES.iter().any(|s| scheme.eq_ignore_ascii_case(s))
}

/// The plugin for a page: `mime_plugin` (what `plugin_for_mime` chose)
/// unless it is the browser, then the engine of the setting; a site of the
/// "Always use Chromium" list (`chromium_sites`) goes to Chromium. The
/// surface does not change the choice today (the owner chose one setting
/// for the three); it is part of the rule so that a later per-surface
/// setting changes this function only.
pub fn choose(
    mime_plugin: &'static str,
    url: &str,
    surface: Surface,
    engine: WebEngine,
    memory: &Memory,
    chromium_sites: &[KeptSite],
) -> &'static str {
    let _ = surface;
    if mime_plugin != CEF_PLUGIN {
        return mime_plugin;
    }
    if engine != WebEngine::ChromiumOnly && site_profiles::site_for(chromium_sites, url).is_some() {
        return CEF_PLUGIN;
    }
    match engine {
        WebEngine::ChromiumOnly => CEF_PLUGIN,
        WebEngine::TrustOnly => TRUST_PLUGIN,
        WebEngine::TrustThenChromium if !trust_serves(url) || memory.fell_back(url) => CEF_PLUGIN,
        WebEngine::TrustThenChromium => TRUST_PLUGIN,
    }
}

/// What a page's engine button offers (a face's controls, a web window's
/// engine line).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Switch {
    /// TRust shows it: "Open with Chromium".
    ToChromium,
    /// Chromium shows it after a fall back or the user's choice: "Try with
    /// TRust".
    ToTrust,
}

/// The engine button for a page `plugin` shows; None when the setting or
/// the site list chose Chromium (Preferences decide those).
pub fn switch(plugin: &str, url: &str, engine: WebEngine, memory: &Memory) -> Option<Switch> {
    if plugin == TRUST_PLUGIN {
        return Some(Switch::ToChromium);
    }
    (plugin == CEF_PLUGIN && engine == WebEngine::TrustThenChromium && trust_serves(url) && memory.fell_back(url))
        .then_some(Switch::ToTrust)
}

/// Whether a failure of `plugin` sends the page to Chromium.
pub fn falls_back(plugin: &str, engine: WebEngine) -> bool {
    plugin == TRUST_PLUGIN && engine == WebEngine::TrustThenChromium
}

// ------------------------------------------------------- the session's state

struct State {
    engine: WebEngine,
    memory: Memory,
    /// "Always use Chromium for these sites".
    chromium_sites: Vec<KeptSite>,
    /// The kept cookies list (patch 76), for TRust's kept profiles.
    kept_sites: Vec<KeptSite>,
    /// `<cache>/trust-run`, `<cache>/trust-sites` and the host's log, for
    /// the sweeps.
    run_root: Option<PathBuf>,
    sites_root: Option<PathBuf>,
    log_file: Option<PathBuf>,
}

static STATE: std::sync::LazyLock<Mutex<State>> = std::sync::LazyLock::new(|| {
    Mutex::new(State {
        engine: WebEngine::TrustThenChromium,
        memory: Memory::default(),
        chromium_sites: Vec::new(),
        kept_sites: Vec::new(),
        run_root: None,
        sites_root: None,
        log_file: None,
    })
});

/// TRust's kept profiles: `<cache>/trust-sites/<domain>/`.
pub const TRUST_SITES: &str = "trust-sites";

/// The `AURORA_WEB_ENGINE` override (tests and demo runs), if set.
fn override_engine() -> Option<WebEngine> {
    let v = std::env::var("AURORA_WEB_ENGINE").ok()?;
    let e = WebEngine::parse(&v);
    if e.is_none() {
        log::warn!("media: AURORA_WEB_ENGINE={v} is not trust, chromium or trust-then-chromium: ignored");
    }
    e
}

/// At start, before the first page: where the hosts' files go, and the
/// stale run folders removed (viewers that no longer run).
pub fn init_paths(cache_dir: &Path, viewer_log: &Path) {
    let run_root = cache_dir.join("trust-run");
    let sites_root = cache_dir.join(TRUST_SITES);
    let log_file = viewer_log.with_extension("trust-host.log");
    let default = directories::ProjectDirs::from("org", "Aurora", "AuroraViewer")
        .map(|d| d.data_local_dir().join("trust-host").join(aurora_media::trust_host::PROGRAM));
    aurora_media::trust_host::configure(|c| {
        c.run_root = Some(run_root.clone());
        c.log_file = Some(log_file.clone());
        c.default = default;
    });
    if let Ok(mut s) = STATE.lock() {
        s.run_root = Some(run_root.clone());
        s.sites_root = Some(sites_root);
        s.log_file = Some(log_file);
    }
    let removed = aurora_media::trust_host::sweep(&run_root, false);
    if removed > 0 {
        log::info!("media: {removed} TRust run folder(s) left by an earlier session removed");
    }
}

/// The settings, at start and after each change of the preferences:
/// the media settings and the kept cookies list (patch 76).
pub fn configure(media: &super::MediaSettings, kept_sites: &[KeptSite]) {
    let engine = override_engine().unwrap_or(media.web_engine);
    let explicit = media.trust_host.trim();
    aurora_media::trust_host::configure(|c| c.explicit = (!explicit.is_empty()).then(|| PathBuf::from(explicit)));
    let Ok(mut s) = STATE.lock() else {
        return;
    };
    let changed = s.engine != engine;
    s.engine = engine;
    s.chromium_sites = clean_list(&media.web_engine_chromium_sites);
    let kept = clean_list(kept_sites);
    if kept != s.kept_sites || changed {
        s.kept_sites = kept.clone();
        // the TRust profiles of sites taken off the list (or all of them
        // with "Chromium only"), never one a page holds
        if let Some(root) = s.sites_root.clone() {
            let keep: Vec<String> = if engine == WebEngine::ChromiumOnly {
                Vec::new()
            } else {
                kept.iter().map(|k| k.domain.clone()).collect()
            };
            let _ = std::thread::Builder::new()
                .name("aurora-trust-sites".into())
                .spawn(move || sweep_sites(&root, &keep));
        }
    }
    if changed {
        log::info!("media: web pages: {}", engine.key());
    }
    if engine == WebEngine::ChromiumOnly {
        // nothing of TRust stays on disk; the hosts still running are
        // dropped at the next frame, their supervisors delete their own
        // folders: wait a little for them, then delete what is left
        let (root, log) = (s.run_root.clone(), s.log_file.clone());
        let _ = std::thread::Builder::new().name("aurora-trust-sweep".into()).spawn(move || {
            if changed {
                std::thread::sleep(std::time::Duration::from_secs(3));
            }
            if let Some(root) = root {
                aurora_media::trust_host::sweep(&root, true);
            }
            if let Some(log) = log {
                let _ = std::fs::remove_file(log);
            }
        });
    }
}

/// A list as the preferences hold it, normalized (`site_profiles::normalize`),
/// without duplicates or entries that are not domains.
fn clean_list(sites: &[KeptSite]) -> Vec<KeptSite> {
    let mut clean: Vec<KeptSite> = Vec::with_capacity(sites.len());
    for site in sites {
        if let Some(domain) = site_profiles::normalize(&site.domain)
            && !clean.iter().any(|s| s.domain == domain)
        {
            clean.push(KeptSite {
                domain,
                subdomains: site.subdomains,
            });
        }
    }
    clean
}

/// Delete the TRust profiles under `root` not in `keep`, skipping one a
/// host holds (its `.lock`, aurora-trust-host profile.rs). Returns how many
/// were removed.
pub fn sweep_sites(root: &Path, keep: &[String]) -> usize {
    let Ok(entries) = std::fs::read_dir(root) else {
        return 0;
    };
    let mut removed = 0;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if keep.contains(&name) {
            continue;
        }
        let path = entry.path();
        let is_dir = entry.file_type().is_ok_and(|t| t.is_dir());
        if is_dir && held(&path.join(".lock")) {
            continue;
        }
        let gone = if is_dir {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        };
        match gone {
            Ok(()) => removed += 1,
            Err(e) => log::warn!("media: a TRust site profile was not removed ({e})"),
        }
    }
    if removed > 0 {
        log::info!("media: {removed} TRust site profile(s) of sites no longer kept removed");
    }
    removed
}

/// Whether a running host holds this lock file.
#[cfg(unix)]
fn held(lock: &Path) -> bool {
    use std::os::fd::AsRawFd;
    let Ok(file) = std::fs::OpenOptions::new().write(true).open(lock) else {
        return false;
    };
    // SAFETY: flock on a descriptor we own; released when it closes
    let r = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    r != 0
}

#[cfg(not(unix))]
fn held(_lock: &Path) -> bool {
    true
}

/// The kept TRust profile for a page whose first address is `url`: its
/// folder and its domain, when the site is on the kept cookies list.
pub fn kept_profile(url: &str) -> Option<(PathBuf, String)> {
    let s = STATE.lock().ok()?;
    let site = site_profiles::site_for(&s.kept_sites, url)?;
    let name = site_profiles::folder_name(&site.domain)?;
    Some((s.sites_root.as_ref()?.join(name), site.domain.clone()))
}

/// Launch a page's plugin; a TRust page of a kept site gets its profile.
pub fn launch<T>(plugin: &str, url: &str, start: impl FnOnce() -> T) -> T {
    if plugin != TRUST_PLUGIN {
        return start();
    }
    let kept = kept_profile(url);
    if kept.is_some() {
        log::info!("media: a TRust page of a listed site uses its kept profile");
    }
    aurora_media::trust_host::with_kept_site(kept, start)
}

/// [`choose`] with the session's setting, memory and site list.
pub fn choose_now(mime_plugin: &'static str, url: &str, surface: Surface) -> &'static str {
    match STATE.lock() {
        Ok(s) => choose(mime_plugin, url, surface, s.engine, &s.memory, &s.chromium_sites),
        Err(_) => mime_plugin,
    }
}

/// [`switch`] with the session's setting and memory.
pub fn switch_now(plugin: &str, url: &str) -> Option<Switch> {
    STATE.lock().ok().and_then(|s| switch(plugin, url, s.engine, &s.memory))
}

/// "Open with Chromium": the page's site goes to Chromium for the session.
pub fn open_with_chromium(url: &str) {
    if let Ok(mut s) = STATE.lock() {
        s.memory.remember(url);
        log::info!("media: {} opened with Chromium for the rest of the session", site(url));
    }
}

/// `plugin` failed for `url` (`reason`: the PluginFailed text or the launch
/// error). True when the page goes to Chromium (remembered for its site);
/// false when the failure stands as it is.
pub fn fall_back(plugin: &str, url: &str, surface: Surface, reason: &str) -> bool {
    let Ok(mut s) = STATE.lock() else {
        return false;
    };
    if !falls_back(plugin, s.engine) {
        return false;
    }
    s.memory.remember(url);
    log::info!(
        "media: TRust could not show a page on a {} ({reason}): shown with Chromium for the rest of the session ({})",
        surface.name(),
        site(url)
    );
    true
}

/// "Try with TRust": the site's fall back is forgotten.
pub fn retry(url: &str) {
    if let Ok(mut s) = STATE.lock() {
        s.memory.forget(url);
        log::info!("media: trying TRust again for {}", site(url));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: &str = "data:text/html,<p>board";
    const WEB: &str = "https://example.org/board?x=1";

    #[test]
    fn every_setting_on_every_surface_with_and_without_a_fall_back() {
        let surfaces = [Surface::Face, Surface::Hud, Surface::Window];
        let fresh = Memory::default();
        let mut fell = Memory::default();
        fell.remember(PAGE);
        fell.remember(WEB);
        for surface in surfaces {
            for (memory, label) in [(&fresh, "fresh"), (&fell, "fell back")] {
                for url in [PAGE, WEB] {
                    let pick = |engine| choose(CEF_PLUGIN, url, surface, engine, memory, &[]);
                    assert_eq!(pick(WebEngine::ChromiumOnly), CEF_PLUGIN, "{surface:?} {label} {url}");
                    assert_eq!(
                        pick(WebEngine::TrustOnly),
                        TRUST_PLUGIN,
                        "{surface:?} {label} {url}: TRust only never falls back"
                    );
                    let expected = if trust_serves(url) && !memory.fell_back(url) {
                        TRUST_PLUGIN
                    } else {
                        CEF_PLUGIN
                    };
                    assert_eq!(pick(WebEngine::TrustThenChromium), expected, "{surface:?} {label} {url}");
                }
            }
        }
    }

    #[test]
    fn trust_then_chromium_sends_trust_the_web_and_local_pages() {
        for url in [
            "https://marketplace.secondlife.com/",
            "http://example.org/",
            "HTTPS://example.org/",
            "data:text/html,<p>hi",
            " DATA:text/html;base64,PHA+",
            "about:blank",
            "file:///srv/site/index.html",
        ] {
            assert!(trust_serves(url), "{url}");
            for surface in [Surface::Face, Surface::Hud, Surface::Window] {
                assert_eq!(
                    choose(CEF_PLUGIN, url, surface, WebEngine::TrustThenChromium, &Memory::default(), &[]),
                    TRUST_PLUGIN,
                    "{url}"
                );
            }
        }
        for url in [
            "ftp://example.org/",
            "secondlife:///app/agent/x/about",
            "javascript:alert(1)",
            "example.org/no-scheme",
            "",
        ] {
            assert!(!trust_serves(url), "{url}");
            for surface in [Surface::Face, Surface::Hud, Surface::Window] {
                let m = Memory::default();
                assert_eq!(
                    choose(CEF_PLUGIN, url, surface, WebEngine::TrustThenChromium, &m, &[]),
                    CEF_PLUGIN,
                    "{url}"
                );
                // the owner's explicit choice: TRust, which refuses it cleanly
                assert_eq!(
                    choose(CEF_PLUGIN, url, surface, WebEngine::TrustOnly, &m, &[]),
                    TRUST_PLUGIN,
                    "{url}"
                );
            }
        }
    }

    #[test]
    fn listed_sites_always_go_to_chromium_unless_nothing_does() {
        let list = clean_list(&[
            KeptSite {
                domain: "Marketplace.SecondLife.com".into(),
                subdomains: false,
            },
            KeptSite {
                domain: "example.org".into(),
                subdomains: true,
            },
            KeptSite {
                domain: "not a domain".into(),
                subdomains: true,
            },
        ]);
        assert_eq!(list.len(), 2);
        let m = Memory::default();
        for engine in [WebEngine::TrustThenChromium, WebEngine::TrustOnly] {
            let pick = |url| choose(CEF_PLUGIN, url, Surface::Window, engine, &m, &list);
            assert_eq!(pick("https://marketplace.secondlife.com/p/1"), CEF_PLUGIN);
            assert_eq!(pick("https://shop.example.org/"), CEF_PLUGIN);
            // the domain's subdomains only when its switch is on
            assert_eq!(pick("https://x.marketplace.secondlife.com/"), TRUST_PLUGIN);
            assert_eq!(pick("https://secondlife.com/"), TRUST_PLUGIN);
            assert_eq!(pick("https://notexample.org/"), TRUST_PLUGIN);
        }
        assert_eq!(
            choose(
                CEF_PLUGIN,
                "https://example.org/",
                Surface::Face,
                WebEngine::ChromiumOnly,
                &m,
                &list
            ),
            CEF_PLUGIN
        );
        // video stays with its player
        assert_eq!(
            choose(
                "media_plugin_gstreamer",
                "https://example.org/a.mp4",
                Surface::Face,
                WebEngine::TrustOnly,
                &m,
                &list
            ),
            "media_plugin_gstreamer"
        );
    }

    #[test]
    fn the_engine_button_offers_the_other_engine_when_the_user_may_switch() {
        let url = "https://example.org/a";
        let mut m = Memory::default();
        assert_eq!(
            switch(TRUST_PLUGIN, url, WebEngine::TrustThenChromium, &m),
            Some(Switch::ToChromium)
        );
        assert_eq!(switch(TRUST_PLUGIN, url, WebEngine::TrustOnly, &m), Some(Switch::ToChromium));
        // Chromium by the setting: nothing to offer
        assert_eq!(switch(CEF_PLUGIN, url, WebEngine::TrustThenChromium, &m), None);
        assert_eq!(switch(CEF_PLUGIN, url, WebEngine::ChromiumOnly, &m), None);
        // after "Open with Chromium" or a fall back: Try with TRust
        m.remember(url);
        assert_eq!(switch(CEF_PLUGIN, url, WebEngine::TrustThenChromium, &m), Some(Switch::ToTrust));
        assert_eq!(
            choose(CEF_PLUGIN, url, Surface::Face, WebEngine::TrustThenChromium, &m, &[]),
            CEF_PLUGIN
        );
        m.forget(url);
        assert_eq!(
            choose(CEF_PLUGIN, url, Surface::Face, WebEngine::TrustThenChromium, &m, &[]),
            TRUST_PLUGIN
        );
        // video has no engine button
        assert_eq!(switch("media_plugin_gstreamer", url, WebEngine::TrustThenChromium, &m), None);
    }

    #[cfg(unix)]
    #[test]
    fn site_profiles_of_sites_no_longer_kept_go_unless_a_page_holds_them() {
        use std::os::fd::AsRawFd;
        let root = std::env::temp_dir().join(format!("aurora-trust-sites-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        for name in ["kept.org", "dropped.org", "held.org"] {
            std::fs::create_dir_all(root.join(name).join("trust")).expect("fixture");
        }
        let lock = std::fs::File::create(root.join("held.org").join(".lock")).expect("lock");
        // SAFETY: flock on the test's own descriptor
        assert_eq!(unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) }, 0);
        assert_eq!(sweep_sites(&root, &["kept.org".to_string()]), 1);
        assert!(root.join("kept.org").is_dir());
        assert!(!root.join("dropped.org").exists());
        assert!(root.join("held.org").is_dir());
        drop(lock);
        assert_eq!(sweep_sites(&root, &[]), 2);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn video_and_music_keep_their_player_whatever_the_setting() {
        for engine in WebEngine::ALL {
            for surface in [Surface::Face, Surface::Hud, Surface::Window] {
                for plugin in ["media_plugin_gstreamer", "media_plugin_libvlc"] {
                    assert_eq!(
                        choose(plugin, "https://x.org/a.mp4", surface, engine, &Memory::default(), &[]),
                        plugin
                    );
                }
            }
        }
    }

    #[test]
    fn only_trust_then_chromium_falls_back_and_only_from_trust() {
        assert!(falls_back(TRUST_PLUGIN, WebEngine::TrustThenChromium));
        assert!(!falls_back(TRUST_PLUGIN, WebEngine::TrustOnly));
        assert!(!falls_back(TRUST_PLUGIN, WebEngine::ChromiumOnly));
        assert!(!falls_back(CEF_PLUGIN, WebEngine::TrustThenChromium));
        assert!(!falls_back("media_plugin_gstreamer", WebEngine::TrustThenChromium));
    }

    #[test]
    fn the_memory_is_per_site_and_forgotten_on_retry() {
        let mut m = Memory::default();
        m.remember("https://www.Example.org/a");
        assert!(m.fell_back("http://example.org/other/page"));
        assert!(m.fell_back("https://user@example.org:8443/x"));
        assert!(!m.fell_back("https://shop.example.org/"));
        assert!(!m.fell_back("https://example.net/"));
        // another inline page goes to TRust: no flapping on one page
        m.remember("data:text/html,<p>one");
        assert_eq!(
            choose(
                CEF_PLUGIN,
                "data:text/html,<p>one",
                Surface::Face,
                WebEngine::TrustThenChromium,
                &m,
                &[]
            ),
            CEF_PLUGIN
        );
        assert_eq!(
            choose(
                CEF_PLUGIN,
                "data:text/html,<p>two",
                Surface::Face,
                WebEngine::TrustThenChromium,
                &m,
                &[]
            ),
            TRUST_PLUGIN
        );
        m.forget("https://example.org/");
        assert!(!m.fell_back("https://example.org/a"));
    }

    #[test]
    fn inline_pages_are_their_own_site() {
        let a = "data:text/html,<p>one";
        let b = "data:text/html,<p>two";
        assert_ne!(site(a), site(b));
        assert_eq!(site(a), site(a));
        assert!(site(a).starts_with("data:"));
        assert_eq!(site("about:blank"), site("about:blank"));
        assert_eq!(site("file:///tmp/x.html"), "file:");
    }

    #[test]
    fn the_setting_loads_from_old_new_and_unknown_files() {
        #[derive(serde::Deserialize)]
        struct S {
            #[serde(default)]
            web_engine: WebEngine,
        }
        let parse = |json: &str| serde_json::from_str::<S>(json).expect("loads").web_engine;
        assert_eq!(parse("{}"), WebEngine::TrustThenChromium);
        assert_eq!(parse(r#"{"web_engine":"chromium_only"}"#), WebEngine::ChromiumOnly);
        assert_eq!(parse(r#"{"web_engine":"trust_only"}"#), WebEngine::TrustOnly);
        assert_eq!(parse(r#"{"web_engine":"servo_someday"}"#), WebEngine::TrustThenChromium);
        for e in WebEngine::ALL {
            let saved = serde_json::to_string(&e).expect("saves");
            assert_eq!(WebEngine::parse(saved.trim_matches('"')), Some(e));
        }
        assert_eq!(WebEngine::parse("trust-then-chromium"), Some(WebEngine::TrustThenChromium));
        assert_eq!(WebEngine::parse("CHROMIUM"), Some(WebEngine::ChromiumOnly));
        assert_eq!(WebEngine::parse("x"), None);
    }

    #[test]
    fn the_media_settings_default_to_trust_then_chromium() {
        let m = super::super::MediaSettings::default();
        assert_eq!(m.web_engine, WebEngine::TrustThenChromium);
        assert!(m.trust_host.is_empty());
        // a settings file from before the patch
        let old: super::super::MediaSettings = serde_json::from_str(r#"{"enabled":true,"max_instances":4}"#).expect("loads");
        assert_eq!(old.web_engine, WebEngine::TrustThenChromium);
        assert_eq!(old.max_instances, 4);
    }
}
