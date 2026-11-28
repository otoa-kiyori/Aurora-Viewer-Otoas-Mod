// aurora-local-patch: trust-engine-media-module
//! TRust beside CEF (local patch set, 77-trust-engine): the second browser
//! plugin, `media_plugin_trust`, is not a library SLPlugin loads but a
//! program of its own, `aurora-trust-host` (a separate local repository),
//! one process per page. It speaks this crate's protocol from the other end
//! (`process.rs`): the same messages, the same shared memory, over a Unix
//! socket pair handed to it as descriptor 3 instead of a loopback TCP port.
//! The host runs its page in bubblewrap; a socket pair needs no loopback
//! port, so the firewall sees only the page's own web traffic.
//!
//! The viewer configures the host's places (`configure`): the program
//! (an explicit path from the preferences, else its default under the
//! viewer's data folder; `AURORA_TRUST_HOST` wins, for tests), the folder
//! of the pages' run folders (`<cache>/trust-run/<viewer pid>-<n>/`, made
//! by the host and deleted when it exits) and the host's log file.

use crate::process::{PluginPaths, PluginProcess};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

/// The plugin name of a page shown by TRust.
pub const PLUGIN: &str = "media_plugin_trust";
/// The CEF plugin's name.
pub const CEF_PLUGIN: &str = "media_plugin_cef";
/// The test override of the host's path.
pub const ENV: &str = "AURORA_TRUST_HOST";
/// The program's file name.
pub const PROGRAM: &str = "aurora-trust-host";
/// The host's own refusal (`media_browser:engine_unsupported`), as the
/// text of the PluginFailed it becomes.
pub const UNSUPPORTED: &str = "engine unsupported: ";

/// A web page's plugin, whichever engine shows it (where upstream compared
/// the name with `media_plugin_cef`).
pub fn is_browser(plugin: &str) -> bool {
    plugin == CEF_PLUGIN || plugin == PLUGIN
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct HostConfig {
    /// The path chosen in the preferences (None: automatic).
    pub explicit: Option<PathBuf>,
    /// The automatic path: `<data>/trust-host/aurora-trust-host`.
    pub default: Option<PathBuf>,
    /// `<cache>/trust-run`.
    pub run_root: Option<PathBuf>,
    pub log_file: Option<PathBuf>,
}

static CONFIG: Mutex<HostConfig> = Mutex::new(HostConfig {
    explicit: None,
    default: None,
    run_root: None,
    log_file: None,
});
static RUN_NUMBER: AtomicU32 = AtomicU32::new(0);

pub fn configure(change: impl FnOnce(&mut HostConfig)) {
    if let Ok(mut c) = CONFIG.lock() {
        change(&mut c);
    }
}

pub fn config() -> HostConfig {
    CONFIG.lock().map(|c| c.clone()).unwrap_or_default()
}

/// The host's path: the override, then the chosen path, then the default.
/// An empty value counts as unset.
pub fn resolve(env: Option<OsString>, explicit: Option<&Path>, default: Option<&Path>) -> Option<PathBuf> {
    env.filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| explicit.filter(|p| !p.as_os_str().is_empty()).map(Path::to_path_buf))
        .or_else(|| default.map(Path::to_path_buf))
}

impl PluginPaths {
    /// The TRust host program, wherever SLPlugin is (it is not in the pack).
    pub fn trust_host(&self) -> Option<PathBuf> {
        let c = config();
        resolve(std::env::var_os(ENV), c.explicit.as_deref(), c.default.as_deref())
    }
}

thread_local! {
    /// The kept profile of the page this thread launches next (a listed
    /// site's TRust profile: its folder and its domain).
    static KEPT_SITE: std::cell::RefCell<Option<(PathBuf, String)>> = const { std::cell::RefCell::new(None) };
}

/// Run `start` (a launch) with `kept` as the page's kept profile.
pub fn with_kept_site<T>(kept: Option<(PathBuf, String)>, start: impl FnOnce() -> T) -> T {
    KEPT_SITE.with(|k| *k.borrow_mut() = kept);
    let out = start();
    KEPT_SITE.with(|k| *k.borrow_mut() = None);
    out
}

/// `<root>/<viewer pid>-<n>`: one run folder per page.
pub fn run_dir(root: &Path, viewer_pid: u32, n: u32) -> PathBuf {
    root.join(format!("{viewer_pid}-{n}"))
}

/// The host's command line, after its program.
pub fn host_args(run_dir: &Path, log_file: Option<&Path>, kept: Option<(&Path, &str)>) -> Vec<OsString> {
    let mut a: Vec<OsString> = vec!["--ipc-fd".into(), "3".into(), "--run-dir".into(), run_dir.into()];
    if let Some(log) = log_file {
        a.extend([OsString::from("--log"), log.into()]);
    }
    if let Some((dir, site)) = kept {
        a.extend([OsString::from("--site-profile"), dir.into(), OsString::from("--site"), site.into()]);
    }
    a
}

/// Start the host for one page.
pub(crate) fn launch(host: &Path, env: &[(String, String)]) -> std::io::Result<PluginProcess> {
    let c = config();
    let Some(root) = c.run_root else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "no run folder for the TRust host",
        ));
    };
    std::fs::create_dir_all(&root)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700));
    }
    let dir = run_dir(&root, std::process::id(), RUN_NUMBER.fetch_add(1, Ordering::Relaxed));
    let kept = KEPT_SITE.with(|k| k.borrow().clone());
    let kept = kept.as_ref().map(|(dir, site)| (dir.as_path(), site.as_str()));
    PluginProcess::launch_host(host, &host_args(&dir, c.log_file.as_deref(), kept), env)
}

/// Run folders a host left behind (killed, or a crash of the viewer): those
/// of viewers that no longer run. `everything`: the whole folder, for
/// "Chromium only", which keeps nothing of TRust on disk. Returns how many
/// folders were removed.
pub fn sweep(root: &Path, everything: bool) -> usize {
    let Ok(entries) = std::fs::read_dir(root) else {
        return 0;
    };
    let mut removed = 0;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let viewer = name
            .to_str()
            .and_then(|n| n.split_once('-'))
            .and_then(|(pid, _)| pid.parse::<u32>().ok());
        let stale = everything || viewer.is_none_or(|pid| pid == std::process::id() || !process_runs(pid));
        if !stale {
            continue;
        }
        // a stray file or link is removed itself, never followed
        let gone = match entry.file_type() {
            Ok(t) if t.is_dir() => std::fs::remove_dir_all(entry.path()),
            _ => std::fs::remove_file(entry.path()),
        };
        match gone {
            Ok(()) => removed += 1,
            Err(e) => log::warn!("media: TRust run folder not removed ({e})"),
        }
    }
    if everything {
        let _ = std::fs::remove_dir(root);
    }
    removed
}

#[cfg(unix)]
fn process_runs(pid: u32) -> bool {
    let Ok(pid) = libc::pid_t::try_from(pid) else {
        return false;
    };
    if pid <= 0 {
        return false;
    }
    // SAFETY: signal 0 sends nothing; it only reports whether the process exists
    let r = unsafe { libc::kill(pid, 0) };
    r == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(not(unix))]
fn process_runs(_pid: u32) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_override_wins_then_the_chosen_path_then_the_default() {
        let chosen = Path::new("/opt/chosen/aurora-trust-host");
        let default = Path::new("/home/x/.local/share/auroraviewer/trust-host/aurora-trust-host");
        assert_eq!(
            resolve(Some("/t/h".into()), Some(chosen), Some(default)),
            Some(PathBuf::from("/t/h"))
        );
        assert_eq!(resolve(Some("".into()), Some(chosen), Some(default)).as_deref(), Some(chosen));
        assert_eq!(resolve(None, Some(Path::new("")), Some(default)).as_deref(), Some(default));
        assert_eq!(resolve(None, None, None), None);
    }

    #[test]
    fn both_engines_are_browsers_and_nothing_else_is() {
        assert!(is_browser("media_plugin_cef"));
        assert!(is_browser(PLUGIN));
        assert!(!is_browser("media_plugin_gstreamer"));
        assert!(!is_browser("media_plugin_libvlc"));
    }

    #[test]
    fn the_host_gets_its_socket_run_folder_and_log() {
        let dir = run_dir(Path::new("/c/trust-run"), 42, 3);
        assert_eq!(dir, PathBuf::from("/c/trust-run/42-3"));
        let a = host_args(&dir, Some(Path::new("/l/x.trust-host.log")), None);
        let a: Vec<_> = a.iter().map(|s| s.to_string_lossy().into_owned()).collect();
        assert_eq!(
            a,
            ["--ipc-fd", "3", "--run-dir", "/c/trust-run/42-3", "--log", "/l/x.trust-host.log"]
        );
        // a listed site's kept profile
        let k = host_args(&dir, None, Some((Path::new("/c/trust-sites/example.org"), "example.org")));
        let k: Vec<_> = k.iter().map(|s| s.to_string_lossy().into_owned()).collect();
        assert_eq!(
            k,
            [
                "--ipc-fd",
                "3",
                "--run-dir",
                "/c/trust-run/42-3",
                "--site-profile",
                "/c/trust-sites/example.org",
                "--site",
                "example.org"
            ]
        );
        // only for the launch it wraps
        let inside = with_kept_site(Some((PathBuf::from("/k"), "k.org".into())), || {
            KEPT_SITE.with(|k| k.borrow().clone())
        });
        assert_eq!(inside, Some((PathBuf::from("/k"), "k.org".to_string())));
        assert_eq!(KEPT_SITE.with(|k| k.borrow().clone()), None);
    }

    #[cfg(unix)]
    #[test]
    fn the_sweep_keeps_a_running_viewers_folders_unless_told_everything() {
        let root = std::env::temp_dir().join(format!("aurora-trust-run-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        // pid 1 runs; a pid far above pid_max does not; ours counts as stale
        // (a folder of this run cannot exist before its first page)
        for name in ["1-0", "2147483000-4", &format!("{}-0", std::process::id()), "junk"] {
            std::fs::create_dir_all(root.join(name).join("home")).expect("fixture");
        }
        assert_eq!(sweep(&root, false), 3);
        assert!(root.join("1-0").is_dir());
        assert_eq!(sweep(&root, true), 1);
        assert!(!root.exists());
        assert_eq!(sweep(&root, true), 0);
    }
}
