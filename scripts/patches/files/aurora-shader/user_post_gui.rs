// aurora-local-patch: aurora-shader-host-gui
//! What a GUI needs from the aurora-shader host (Preferences › Extras):
//! whether the folder is installed, each effect's parameters with the
//! control that edits them, and the in-place editor of `config.toml`.
//!
//! `config.toml` stays the only source of truth: the GUI writes its values
//! into it (comments and layout kept, `user_post_config.rs` `edit`), with
//! an atomic replace, and the watcher applies the file as it applies a
//! hand edit. Nothing here touches the frame thread.
//!
//! The controls come from an optional `[ui]` table of the effect's `//!`
//! header (an addition to contract v1, see CONTRACT.md "GUI hints"):
//!
//! ```text
//! //! [ui]
//! //! density = [0.0, 1.0, 0.01]        # min, max, optional step
//! //! density-label = "Density"         # a short label
//! //! radial-kind = "flag"              # "slider", "flag" or "color"
//! ```
//!
//! (`-` cannot be part of a WGSL name, so `<member>-label` never meets a
//! member.) Without a hint, v1's `[ranges]` gives min and max; without
//! either, a number gets 0..max(1, 2 × default) (down to 2 × default when
//! that is negative), an integer scalar whose default is 0 or 1 is a flag,
//! a vec3 a colour.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::config::{self, Doc, Table, Value};
use super::library::{Member, ScalarKind};

pub use super::config::{Edit, bool_token, names_token, number_token, numbers_token};

/// The control a parameter takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    /// One slider per component.
    Slider,
    /// A switch: 0 or 1 (a scalar).
    Flag,
    /// A colour button: a vec3 (or a vec4 with alpha), 0..1 per channel.
    Color,
}

/// One member of an effect's `Params`, as a GUI shows it.
#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    /// The member, also its key in `config.toml`.
    pub name: String,
    /// `[ui] <name>-label`, else the member's name.
    pub label: String,
    /// `[ui] <name>-group`: the related values shown together.
    pub group: Option<String>,
    /// `[ui] <name>-ordered = true`: with the other ordered members of its
    /// group, in member order, a chain that never decreases (p1 <= p2 ...).
    pub ordered: bool,
    pub control: Control,
    pub components: usize,
    /// An `i32` / `u32` member: whole values.
    pub integer: bool,
    pub min: f64,
    pub max: f64,
    pub step: Option<f64>,
    /// The header's `[defaults]` (0 where it has none).
    pub default: Vec<f64>,
}

/// An effect of `order`, running or not.
#[derive(Clone, Debug, PartialEq)]
pub struct EffectInfo {
    /// The file stem, also its `config.toml` table.
    pub name: String,
    /// The header's `name`, else the stem.
    pub title: String,
    pub params: Vec<Param>,
    /// Why it cannot run (missing file, compile errors with no version that
    /// worked); its parameters are then unknown.
    pub problem: Option<String>,
}

/// The parameters of a compiled effect, with the hints of its header;
/// problems of the hints go to `warnings` (they never stop the effect).
pub(crate) fn params(
    file: &str,
    members: &[Member],
    defaults: &Table,
    ui: Option<&Table>,
    ranges: Option<&Table>,
    warnings: &mut Vec<String>,
) -> Vec<Param> {
    if let Some(ui) = ui {
        check_ui(file, members, ui, warnings);
    }
    members
        .iter()
        .map(|m| {
            let n = m.components as usize;
            let integer = m.kind != ScalarKind::F32;
            let default = defaults.get(&m.name).and_then(|v| v.numbers(n)).unwrap_or_else(|| vec![0.0; n]);
            let hint = |suffix: &str| ui.and_then(|t| t.get(&format!("{}{suffix}", m.name)));
            let range = hint("")
                .and_then(range_of)
                .or_else(|| ranges.and_then(|t| t.get(&m.name)).and_then(range_of));
            let label = hint("-label")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .unwrap_or(&m.name)
                .to_owned();
            let group = hint("-group")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|g| !g.is_empty())
                .map(str::to_owned);
            let ordered = n == 1 && hint("-ordered").and_then(Value::as_bool) == Some(true);
            let asked = hint("-kind").and_then(Value::as_str).and_then(control_named);
            let control = match asked {
                Some(c) if fits(c, n) => c,
                _ => {
                    let zero_one = default.iter().all(|v| *v == 0.0 || *v == 1.0);
                    let range_zero_one = range.is_none_or(|(lo, hi, _)| lo == 0.0 && hi == 1.0);
                    if n == 1 && integer && zero_one && range_zero_one {
                        Control::Flag
                    } else if n == 3 && range.is_none() {
                        Control::Color
                    } else {
                        Control::Slider
                    }
                }
            };
            let (min, max, step) = match (control, range) {
                (Control::Flag | Control::Color, _) => (0.0, 1.0, None),
                (Control::Slider, Some(r)) => r,
                (Control::Slider, None) => {
                    let lo = default.iter().fold(0.0_f64, |a, v| a.min(2.0 * v));
                    let hi = default.iter().fold(1.0_f64, |a, v| a.max(2.0 * v));
                    (lo, hi, None)
                }
            };
            Param {
                name: m.name.clone(),
                label,
                group,
                ordered,
                control,
                components: n,
                integer,
                min,
                max,
                step: step.or(integer.then_some(1.0)),
                default,
            }
        })
        .collect()
}

fn control_named(name: &str) -> Option<Control> {
    match name {
        "slider" => Some(Control::Slider),
        "flag" => Some(Control::Flag),
        "color" => Some(Control::Color),
        _ => None,
    }
}

/// A flag edits a scalar, a colour 3 or 4 channels, sliders anything.
fn fits(control: Control, components: usize) -> bool {
    match control {
        Control::Slider => true,
        Control::Flag => components == 1,
        Control::Color => matches!(components, 3 | 4),
    }
}

/// `[min, max]` or `[min, max, step]`, min < max, step > 0.
fn range_of(value: &Value) -> Option<(f64, f64, Option<f64>)> {
    let v = value.numbers(2).or_else(|| value.numbers(3))?;
    let (lo, hi) = (v[0], v[1]);
    let step = v.get(2).copied();
    (lo < hi && step.is_none_or(|s| s > 0.0)).then_some((lo, hi, step))
}

fn check_ui(file: &str, members: &[Member], ui: &Table, warnings: &mut Vec<String>) {
    for e in &ui.entries {
        let (member, suffix) = match e.key.split_once('-') {
            Some((m, s)) => (m, Some(s)),
            None => (e.key.as_str(), None),
        };
        let Some(m) = members.iter().find(|m| m.name == member) else {
            warnings.push(format!(
                "{file}:{}: [ui] `{}`: `{member}` is not a member of Params (ignored)",
                e.line, e.key
            ));
            continue;
        };
        let problem = match suffix {
            None => range_of(&e.value)
                .is_none()
                .then_some("wants [min, max] or [min, max, step], min < max, step > 0"),
            Some("label") => e
                .value
                .as_str()
                .is_none_or(|s| s.trim().is_empty())
                .then_some("wants a short \"label\""),
            Some("kind") => match e.value.as_str().and_then(control_named) {
                None => Some("wants \"slider\", \"flag\" or \"color\""),
                Some(c) if !fits(c, m.components as usize) => Some("does not fit the member (a flag is a scalar, a colour a vec3 or vec4)"),
                Some(_) => None,
            },
            Some("group") => e
                .value
                .as_str()
                .is_none_or(|s| s.trim().is_empty())
                .then_some("wants a \"group name\""),
            Some("ordered") => match e.value.as_bool() {
                None => Some("wants true or false"),
                Some(true) if m.components != 1 => Some("is for a scalar member"),
                Some(_) => None,
            },
            Some(_) => Some("is unknown (<member>, <member>-label, -kind, -group, -ordered)"),
        };
        if let Some(p) = problem {
            warnings.push(format!("{file}:{}: [ui] `{}` {p} (ignored)", e.line, e.key));
        }
    }
}

// ------------------------------------------------------------ public API

/// An ordered chain (p1 <= p2 <= ...) after `values[moved]` changed: a
/// neighbour it passed is pushed along to it, the others stay.
pub fn push_ordered(values: &mut [f64], moved: usize) {
    let Some(&v) = values.get(moved) else { return };
    for (j, w) in values.iter_mut().enumerate() {
        if (j > moved && *w < v) || (j < moved && *w > v) {
            *w = v;
        }
    }
}

/// The folder the host reads has a `config.toml` (checked by the watcher
/// once a second). False when the host is off (`AURORA_SHADER=0`) or not
/// configured.
pub fn installed() -> bool {
    super::library::installed()
}

/// The chain has an effect to draw: false while the master switch (`[host]
/// enabled`) is off, or no effect of `order` loads.
pub fn running() -> bool {
    super::library::current().is_some_and(|l| !l.effects.is_empty())
}

/// Bumped each time the watcher loads the folder again: a GUI reads
/// `config.toml` and [`effects`] again when it changes.
pub fn generation() -> u64 {
    super::library::generation()
}

/// The effects `order` lists, as last loaded (also those turned off).
pub fn effects() -> Arc<Vec<EffectInfo>> {
    super::library::current().map(|l| l.ui.clone()).unwrap_or_default()
}

/// The effects of a folder (its `.wgsl` files whose stem is a table name),
/// by stem, with the header's `name` (else the stem): what a GUI offers to
/// add to `order`. Reads the headers only.
pub fn available(dir: &Path) -> Vec<(String, String)> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<(String, String)> = entries
        .flatten()
        .filter_map(|e| {
            let path = e.path();
            let stem = path
                .file_name()?
                .to_str()?
                .strip_suffix(".wgsl")
                .filter(|s| config::is_table_name(s))?
                .to_owned();
            let title = super::library::read_text(&path, super::library::EFFECT_MAX_BYTES)
                .ok()
                .flatten()
                .and_then(|body| {
                    config::parse_lines(config::header_lines(&body).into_iter())
                        .ok()
                        .and_then(|h| h.root().and_then(|r| r.get("name")).and_then(Value::as_str).map(str::to_owned))
                })
                .unwrap_or_else(|| stem.clone());
            Some((stem, title))
        })
        .collect();
    out.sort();
    out
}

/// `config.toml` of the folder the host reads (None: the host is off).
pub fn config_path() -> Option<PathBuf> {
    super::library::folder().map(|d| d.join("config.toml"))
}

/// Reads `config.toml` (at most 64 KiB, UTF-8).
pub fn read_config(path: &Path) -> Result<String, String> {
    match super::library::read_text(path, super::library::CONFIG_MAX_BYTES) {
        Ok(Some(text)) => Ok(text),
        Ok(None) => Err(format!("{}: missing", path.display())),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

/// `config.toml` read with the host's own rules.
#[derive(Clone, Debug, Default)]
pub struct ConfigView(Doc);

impl ConfigView {
    /// The error names the line, as the host's warning does.
    pub fn parse(text: &str) -> Result<ConfigView, String> {
        config::parse(text, 1)
            .map(ConfigView)
            .map_err(|e| format!("config.toml line {}: {}", e.line, e.message))
    }

    fn get(&self, table: &str, key: &str) -> Option<&Value> {
        self.0.table(table).and_then(|t| t.get(key))
    }

    /// `[host] enabled` (default true).
    pub fn host_enabled(&self) -> bool {
        self.get("host", "enabled").and_then(Value::as_bool).unwrap_or(true)
    }

    /// `[host] photos` (default true).
    pub fn photos(&self) -> bool {
        self.get("host", "photos").and_then(Value::as_bool).unwrap_or(true)
    }

    /// `[host] order` as written (names the host refuses included).
    pub fn order(&self) -> Vec<String> {
        self.get("host", "order")
            .and_then(Value::as_strs)
            .map(|names| names.into_iter().map(str::to_owned).collect())
            .unwrap_or_default()
    }

    /// `[<effect>] enabled` (default true).
    pub fn effect_enabled(&self, effect: &str) -> bool {
        self.get(effect, "enabled").and_then(Value::as_bool).unwrap_or(true)
    }

    /// `[<effect>] <key>` as `n` numbers, when it has that shape.
    pub fn numbers(&self, effect: &str, key: &str, n: usize) -> Option<Vec<f64>> {
        self.get(effect, key).and_then(|v| v.numbers(n))
    }

    /// The key is written in the file.
    pub fn has(&self, effect: &str, key: &str) -> bool {
        self.get(effect, key).is_some()
    }
}

/// The text of `config.toml` with `edits` applied in place (see
/// `user_post_config.rs` `edit`). Refuses a text the parser refuses.
pub fn edit_config(text: &str, edits: &[Edit]) -> Result<String, String> {
    let out = config::edit(text, edits)?;
    if out.len() as u64 > super::library::CONFIG_MAX_BYTES {
        return Err(format!(
            "config.toml would be larger than {} KiB",
            super::library::CONFIG_MAX_BYTES / 1024
        ));
    }
    Ok(out)
}

/// Applies `edits` to `config.toml` as it is on disk now (a hand edit
/// made meanwhile stays) and replaces it atomically. Returns the new text.
pub fn save_config(path: &Path, edits: &[Edit]) -> Result<String, String> {
    let text = read_config(path)?;
    let out = edit_config(&text, edits)?;
    if out != text {
        write_atomic(path, &out).map_err(|e| format!("{}: cannot be written ({e})", path.display()))?;
    }
    Ok(out)
}

/// Writes `text` to a temporary file of the same folder, then renames it
/// over `path`: the watcher (or an editor) never sees half a file.
pub fn write_atomic(path: &Path, text: &str) -> std::io::Result<()> {
    use std::io::Write;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "config.toml".to_owned());
    let tmp = path.with_file_name(format!(".{name}.aurora-{}.tmp", std::process::id()));
    let written = (|| {
        let mut file = std::fs::File::create(&tmp)?;
        // the file keeps its permissions (install.sh's 0644, not the umask's)
        if let Ok(meta) = std::fs::metadata(path) {
            file.set_permissions(meta.permissions())?;
        }
        file.write_all(text.as_bytes())?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&tmp, path)
    })();
    if written.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    written
}
