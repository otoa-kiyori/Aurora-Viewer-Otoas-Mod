// aurora-local-patch: aurora-shader-host-library
//! The aurora-shader folder, read off the frame thread: `config.toml`, the
//! effects it lists, each checked against the contract its header names (v1
//! or v2) with naga (the one wgpu validates with, `wgpu::naga`) and its
//! `Params` reflected.
//!
//! One thread, `aurora-shader-watch`, polls the folder once a second
//! (`notify` is not a dependency): it stats `config.toml` and the listed
//! `.wgsl` files (mtime and length), waits 200 ms after a change for the
//! writer to finish (editors and install.sh replace files atomically, but
//! not every editor), then loads and publishes an `Arc<Library>` with a new
//! generation. The frame thread reads one atomic per frame.
//!
//! A problem is a `log::warn!` (Dev Toolbox L3 "Live problems" keeps the
//! last ones), logged once per state of a file, never per frame. A broken
//! effect keeps running its last good version; a broken `config.toml`
//! keeps the whole previous configuration.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime};

use parking_lot::Mutex;
use wgpu::naga;

use super::config::{self, Doc, Table, Value};
use super::gui::{self, EffectInfo};

/// The host's prelude: group 0, `Frame`, `FsIn`, the helpers (contract v1,
/// frozen).
pub(crate) const PRELUDE: &str = include_str!("shaders/user_post_prelude.wgsl");
/// Contract v2's additions (the blended faces' depth), put after the v1
/// prelude for an `api = 2` effect only: a v1 effect is compiled from the
/// same text as before, so it runs as before.
pub(crate) const PRELUDE_V2: &str = include_str!("shaders/user_post_prelude_v2.wgsl");
/// The newest contract version this host runs; it runs every one from 1.
pub(crate) const API: u32 = 2;
pub(crate) const MAX_EFFECTS: usize = 8;
/// Bytes of an effect's `Params` (group 1 binding 0) and of its buffer.
pub(crate) const PARAMS_SIZE: usize = 256;
/// Invocations of the one workgroup `prepare` runs as.
pub(crate) const PREPARE_MAX_INVOCATIONS: u32 = 256;
pub(crate) const CONFIG_MAX_BYTES: u64 = 64 * 1024;
pub(crate) const EFFECT_MAX_BYTES: u64 = 256 * 1024;
const POLL: Duration = Duration::from_secs(1);
const SETTLE: Duration = Duration::from_millis(200);

/// Group 0 as the prelude declares it: binding → name; v2 adds the last.
const GROUP0: [(u32, &str); 8] = [
    (0, "color"),
    (1, "depth"),
    (2, "normals"),
    (3, "linear_sampler"),
    (4, "point_sampler"),
    (5, "frame"),
    (6, "state"),
    (7, "alpha_depth"),
];

/// Group 0's bindings under contract `api`.
fn group0(api: u32) -> &'static [(u32, &'static str)] {
    let n = if api >= 2 { GROUP0.len() } else { 7 };
    &GROUP0[..n]
}

/// The prelude of contract `api`: v1's, then v2's additions.
pub(crate) fn prelude(api: u32) -> String {
    if api >= 2 {
        format!("{PRELUDE}\n{PRELUDE_V2}")
    } else {
        PRELUDE.to_owned()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScalarKind {
    F32,
    I32,
    U32,
}

/// One member of an effect's `Params`, where naga laid it out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Member {
    pub(crate) name: String,
    pub(crate) offset: u32,
    pub(crate) kind: ScalarKind,
    pub(crate) components: u32,
}

/// An effect that passed naga and the contract checks.
#[derive(Debug)]
pub(crate) struct Compiled {
    /// Its file name, for messages (`ink.wgsl`).
    pub(crate) file: String,
    /// Of its source text: a new hash is a new version (new pipelines,
    /// `state` zeroed).
    pub(crate) hash: u64,
    /// The prelude, a newline, the effect.
    pub(crate) source: Arc<str>,
    pub(crate) title: Option<String>,
    /// The header's `api`: its prelude, its `Frame::api` and flags.
    pub(crate) api: u32,
    /// fs_main or prepare reads `normals`: the host fills them.
    pub(crate) reads_normals: bool,
    pub(crate) prepare: bool,
    pub(crate) params: Vec<Member>,
    /// The header's `[defaults]`.
    pub(crate) defaults: Table,
    /// Its parameters as a GUI edits them (the header's `[ui]` hints).
    pub(crate) ui: Vec<gui::Param>,
}

/// An effect of the chain, its parameters resolved.
#[derive(Debug)]
pub(crate) struct Effect {
    /// The file stem, also its `config.toml` table.
    pub(crate) name: String,
    pub(crate) compiled: Arc<Compiled>,
    /// The `Params` buffer's bytes (PARAMS_SIZE).
    pub(crate) params: Vec<u8>,
}

/// What the frame thread runs: the effects in their order.
#[derive(Debug, Default)]
pub(crate) struct Library {
    pub(crate) effects: Vec<Effect>,
    /// `[host] photos`: the chain is drawn on photos too.
    pub(crate) photos: bool,
    /// `[host] vr`: the chain is drawn in each VR eye.
    pub(crate) vr: bool,
    /// Every effect of `order`, running or not, for the GUI (`gui.rs`).
    pub(crate) ui: Arc<Vec<EffectInfo>>,
}

// ------------------------------------------------------------ compiling

fn hash_of(text: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut h);
    h.finish()
}

/// Lines the prelude of contract `api` and its joining newline take before
/// an effect's line 1.
pub(crate) fn prelude_lines(api: u32) -> usize {
    prelude(api).matches('\n').count() + 1
}

/// `file:line:col` of a position in prelude + effect, in the effect's own
/// numbers (a position in the prelude says so).
fn place(file: &str, api: u32, loc: Option<naga::SourceLocation>) -> String {
    let Some(loc) = loc else {
        return file.to_owned();
    };
    let offset = prelude_lines(api);
    let line = loc.line_number as usize;
    if line > offset {
        format!("{file}:{}:{}", line - offset, loc.line_position)
    } else {
        format!("{file}: in the host prelude (line {line}:{})", loc.line_position)
    }
}

/// An error and the errors it comes from, on one line.
fn chain(error: &dyn std::error::Error) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(e) = source {
        let more = e.to_string();
        if !text.contains(&more) {
            text.push_str(": ");
            text.push_str(&more);
        }
        source = e.source();
    }
    text
}

/// Checks one effect: its header, then naga's parse and validation of
/// prelude + effect, then the contract of its `api`. `Err` lists why it
/// cannot run; warnings go to `warnings`.
pub(crate) fn compile(file: &str, body: &str, warnings: &mut Vec<String>) -> Result<Compiled, Vec<String>> {
    let header = config::parse_lines(config::header_lines(body).into_iter())
        .map_err(|e| vec![format!("{file}:{}: header: {}", e.line, e.message)])?;
    let root = header.root().cloned().unwrap_or_default();
    // every version from 1 runs: an effect written for v1 needs no change
    let api = match root.get("api") {
        Some(Value::Num(v)) if (1..=API).any(|a| *v == f64::from(a)) => *v as u32,
        Some(v) => {
            return Err(vec![format!(
                "{file}: header says api = {}, this host runs api = 1 to {API}",
                match v {
                    Value::Num(n) => n.to_string(),
                    other => other.kind().to_owned(),
                }
            )]);
        }
        None => return Err(vec![format!("{file}: the header has no `//! api = <1 to {API}>` line")]),
    };
    let title = match root.get("name") {
        None => None,
        Some(Value::Str(s)) => Some(s.clone()),
        Some(other) => {
            warnings.push(format!("{file}: header `name` should be a string, not {}", other.kind()));
            None
        }
    };
    for e in root.entries.iter().filter(|e| e.key != "api" && e.key != "name") {
        warnings.push(format!("{file}:{}: header key `{}` is unknown (ignored)", e.line, e.key));
    }
    for t in header
        .tables
        .iter()
        .filter(|t| !t.name.is_empty() && !matches!(t.name.as_str(), "defaults" | "ranges" | "ui"))
    {
        warnings.push(format!("{file}:{}: header table [{}] is unknown (ignored)", t.line, t.name));
    }
    let defaults = header.table("defaults").cloned().unwrap_or_default();

    // (v1: the same text as before v2 existed)
    let source: Arc<str> = format!("{}\n{body}", prelude(api)).into();
    // naga panicking on odd input must not end the watcher
    let checked = catch_unwind(AssertUnwindSafe(|| {
        let module =
            naga::front::wgsl::parse_str(&source).map_err(|e| format!("{}: {}", place(file, api, e.location(&source)), e.message()))?;
        let mut validator = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty());
        let info = validator
            .validate(&module)
            .map_err(|e| format!("{}: {}", place(file, api, e.location(&source)), chain(e.as_inner())))?;
        Ok::<_, String>((module, info))
    }));
    let (module, info) = match checked {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return Err(vec![e]),
        Err(_) => return Err(vec![format!("{file}: the shader compiler failed on this file (naga panicked)")]),
    };
    let contract = check_contract(file, api, &module, &info, &source)?;
    for e in &defaults.entries {
        if !contract.params.iter().any(|m| m.name == e.key) {
            let known: Vec<&str> = contract.params.iter().map(|m| m.name.as_str()).collect();
            warnings.push(format!(
                "{file}:{}: [defaults] `{}` is not a member of Params ({})",
                e.line,
                e.key,
                if known.is_empty() {
                    "it has none".to_owned()
                } else {
                    known.join(", ")
                }
            ));
        }
    }
    let ui = gui::params(
        file,
        &contract.params,
        &defaults,
        header.table("ui"),
        header.table("ranges"),
        warnings,
    );
    Ok(Compiled {
        file: file.to_owned(),
        hash: hash_of(body),
        source,
        title,
        api,
        reads_normals: contract.reads_normals,
        prepare: contract.prepare,
        params: contract.params,
        defaults,
        ui,
    })
}

struct Contract {
    reads_normals: bool,
    prepare: bool,
    params: Vec<Member>,
}

fn check_contract(
    file: &str,
    api: u32,
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    source: &str,
) -> Result<Contract, Vec<String>> {
    let mut errors = Vec::new();
    let at = |span: naga::Span| place(file, api, span.is_defined().then(|| span.location(source)));
    let mut fs_main = false;
    let mut prepare = false;
    for ep in &module.entry_points {
        match (ep.name.as_str(), ep.stage) {
            ("fs_main", naga::ShaderStage::Fragment) => fs_main = true,
            ("prepare", naga::ShaderStage::Compute) => {
                let invocations = ep.workgroup_size.iter().map(|&n| u64::from(n)).product::<u64>();
                if ep.workgroup_size_overrides.is_some() || invocations > u64::from(PREPARE_MAX_INVOCATIONS) {
                    errors.push(format!(
                        "{file}: prepare's @workgroup_size must be constant and at most {PREPARE_MAX_INVOCATIONS} invocations"
                    ));
                }
                prepare = true;
            }
            (name, stage) => errors.push(format!(
                "{file}: entry point `{name}` ({stage:?}) is not part of the contract: only `fs_main` (@fragment) and `prepare` (@compute)"
            )),
        }
    }
    if !fs_main {
        errors.push(format!("{file}: no `@fragment fn fs_main(in: FsIn) -> @location(0) vec4<f32>`"));
    }
    if !module.overrides.is_empty() {
        errors.push(format!("{file}: `override` declarations are not supported (contract v{api})"));
    }
    let mut params = Vec::new();
    let mut normals = None;
    for (handle, var) in module.global_variables.iter() {
        let Some(binding) = &var.binding else { continue };
        let name = var.name.as_deref().unwrap_or("?");
        let span = module.global_variables.get_span(handle);
        match binding.group {
            0 if group0(api).contains(&(binding.binding, name)) => {
                if name == "normals" {
                    normals = Some(handle);
                }
            }
            1 if binding.binding == 0 => match (&var.space, &module.types[var.ty].inner) {
                (naga::AddressSpace::Uniform, naga::TypeInner::Struct { members, span: size }) => {
                    if *size as usize > PARAMS_SIZE {
                        errors.push(format!("{}: `{name}` is {size} bytes, at most {PARAMS_SIZE}", at(span)));
                    }
                    for m in members {
                        let member = m.name.clone().unwrap_or_default();
                        let (kind, components) = match &module.types[m.ty].inner {
                            naga::TypeInner::Scalar(s) => (scalar_kind(*s), 1),
                            naga::TypeInner::Vector { size, scalar } => (scalar_kind(*scalar), *size as u32),
                            _ => (None, 0),
                        };
                        match kind {
                            Some(kind) => params.push(Member {
                                name: member,
                                offset: m.offset,
                                kind,
                                components,
                            }),
                            None => errors.push(format!(
                                "{}: Params member `{member}`: only f32, i32, u32 and their vec2/3/4 are supported",
                                at(span)
                            )),
                        }
                    }
                }
                _ => errors.push(format!("{}: @group(1) @binding(0) must be `var<uniform> params: <a struct>`", at(span))),
            },
            group => errors.push(format!(
                "{}: @group({group}) @binding({}) `{name}` is not part of the contract (group 0 is the prelude's, group 1 has binding 0 only)",
                at(span),
                binding.binding
            )),
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }
    let reads_normals = normals.is_some_and(|h| (0..module.entry_points.len()).any(|i| !info.get_entry_point(i)[h].is_empty()));
    Ok(Contract {
        reads_normals,
        prepare,
        params,
    })
}

fn scalar_kind(s: naga::Scalar) -> Option<ScalarKind> {
    match (s.kind, s.width) {
        (naga::ScalarKind::Float, 4) => Some(ScalarKind::F32),
        (naga::ScalarKind::Sint, 4) => Some(ScalarKind::I32),
        (naga::ScalarKind::Uint, 4) => Some(ScalarKind::U32),
        _ => None,
    }
}

/// The `Params` bytes of an effect: each member from `config.toml`'s
/// table of the effect, else the header's `[defaults]`, else 0 (warned).
pub(crate) fn resolve_params(name: &str, compiled: &Compiled, table: Option<&Table>, warnings: &mut Vec<String>) -> Vec<u8> {
    let mut bytes = vec![0u8; PARAMS_SIZE];
    for m in &compiled.params {
        let n = m.components as usize;
        let from_config = table.and_then(|t| t.entries.iter().find(|e| e.key == m.name));
        let mut values = None;
        if let Some(e) = from_config {
            values = e.value.numbers(n);
            if values.is_none() {
                warnings.push(format!(
                    "config.toml:{}: [{name}] `{}` wants {}, not {}; the default is used",
                    e.line,
                    m.name,
                    wants(n),
                    e.value.kind()
                ));
            }
        }
        if values.is_none()
            && let Some(e) = compiled.defaults.entries.iter().find(|e| e.key == m.name)
        {
            values = e.value.numbers(n);
            if values.is_none() {
                warnings.push(format!(
                    "{}:{}: [defaults] `{}` wants {}, not {}",
                    compiled.file,
                    e.line,
                    m.name,
                    wants(n),
                    e.value.kind()
                ));
            }
        }
        let Some(values) = values else {
            warnings.push(format!(
                "{}: Params `{}` has no default and no value in config.toml; 0 is used",
                compiled.file, m.name
            ));
            continue;
        };
        for (i, v) in values.iter().enumerate() {
            let word = match m.kind {
                ScalarKind::F32 => (*v as f32).to_le_bytes(),
                ScalarKind::I32 => (v.round().clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32).to_le_bytes(),
                ScalarKind::U32 => (v.round().clamp(0.0, f64::from(u32::MAX)) as u32).to_le_bytes(),
            };
            let at = m.offset as usize + i * 4;
            if let Some(slot) = bytes.get_mut(at..at + 4) {
                slot.copy_from_slice(&word);
            }
        }
    }
    if let Some(t) = table {
        let known: Vec<&str> = compiled.params.iter().map(|m| m.name.as_str()).collect();
        for e in t.entries.iter().filter(|e| e.key != "enabled" && !known.contains(&e.key.as_str())) {
            warnings.push(format!(
                "config.toml:{}: [{name}] `{}` is not a parameter of {} ({})",
                e.line,
                e.key,
                compiled.file,
                if known.is_empty() {
                    "it has none".to_owned()
                } else {
                    known.join(", ")
                }
            ));
        }
    }
    bytes
}

fn wants(n: usize) -> String {
    if n == 1 {
        "a number".to_owned()
    } else {
        format!("an array of {n} numbers")
    }
}

// ------------------------------------------------------------- the folder

/// `config.toml` as last read correctly.
#[derive(Clone, Debug, Default)]
struct HostConfig {
    doc: Doc,
    enabled: bool,
    order: Vec<String>,
    photos: bool,
    vr: bool,
}

fn host_config(doc: Doc, warnings: &mut Vec<String>) -> HostConfig {
    let mut c = HostConfig {
        enabled: true,
        photos: true,
        vr: false,
        ..Default::default()
    };
    if let Some(root) = doc.root() {
        for e in &root.entries {
            warnings.push(format!("config.toml:{}: `{}` is outside a table (ignored)", e.line, e.key));
        }
    }
    if let Some(host) = doc.table("host") {
        for e in &host.entries {
            let bool_into = |slot: &mut bool, warnings: &mut Vec<String>| match e.value.as_bool() {
                Some(b) => *slot = b,
                None => warnings.push(format!("config.toml:{}: [host] `{}` wants true or false", e.line, e.key)),
            };
            match e.key.as_str() {
                "enabled" => bool_into(&mut c.enabled, warnings),
                "photos" => bool_into(&mut c.photos, warnings),
                "vr" => bool_into(&mut c.vr, warnings),
                "order" => match e.value.as_strs() {
                    Some(names) => {
                        for name in names {
                            if !config::is_table_name(name) {
                                warnings.push(format!(
                                    "config.toml:{}: order: {name:?} is not a file stem (a-z, 0-9, '_', '-')",
                                    e.line
                                ));
                            } else if c.order.iter().any(|n| n == name) {
                                warnings.push(format!("config.toml:{}: order lists {name:?} twice; the first is kept", e.line));
                            } else if c.order.len() == MAX_EFFECTS {
                                warnings.push(format!(
                                    "config.toml:{}: order: more than {MAX_EFFECTS} effects; {name:?} skipped",
                                    e.line
                                ));
                            } else {
                                c.order.push(name.to_owned());
                            }
                        }
                    }
                    None => warnings.push(format!("config.toml:{}: [host] order wants an array of names", e.line)),
                },
                other => warnings.push(format!(
                    "config.toml:{}: [host] `{other}` is unknown (enabled, order, photos, vr)",
                    e.line
                )),
            }
        }
    }
    c.doc = doc;
    c
}

struct Cached {
    hash: u64,
    result: Result<Arc<Compiled>, Vec<String>>,
    warnings: Vec<String>,
    last_good: Option<Arc<Compiled>>,
}

/// Reads a text file of at most `max` bytes. `Ok(None)`: no such file.
pub(crate) fn read_text(path: &Path, max: u64) -> Result<Option<String>, String> {
    use std::io::Read;
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.to_string()),
    };
    let mut bytes = Vec::new();
    file.take(max + 1).read_to_end(&mut bytes).map_err(|e| e.to_string())?;
    if bytes.len() as u64 > max {
        return Err(format!("larger than {} KiB", max / 1024));
    }
    String::from_utf8(bytes).map(Some).map_err(|_| "not UTF-8 text".to_owned())
}

/// The folder's state between two loads.
pub(crate) struct Loader {
    dir: PathBuf,
    /// `config.toml`'s hash and what it said, as last read correctly.
    config: Option<(u64, HostConfig)>,
    effects: HashMap<String, Cached>,
    /// Per file, the problems last logged (they are logged again only when
    /// they change).
    logged: HashMap<String, u64>,
}

/// Mtime and length of a watched file (`None`: absent).
type Stamp = Option<(SystemTime, u64)>;

impl Loader {
    pub(crate) fn new(dir: PathBuf) -> Loader {
        Loader {
            dir,
            config: None,
            effects: HashMap::new(),
            logged: HashMap::new(),
        }
    }

    fn watched(&self) -> Vec<PathBuf> {
        let mut paths = vec![self.dir.join("config.toml")];
        if let Some((_, c)) = &self.config {
            paths.extend(c.order.iter().map(|n| self.dir.join(format!("{n}.wgsl"))));
        }
        paths
    }

    fn stamps(&self) -> Vec<(PathBuf, Stamp)> {
        self.watched()
            .into_iter()
            .map(|p| {
                let stamp = std::fs::metadata(&p)
                    .ok()
                    .map(|m| (m.modified().unwrap_or(SystemTime::UNIX_EPOCH), m.len()));
                (p, stamp)
            })
            .collect()
    }

    /// Reads the folder; returns the library and, per file, its problems
    /// (an empty list for a file that has none now).
    pub(crate) fn load(&mut self) -> (Library, Vec<(String, Vec<String>)>) {
        let mut report: Vec<(String, Vec<String>)> = Vec::new();
        let mut config_warnings = Vec::new();
        if !self.dir.is_dir() {
            // no folder, no aurora-shader: nothing to say
            self.config = None;
            return (Library::default(), report);
        }
        match read_text(&self.dir.join("config.toml"), CONFIG_MAX_BYTES) {
            Ok(None) => {
                config_warnings.push("config.toml: missing; no effect runs".to_owned());
                self.config = None;
            }
            Ok(Some(text)) => {
                let hash = hash_of(&text);
                if self.config.as_ref().is_none_or(|(h, _)| *h != hash) {
                    match config::parse(&text, 1) {
                        Ok(doc) => self.config = Some((hash, host_config(doc, &mut config_warnings))),
                        Err(e) => config_warnings.push(format!(
                            "config.toml:{}: {}; {}",
                            e.line,
                            e.message,
                            if self.config.is_some() {
                                "the previous configuration stays"
                            } else {
                                "no effect runs"
                            }
                        )),
                    }
                } else if let Some((_, c)) = &self.config {
                    // the same text: its warnings again, for the report
                    host_config(c.doc.clone(), &mut config_warnings);
                }
            }
            Err(e) => config_warnings.push(format!("config.toml: cannot be read ({e})")),
        }
        let Some((_, host)) = self.config.clone() else {
            report.push(("config.toml".into(), config_warnings));
            return (Library::default(), report);
        };
        let mut library = Library {
            effects: Vec::new(),
            photos: host.photos,
            vr: host.vr,
            ui: Arc::default(),
        };
        let mut ui = Vec::new();
        // every effect of `order` is compiled (once per version), also one
        // turned off: the GUI shows its parameters. Only those that run
        // report their problems
        for name in &host.order {
            let table = host.doc.table(name);
            let runs = host.enabled && table.and_then(|t| t.get("enabled")).and_then(Value::as_bool) != Some(false);
            let file = format!("{name}.wgsl");
            let mut info = EffectInfo {
                name: name.clone(),
                title: name.clone(),
                params: Vec::new(),
                problem: None,
            };
            let body = match read_text(&self.dir.join(&file), EFFECT_MAX_BYTES) {
                Ok(Some(body)) => body,
                Ok(None) => {
                    info.problem = Some(format!("{file} is missing"));
                    ui.push(info);
                    if runs {
                        config_warnings.push(format!("config.toml: order names {name:?} but {file} is missing; skipped"));
                    }
                    continue;
                }
                Err(e) => {
                    let problem = format!("{file}: cannot be read ({e}); skipped");
                    info.problem = Some(problem.clone());
                    ui.push(info);
                    if runs {
                        report.push((file.clone(), vec![problem]));
                    }
                    continue;
                }
            };
            let hash = hash_of(&body);
            let cached = self.effects.entry(name.clone()).or_insert_with(|| Cached {
                hash: hash.wrapping_add(1),
                result: Err(Vec::new()),
                warnings: Vec::new(),
                last_good: None,
            });
            if cached.hash != hash {
                let mut warnings = Vec::new();
                cached.result = compile(&file, &body, &mut warnings).map(Arc::new);
                cached.hash = hash;
                cached.warnings = warnings;
                if let Ok(c) = &cached.result {
                    cached.last_good = Some(c.clone());
                }
            }
            let mut problems = cached.warnings.clone();
            let compiled = match &cached.result {
                Ok(c) => c.clone(),
                Err(errors) => {
                    problems.extend(errors.iter().cloned());
                    match &cached.last_good {
                        Some(good) => {
                            problems.push(format!("{file}: the last version that worked keeps running"));
                            good.clone()
                        }
                        None => {
                            problems.push(format!("{file}: skipped until it is fixed"));
                            info.problem = Some(errors.join("\n"));
                            ui.push(info);
                            if runs {
                                report.push((file, problems));
                            }
                            continue;
                        }
                    }
                }
            };
            info.title = compiled.title.clone().unwrap_or_else(|| name.clone());
            info.params = compiled.ui.clone();
            ui.push(info);
            if !runs {
                continue;
            }
            let params = resolve_params(name, &compiled, table, &mut config_warnings);
            report.push((file, problems));
            library.effects.push(Effect {
                name: name.clone(),
                compiled,
                params,
            });
        }
        library.ui = Arc::new(ui);
        report.push(("config.toml".into(), config_warnings));
        (library, report)
    }

    /// Logs the problems of each file when they changed since last logged.
    fn log_report(&mut self, report: &[(String, Vec<String>)]) {
        for (file, problems) in report {
            let h = hash_of(&problems.join("\n"));
            if self.logged.insert(file.clone(), h) != Some(h) {
                for p in problems {
                    log::warn!("aurora-shader: {p}");
                }
            }
        }
    }
}

// ------------------------------------------------------- the watcher thread

struct Shared {
    dir: Option<PathBuf>,
    library: Option<Arc<Library>>,
}

static SHARED: Mutex<Shared> = Mutex::new(Shared { dir: None, library: None });
/// Bumped at each publication; 0 = never published (the frame thread then
/// does nothing past this one load).
static GENERATION: AtomicU64 = AtomicU64::new(0);
static WATCHING: AtomicBool = AtomicBool::new(false);
/// The folder has a `config.toml`, as the watcher last saw it.
static INSTALLED: AtomicBool = AtomicBool::new(false);

pub(crate) fn installed() -> bool {
    INSTALLED.load(Ordering::Acquire)
}

/// The folder being watched (None: the host is off).
pub(crate) fn folder() -> Option<PathBuf> {
    SHARED.lock().dir.clone()
}

pub(crate) fn generation() -> u64 {
    GENERATION.load(Ordering::Acquire)
}

pub(crate) fn current() -> Option<Arc<Library>> {
    SHARED.lock().library.clone()
}

/// The folder to watch; starts the watcher the first time.
pub(crate) fn watch(dir: PathBuf) {
    SHARED.lock().dir = Some(dir);
    if WATCHING.swap(true, Ordering::AcqRel) {
        return;
    }
    // (naga's parser is recursive: a roomy stack, as wgpu's docs advise)
    let spawned = std::thread::Builder::new()
        .name("aurora-shader-watch".into())
        .stack_size(8 << 20)
        .spawn(watch_loop);
    if let Err(e) = spawned {
        WATCHING.store(false, Ordering::Release);
        log::warn!("aurora-shader: cannot start the folder watcher: {e}");
    }
}

fn publish(library: Library) {
    SHARED.lock().library = Some(Arc::new(library));
    GENERATION.fetch_add(1, Ordering::AcqRel);
}

fn watch_loop() {
    let mut loader: Option<Loader> = None;
    let mut seen: Option<Vec<(PathBuf, Stamp)>> = None;
    loop {
        let Some(dir) = SHARED.lock().dir.clone() else {
            std::thread::sleep(POLL);
            continue;
        };
        if loader.as_ref().is_none_or(|l| l.dir != dir) {
            loader = Some(Loader::new(dir));
            seen = None;
        }
        let Some(l) = loader.as_mut() else { continue };
        let stamps = l.stamps();
        INSTALLED.store(stamps.first().is_some_and(|(_, s)| s.is_some()), Ordering::Release);
        if seen.as_ref() != Some(&stamps) {
            // a file is being written: load once it stays the same
            if seen.is_some() {
                std::thread::sleep(SETTLE);
                if l.stamps() != stamps {
                    continue;
                }
            }
            let started = Instant::now();
            let loaded = catch_unwind(AssertUnwindSafe(|| l.load()));
            match loaded {
                Ok((library, report)) => {
                    l.log_report(&report);
                    if !library.effects.is_empty() || seen.is_some() {
                        let names: Vec<&str> = library.effects.iter().map(|e| e.name.as_str()).collect();
                        log::info!(
                            "aurora-shader: {} effect(s) [{}] from {} ({:.1} ms)",
                            names.len(),
                            names.join(", "),
                            l.dir.display(),
                            started.elapsed().as_secs_f64() * 1e3
                        );
                    }
                    publish(library);
                }
                Err(_) => log::warn!("aurora-shader: loading the folder failed (internal error); the previous effects stay"),
            }
            // a file changed while it was read: the stamps taken before the
            // load make the next turn load again (unless the list of files
            // itself changed, `order` being edited)
            let after = l.stamps();
            let same_files = after.iter().map(|s| &s.0).eq(stamps.iter().map(|s| &s.0));
            seen = Some(if same_files { stamps } else { after });
        }
        std::thread::sleep(POLL);
    }
}
