//! A local log of the web addresses the world shows (local patch set,
//! scripts/patches/78-url-log.toml), for a separate program that checks
//! them later. The viewer only writes the file: it never reads it back,
//! shows it or sends it anywhere.
//!
//! What is written, once per session for the same address from the same
//! face, parcel or speaker: the address of a media on a prim face (in the
//! world, on a worn item or a HUD), of the parcel media and of the parcel
//! music when it is turned on: first seen or changed while that kind of
//! media is on, or present when the log is turned on (the media filter's
//! answer at that moment is the entry's `state`), and every http or https
//! address said in nearby chat or in an IM by an avatar or an object (not
//! group chat, not our own lines). While a kind of media is off nothing is
//! written for it and its address is not taken as seen, so it is written
//! when it is turned on; turning it off, stopping or unloading it writes
//! nothing.
//!
//! Where: `<data dir>/url-log/YYYY-MM-DD.jsonl` (the folder of the logs'
//! folder; `url-log-demo` in the offline demo), the day of the entry in
//! UTC, one JSON object per line. The folder is made 0700 and the files
//! 0600 on Unix; files more than 30 days old are deleted. An address is
//! stored with a `*` before its scheme (`*https://…`), so that an editor
//! does not make the log clickable: a reader removes the first `*`.
//!
//! The frame thread only formats a line and queues it (bounded: a full
//! queue drops the line and counts it); one thread writes, nothing is
//! created on disk while the switch (media.url_log) is off.
//!
//! What a line says of where an address was shown, for a report: a prim
//! or HUD line has the linkset's name and owner when the viewer knows them
//! (`identity`), a chat line the speaker's region and position, an
//! object's IM the location it carries, a parcel line the parcel's owner.
//! Nothing is asked of the region for the log. Every line ends with the
//! format's version (`"v":2`); a line without it is of version 1, which
//! had none of these fields. The other keys keep their names and forms.
// aurora-local-patch: url-log-module

use crate::media::MediaSettings;
use crate::media::filter::Verdict;
use crate::world::{ChatKind, World};
use aurora_net::ParcelInfo;
use aurora_net::build::ObjectProps;
use chrono::{DateTime, NaiveDate, SecondsFormat, Utc};
use crossbeam_channel::{RecvTimeoutError, Sender, TrySendError};
use glam::Vec3;
use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;
use uuid::Uuid;

/// Days a file is kept, counted from the day in its name.
const KEEP_DAYS: i64 = 30;
/// Lines waiting for the writer; past that they are dropped.
const QUEUE_LINES: usize = 1024;
/// Addresses remembered for "once per session"; past that the set starts
/// again empty, so it cannot grow without limit.
const SEEN_CAP: usize = 20_000;
/// The writer wakes at least this often, for the daily clean-up.
const TICK: Duration = Duration::from_secs(3600);
/// Objects whose name and owner the region's answers told (`on_properties`);
/// past that the map starts again empty.
const KNOWN_CAP: usize = 4096;
/// The format of the lines, written last as `"v"`. Lines without it are of
/// version 1 (no `object_name` nor `owner` on media and parcel lines, no
/// `region` nor `pos` on chat and IM lines).
const VERSION: u32 = 2;

/// The mark written before an address's scheme. Changed here only.
const MARK: char = '*';

/// Where an address was shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Prim,
    Hud,
    ParcelMedia,
    ParcelMusic,
    Chat,
    Im,
}

impl Source {
    fn name(self) -> &'static str {
        match self {
            Source::Prim => "prim",
            Source::Hud => "hud",
            Source::ParcelMedia => "parcel-media",
            Source::ParcelMusic => "parcel-music",
            Source::Chat => "chat",
            Source::Im => "im",
        }
    }
}

/// What an address is "once per session" for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Key {
    Face(Uuid, u8),
    ParcelMedia(Option<u64>, i32),
    ParcelMusic(Option<u64>, i32),
    Speaker(Uuid),
}

/// One line of the log; `None`, empty names and nil ids are left out.
#[derive(Debug, Clone)]
struct Entry {
    time: DateTime<Utc>,
    source: Source,
    /// As it was expressed (the query and the fragment included).
    url: String,
    state: Option<&'static str>,
    region: Option<String>,
    /// Region metres.
    pos: Option<Vec3>,
    parcel: Option<String>,
    object: Option<Uuid>,
    face: Option<u8>,
    object_name: Option<String>,
    speaker: Option<Uuid>,
    speaker_name: Option<String>,
    speaker_kind: Option<&'static str>,
    owner: Option<Uuid>,
}

impl Entry {
    fn new(source: Source, url: &str) -> Self {
        Entry {
            time: Utc::now(),
            source,
            url: url.to_owned(),
            state: None,
            region: None,
            pos: None,
            parcel: None,
            object: None,
            face: None,
            object_name: None,
            speaker: None,
            speaker_name: None,
            speaker_kind: None,
            owner: None,
        }
    }
}

/// The address with `MARK` just before its scheme: after any leading
/// whitespace or control character, so that nothing an editor would
/// recognise is left in front of it; an address without a scheme gets it
/// there too. Nothing else changes.
pub fn defang(url: &str) -> String {
    let at = url.find(|c: char| !(c.is_whitespace() || c.is_control())).unwrap_or(url.len());
    let mut out = String::with_capacity(url.len() + MARK.len_utf8());
    out.push_str(&url[..at]);
    out.push(MARK);
    out.push_str(&url[at..]);
    out
}

/// The host of an address, lower case, when it parses as a URL with one.
fn host_of(url: &str) -> Option<String> {
    reqwest::Url::parse(url.trim())
        .ok()
        .and_then(|u| u.host_str().map(str::to_ascii_lowercase))
        .filter(|h| !h.is_empty())
}

fn json_str(v: &str) -> String {
    serde_json::Value::String(v.to_owned()).to_string()
}

/// A JSON object written key by key, in the order they are given.
struct Obj(String);

impl Obj {
    fn key(&mut self, k: &str) {
        if self.0.len() > 1 {
            self.0.push(',');
        }
        self.0.push_str(&json_str(k));
        self.0.push(':');
    }
    fn str(&mut self, k: &str, v: Option<&str>) {
        if let Some(v) = v.filter(|v| !v.is_empty()) {
            self.key(k);
            self.0.push_str(&json_str(v));
        }
    }
    fn id(&mut self, k: &str, v: Option<Uuid>) {
        if let Some(v) = v.filter(|v| !v.is_nil()) {
            self.key(k);
            self.0.push_str(&json_str(&v.to_string()));
        }
    }
}

/// The line of an entry, ending with its line break: the keys in a fixed
/// order, only those that apply.
fn line(e: &Entry) -> String {
    let mut o = Obj(String::from("{"));
    o.str("t", Some(e.time.to_rfc3339_opts(SecondsFormat::Secs, true).as_str()));
    o.str("source", Some(e.source.name()));
    o.str("url", Some(defang(&e.url).as_str()));
    o.str("host", host_of(&e.url).as_deref());
    o.str("state", e.state);
    o.str("region", e.region.as_deref());
    if let Some(p) = e.pos.filter(|p| p.is_finite()) {
        o.key("pos");
        o.0.push_str(&format!("[{},{},{}]", p.x.round() as i64, p.y.round() as i64, p.z.round() as i64));
    }
    o.str("parcel", e.parcel.as_deref());
    o.id("object", e.object);
    if let Some(face) = e.face {
        o.key("face");
        o.0.push_str(&face.to_string());
    }
    o.str("object_name", e.object_name.as_deref());
    o.id("speaker", e.speaker);
    o.str("speaker_name", e.speaker_name.as_deref());
    o.str("speaker_kind", e.speaker_kind);
    o.id("owner", e.owner);
    // last, so that the keys of version 1 keep their places
    o.key("v");
    o.0.push_str(&VERSION.to_string());
    o.0.push_str("}\n");
    o.0
}

/// The addresses already written this session, by what showed them.
struct Seen {
    hashes: HashSet<u64>,
    cap: usize,
}

impl Default for Seen {
    fn default() -> Self {
        Seen::with_cap(SEEN_CAP)
    }
}

impl Seen {
    fn with_cap(cap: usize) -> Self {
        Seen {
            hashes: HashSet::new(),
            cap,
        }
    }

    /// True the first time this address is seen for this key (since the
    /// set was last emptied).
    fn first(&mut self, key: Key, url: &str) -> bool {
        let mut h = DefaultHasher::new();
        (key, url).hash(&mut h);
        let v = h.finish();
        if self.hashes.len() >= self.cap && !self.hashes.contains(&v) {
            self.hashes.clear();
        }
        self.hashes.insert(v)
    }
}

/// The name of a day's file.
fn file_name(day: NaiveDate) -> String {
    format!("{}.jsonl", day.format("%Y-%m-%d"))
}

/// A file of the log (`YYYY-MM-DD.jsonl` exactly) whose day is more than
/// `KEEP_DAYS` before `today`. Any other name is never touched.
fn expired(name: &str, today: NaiveDate) -> bool {
    let Some(stem) = name.strip_suffix(".jsonl") else {
        return false;
    };
    let b = stem.as_bytes();
    let shape = b.len() == 10
        && b.iter()
            .enumerate()
            .all(|(i, c)| if i == 4 || i == 7 { *c == b'-' } else { c.is_ascii_digit() });
    if !shape {
        return false;
    }
    NaiveDate::parse_from_str(stem, "%Y-%m-%d").is_ok_and(|day| (today - day).num_days() > KEEP_DAYS)
}

// ------------------------------------------------------------ the writer

static ENABLED: AtomicBool = AtomicBool::new(false);
static DROPPED: AtomicU64 = AtomicU64::new(0);
/// Counts the times the log was turned on (`generation`).
static GENERATION: AtomicU64 = AtomicU64::new(0);

/// An object's name and owner, as an ObjectProperties answer gave them.
#[derive(Debug, Clone, Default)]
struct Known {
    name: String,
    owner: Uuid,
}

#[derive(Default)]
struct State {
    seen: Seen,
    known: HashMap<Uuid, Known>,
    tx: Option<Sender<(NaiveDate, String)>>,
}

static STATE: LazyLock<Mutex<State>> = LazyLock::new(|| Mutex::new(State::default()));

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn dir() -> PathBuf {
    let name = if std::env::var_os("AURORA_DEMO").is_some() {
        "url-log-demo"
    } else {
        "url-log"
    };
    crate::logging::logs_dir()
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default()
        .join(name)
}

/// The switch (media.url_log), followed every frame: the writer starts
/// the first time it is turned on.
pub fn follow(on: bool) {
    let was = ENABLED.swap(on, Ordering::Relaxed);
    if !on || was {
        return;
    }
    GENERATION.fetch_add(1, Ordering::Relaxed);
    let mut st = state();
    if st.tx.is_some() {
        return;
    }
    let dir = dir();
    let (tx, rx) = crossbeam_channel::bounded::<(NaiveDate, String)>(QUEUE_LINES);
    let thread_dir = dir.clone();
    match std::thread::Builder::new()
        .name("aurora-url-log".into())
        .spawn(move || run(&thread_dir, &rx))
    {
        Ok(_) => {
            log::info!("url log: on, writing to {}", dir.display());
            st.tx = Some(tx);
        }
        Err(e) => log::warn!("url log: no writer thread: {e}"),
    }
}

fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// 0 while the log is off, else a number that changes each time it is
/// turned on: a media that kept the number it was logged under is not
/// looked at again, one that did not (new, changed, or there before the
/// log was turned on) is given to `face`. Two atomic loads, no allocation.
pub fn generation() -> u64 {
    if enabled() { GENERATION.load(Ordering::Relaxed) } else { 0 }
}

/// True the first time (this session) for this key and address, while the
/// log is on.
fn first(key: Key, url: &str) -> bool {
    enabled() && !url.is_empty() && state().seen.first(key, url)
}

fn send(e: &Entry) {
    let job = (e.time.date_naive(), line(e));
    let st = state();
    let Some(tx) = st.tx.as_ref() else {
        return;
    };
    if let Err(TrySendError::Full(_)) = tx.try_send(job) {
        DROPPED.fetch_add(1, Ordering::Relaxed);
    }
}

fn run(dir: &Path, rx: &crossbeam_channel::Receiver<(NaiveDate, String)>) {
    let mut open: Option<(NaiveDate, std::fs::File)> = None;
    let mut swept: Option<NaiveDate> = None;
    let (mut written, mut failed) = (0u64, 0u64);
    loop {
        let today = Utc::now().date_naive();
        if swept != Some(today) {
            swept = Some(today);
            sweep(dir, today);
            log::debug!(
                "url log: {written} written, {failed} failed, {} dropped",
                DROPPED.load(Ordering::Relaxed)
            );
        }
        match rx.recv_timeout(TICK) {
            Ok((day, text)) => match write(dir, day, &text, &mut open) {
                Ok(()) => written += 1,
                Err(e) => {
                    if failed == 0 {
                        log::warn!("url log: cannot write: {e}");
                    }
                    failed += 1;
                    open = None;
                }
            },
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return,
        }
    }
}

fn refused(what: &str) -> std::io::Error {
    std::io::Error::other(format!("{what} is a link or not what it should be"))
}

/// The folder, made private (0700) if needed; never through a link.
fn ensure_dir(dir: &Path) -> std::io::Result<()> {
    match std::fs::symlink_metadata(dir) {
        Ok(m) if m.file_type().is_symlink() || !m.is_dir() => Err(refused("the url-log folder")),
        Ok(_) => private(dir, 0o700),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let mut b = std::fs::DirBuilder::new();
            b.recursive(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt;
                b.mode(0o700);
            }
            b.create(dir)?;
            private(dir, 0o700)
        }
        Err(e) => Err(e),
    }
}

#[cfg(unix)]
fn private(path: &Path, mode: u32) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))
}

#[cfg(not(unix))]
fn private(_path: &Path, _mode: u32) -> std::io::Result<()> {
    Ok(())
}

/// One line appended to its day's file in a single write, so a reader
/// never sees half of it.
fn write(dir: &Path, day: NaiveDate, text: &str, open: &mut Option<(NaiveDate, std::fs::File)>) -> std::io::Result<()> {
    if open.as_ref().is_none_or(|(d, _)| *d != day) {
        *open = None;
        ensure_dir(dir)?;
        let path = dir.join(file_name(day));
        if std::fs::symlink_metadata(&path).is_ok_and(|m| !m.file_type().is_file()) {
            return Err(refused("the day's file"));
        }
        let mut options = std::fs::OpenOptions::new();
        options.append(true).create(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options.open(&path)?;
        private(&path, 0o600)?;
        *open = Some((day, file));
    }
    match open.as_mut() {
        Some((_, file)) => file.write_all(text.as_bytes()),
        None => Ok(()),
    }
}

/// Deletes the files of the log more than `KEEP_DAYS` old: regular files
/// with the exact name pattern only, never through a link.
fn sweep(dir: &Path, today: NaiveDate) {
    if !std::fs::symlink_metadata(dir).is_ok_and(|m| m.is_dir()) {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if expired(name, today)
            && entry.file_type().is_ok_and(|t| t.is_file())
            && let Err(e) = std::fs::remove_file(entry.path())
        {
            log::warn!("url log: cannot delete an old file: {e}");
        }
    }
}

// ------------------------------------------------------------- the hooks

/// The media filter's answer for an address at the moment it is seen
/// (`MediaFilter::peek`): `allowed`, `held` (it asks first) or `blocked`.
/// With the filter off, only a refused scheme is blocked.
fn filter_state(world: &World, settings: &MediaSettings, url: &str, trusted: bool) -> &'static str {
    let verdict = if settings.filter {
        world.media_filter.peek(url, trusted)
    } else if crate::media::filter::refused_scheme(url).is_some() {
        Verdict::Deny
    } else {
        Verdict::Allow
    };
    match verdict {
        Verdict::Allow => "allowed",
        Verdict::Ask => "held",
        Verdict::Deny => "blocked",
    }
}

/// The region and position of an object: those of the top of its
/// linkset, or of the avatar wearing it (a worn item, a HUD).
fn place_of(world: &World, mut idx: usize) -> (Option<String>, Option<Vec3>) {
    for _ in 0..64 {
        let Some(o) = world.objects.get(idx) else {
            return (None, None);
        };
        if o.parent_id != 0
            && let Some(parent) = world.objects.parent_of(o)
        {
            idx = parent;
            continue;
        }
        let region = world.regions.get(&o.key.region).map(|r| r.name.clone());
        return (region, (o.parent_id == 0).then_some(o.position));
    }
    (None, None)
}

/// A media face, as `reconcile_prims` sees it.
pub struct Face {
    pub object: Uuid,
    /// Its index in `world.objects`.
    pub idx: usize,
    pub face: u8,
    pub hud: bool,
    /// Worn by us: the filter may trust it (filter_trust_own_worn).
    pub own: bool,
}

/// The names and owners of the ObjectProperties answers the viewer gets
/// anyway (hover tips, selection, Inspect, Area Search…): kept while the
/// log is on, for the name and owner of a media's linkset. The log asks
/// for none.
pub fn on_properties(props: &[ObjectProps]) {
    if !enabled() {
        return;
    }
    let mut st = state();
    for p in props.iter().filter(|p| !p.object_id.is_nil()) {
        if st.known.len() >= KNOWN_CAP && !st.known.contains_key(&p.object_id) {
            st.known.clear();
        }
        let known = Known {
            name: p.name.clone(),
            owner: p.owner_id,
        };
        st.known.insert(p.object_id, known);
    }
}

/// The root of an object's linkset and, for a worn item or a HUD, the
/// avatar wearing it.
fn root_of(world: &World, mut idx: usize) -> Option<(usize, Option<usize>)> {
    for _ in 0..64 {
        let o = world.objects.get(idx)?;
        if o.parent_id != 0
            && let Some(parent) = world.objects.parent_of(o)
        {
            if world.objects.get(parent).is_some_and(|p| p.is_avatar()) {
                return Some((idx, Some(parent)));
            }
            idx = parent;
            continue;
        }
        return Some((idx, None));
    }
    None
}

/// The name and owner of a media's linkset, from what the viewer already
/// knows, `None` for what it does not:
/// - the name of the root's ObjectProperties answer (`known`), else, for
///   an item we wear, the name of the inventory item it was worn from;
/// - the owner: the avatar wearing it (a worn item is its wearer's), else
///   the answer's owner, else the owner the object's update carried (the
///   region sends it with a sound or particles).
fn identity(known: Option<&Known>, wearer: Option<Uuid>, worn_name: Option<&str>, update_owner: Uuid) -> (Option<String>, Option<Uuid>) {
    let name = known
        .map(|k| k.name.as_str())
        .filter(|n| !n.is_empty())
        .or(worn_name.filter(|n| !n.is_empty()))
        .map(str::to_owned);
    let owner = [wearer, known.map(|k| k.owner), Some(update_owner)]
        .into_iter()
        .flatten()
        .find(|o| !o.is_nil());
    (name, owner)
}

/// `identity` of the linkset of the object at `idx`.
fn identity_of(world: &World, idx: usize) -> (Option<String>, Option<Uuid>) {
    let Some((root, wearer)) = root_of(world, idx) else {
        return (None, None);
    };
    let Some(o) = world.objects.get(root) else {
        return (None, None);
    };
    let wearer = wearer.and_then(|a| world.objects.get(a)).map(|a| a.full_id).filter(|a| !a.is_nil());
    let worn_name = (wearer.is_some() && wearer == Some(world.agent_id))
        .then(|| o.attachment_item_id())
        .flatten()
        .and_then(|item| world.inventory.items.get(&item))
        .map(|item| item.name.as_str());
    let known = state().known.get(&o.full_id).cloned();
    identity(known.as_ref(), wearer, worn_name, o.owner_id)
}

/// The line of a media face: `place` from `place_of`, `who` from
/// `identity`.
fn face_entry(
    f: &Face,
    url: &str,
    state: &'static str,
    place: (Option<String>, Option<Vec3>),
    who: (Option<String>, Option<Uuid>),
) -> Entry {
    let mut e = Entry::new(if f.hud { Source::Hud } else { Source::Prim }, url);
    e.state = Some(state);
    (e.region, e.pos) = place;
    e.object = Some(f.object);
    e.face = Some(f.face);
    (e.object_name, e.owner) = who;
    e
}

/// A media on a prim face, new, with a new address, or there when the log
/// was turned on (`generation`). Only called while prim media is on.
pub fn face(world: &World, settings: &MediaSettings, f: Face, url: &str) {
    if !first(Key::Face(f.object, f.face), url) {
        return;
    }
    let state = filter_state(world, settings, url, f.own && settings.filter_trust_own_worn);
    let e = face_entry(&f, url, state, place_of(world, f.idx), identity_of(world, f.idx));
    send(&e);
}

/// The line of a parcel's media or music: its region, our position, the
/// parcel's name and owner (a group for a group's land), when known.
fn parcel_entry_of(source: Source, url: &str, region: String, pos: Vec3, parcel: Option<&ParcelInfo>) -> Entry {
    let mut e = Entry::new(source, url);
    e.region = Some(region);
    e.pos = Some(pos);
    e.parcel = parcel.map(|p| p.name.clone());
    e.owner = parcel.map(|p| p.owner_id);
    e
}

/// The parcel we are on, with its region, its name and owner, and our
/// position.
fn parcel_entry(world: &World, source: Source, url: &str) -> Entry {
    parcel_entry_of(source, url, world.region_name(), world.agent.position, world.parcel.as_deref())
}

fn parcel_id(world: &World) -> i32 {
    world.parcel.as_ref().map_or(0, |p| p.local_id)
}

/// The parcel media's address as the media update reads it each frame
/// (`update_parcel`): written when it is first seen or changes while media
/// is on. With media off nothing is written nor taken as seen, so the
/// entry comes when media is turned on.
pub fn parcel_media(world: &World, settings: &MediaSettings, url: &str) {
    if !settings.enabled
        || world
            .parcel
            .as_ref()
            .is_some_and(|p| p.media.mime.eq_ignore_ascii_case("none/none"))
    {
        return;
    }
    if !first(Key::ParcelMedia(world.main_region, parcel_id(world)), url) {
        return;
    }
    let mut e = parcel_entry(world, Source::ParcelMedia, url);
    e.state = Some(filter_state(world, settings, url, false));
    send(&e);
}

/// The state of the parcel's music address `url`: `station` is what the
/// radio would play (`Radio::station`), `want` what it plays after the
/// filter, `verdict` the filter's answer for it. `None` while the music is
/// off (the radio stopped, the Music channel off, or a `/music` stream
/// playing instead): nothing is written then. `unsupported`: an address
/// the radio never plays (not starting with `http://` or `https://`),
/// written whatever the radio does.
fn music_state(url: &str, station: &str, want: &str, verdict: Verdict) -> Option<&'static str> {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Some("unsupported");
    }
    if station != url {
        return None;
    }
    if want == url {
        return Some("allowed");
    }
    match verdict {
        Verdict::Ask => Some("held"),
        Verdict::Deny => Some("blocked"),
        Verdict::Allow => None,
    }
}

/// The parcel music's address, each frame (`update_audio`): written when
/// the music is turned on (see `music_state`), once per session.
pub fn parcel_music(world: &World, station: &str, want: &str, verdict: Verdict) {
    if !enabled() {
        return;
    }
    let Some(url) = world.parcel.as_ref().map(|p| p.music_url.trim()).filter(|u| !u.is_empty()) else {
        return;
    };
    let Some(state) = music_state(url, station, want, verdict) else {
        return;
    };
    if !first(Key::ParcelMusic(world.main_region, parcel_id(world)), url) {
        return;
    }
    let mut e = parcel_entry(world, Source::ParcelMusic, url);
    e.state = Some(state);
    send(&e);
}

/// Who said a line.
struct Speaker<'a> {
    id: Uuid,
    name: &'a str,
    object: bool,
    owner: Uuid,
}

/// Where a speaker was: `at` the region and position the message gives,
/// when they fall in a region we know; else `own`, our own region, alone
/// (`None` where our region says nothing of the speaker, an avatar's IM).
fn speaker_place(at: Option<(&str, Option<Vec3>)>, own: Option<String>) -> (Option<String>, Option<Vec3>) {
    match at.filter(|(region, _)| !region.is_empty()) {
        Some((region, pos)) => (Some(region.to_owned()), pos.filter(|p| p.is_finite())),
        None => (own.filter(|r| !r.is_empty()), None),
    }
}

/// The line of an address said by `s`, at `place` (`speaker_place`).
fn said_entry(source: Source, url: &str, s: &Speaker<'_>, place: (Option<String>, Option<Vec3>)) -> Entry {
    let mut e = Entry::new(source, url);
    (e.region, e.pos) = place;
    e.speaker = Some(s.id);
    e.speaker_name = Some(s.name.to_owned());
    e.speaker_kind = Some(if s.object { "object" } else { "avatar" });
    e.owner = s.object.then_some(s.owner);
    e
}

/// `place` is only worked out when the line has an address to write.
fn said(agent: Uuid, source: Source, text: &str, s: Speaker<'_>, place: impl FnOnce() -> (Option<String>, Option<Vec3>)) {
    if !enabled() || s.id == agent || s.id.is_nil() {
        return;
    }
    let urls: Vec<String> = crate::ui::chat::web_addresses(text)
        .into_iter()
        .filter(|url| first(Key::Speaker(s.id), url))
        .collect();
    if urls.is_empty() {
        return;
    }
    let place = place();
    for url in urls {
        send(&said_entry(source, &url, &s, place.clone()));
    }
}

/// A line of nearby chat accepted into the conversation: those of
/// avatars and objects, not ours nor the system's. `at`: the region and
/// position of the speaker the message gives (ChatFromSimulator's
/// Position, in the region under it), `None` when no region we know holds
/// it; the line then names our own region only.
pub fn chat(world: &World, text: &str, kind: ChatKind, speaker: Uuid, name: &str, owner: Uuid, at: Option<(&str, Vec3)>) {
    let object = match kind {
        ChatKind::Local(_) => false,
        ChatKind::Object(_) => true,
        _ => return,
    };
    let s = Speaker {
        id: speaker,
        name,
        object,
        owner,
    };
    said(world.agent_id, Source::Chat, text, s, || {
        speaker_place(at.map(|(r, p)| (r, Some(p))), Some(world.region_name()))
    });
}

/// An IM accepted into its conversation (an avatar's) or into the nearby
/// chat (an object's, llInstantMessage); never a group's. `at`: the
/// location an object's IM carries (its region, and its position when
/// given); an avatar's IM has none.
pub fn im(agent: Uuid, text: &str, speaker: Uuid, name: &str, object: bool, owner: Uuid, at: Option<(&str, Option<Vec3>)>) {
    let s = Speaker {
        id: speaker,
        name,
        object,
        owner,
    };
    said(agent, Source::Im, text, s, || speaker_place(at, None));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-11-14T09:30:05.750Z")
            .expect("test date")
            .with_timezone(&Utc)
    }

    fn day(s: &str) -> NaiveDate {
        NaiveDate::parse_from_str(s, "%Y-%m-%d").expect("test day")
    }

    #[test]
    fn defang_marks_the_scheme_and_changes_nothing_else() {
        assert_eq!(defang("https://example.com/a?b=1"), "*https://example.com/a?b=1");
        assert_eq!(defang("http://example.com"), "*http://example.com");
        assert_eq!(defang("secondlife://Region/1/2/3"), "*secondlife://Region/1/2/3");
        assert_eq!(defang("rtsp://radio.example/live"), "*rtsp://radio.example/live");
        assert_eq!(defang("www.example.com/x"), "*www.example.com/x");
        assert_eq!(defang("*https://example.com"), "**https://example.com");
        assert_eq!(defang("https://example.com/a*b?c=*"), "*https://example.com/a*b?c=*");
        assert_eq!(defang(" https://example.com"), " *https://example.com");
        assert_eq!(defang(""), "*");
    }

    /// Lines of version 1, as the log wrote them before the version key:
    /// the same entries must give them back with only `"v":2` added last.
    const V1_MEDIA: &str = "{\"t\":\"2026-11-14T09:30:05Z\",\"source\":\"prim\",\"url\":\"*https://Media.Example.com/p?q=1#top\",\
         \"host\":\"media.example.com\",\"state\":\"held\",\"region\":\"Sandbox\",\"pos\":[12,201,23],\
         \"object\":\"00000000-0000-0000-0000-000000001234\",\"face\":3}\n";
    const V1_CHAT: &str = "{\"t\":\"2026-11-14T09:30:05Z\",\"source\":\"chat\",\"url\":\"*http://a.example/\",\"host\":\"a.example\",\
         \"speaker\":\"00000000-0000-0000-0000-000000000007\",\"speaker_kind\":\"avatar\"}\n";

    /// A version 1 line as version 2 writes it: `"v":2` before the end.
    fn as_v2(v1: &str) -> String {
        format!("{},\"v\":2}}\n", v1.strip_suffix("}\n").expect("a line"))
    }

    fn json(l: &str) -> serde_json::Value {
        assert_eq!(l.matches('\n').count(), 1, "{l}");
        serde_json::from_str(l.trim_end()).expect("valid JSON")
    }

    fn id(n: u128) -> Uuid {
        Uuid::from_u128(n)
    }

    fn a_face(hud: bool) -> Face {
        Face {
            object: id(0x1234),
            idx: 0,
            face: 3,
            hud,
            own: hud,
        }
    }

    fn a_speaker(object: bool) -> Speaker<'static> {
        Speaker {
            id: id(7),
            name: if object { "Vendor" } else { "Jane Doe" },
            object,
            owner: id(8),
        }
    }

    #[test]
    fn a_media_line_has_its_keys_in_order() {
        let mut e = Entry::new(Source::Prim, "https://Media.Example.com/p?q=1#top");
        e.time = at();
        e.state = Some("held");
        e.region = Some("Sandbox".into());
        e.pos = Some(Vec3::new(12.4, 200.6, 22.5));
        e.object = Some(Uuid::from_u128(0x1234));
        e.face = Some(3);
        assert_eq!(line(&e), as_v2(V1_MEDIA));
    }

    #[test]
    fn a_chat_line_leaves_out_what_does_not_apply() {
        let mut e = Entry::new(Source::Chat, "http://a.example/");
        e.time = at();
        e.speaker = Some(Uuid::from_u128(7));
        e.speaker_name = Some(String::new());
        e.speaker_kind = Some("avatar");
        e.owner = Some(Uuid::nil());
        e.pos = Some(Vec3::new(f32::NAN, 0.0, 0.0));
        assert_eq!(line(&e), as_v2(V1_CHAT));
        // no host when the address does not parse
        let e = Entry::new(Source::ParcelMusic, "not an address");
        assert!(!line(&e).contains("\"host\""));
    }

    #[test]
    fn every_key_of_version_1_keeps_its_name_and_value() {
        for (v1, l) in [
            (V1_MEDIA, {
                let place = (Some("Sandbox".to_owned()), Some(Vec3::new(12.4, 200.6, 22.5)));
                let who = identity(None, None, None, Uuid::nil());
                let mut e = face_entry(&a_face(false), "https://Media.Example.com/p?q=1#top", "held", place, who);
                e.time = at();
                line(&e)
            }),
            (V1_CHAT, {
                let s = Speaker {
                    name: "",
                    ..a_speaker(false)
                };
                let mut e = said_entry(Source::Chat, "http://a.example/", &s, speaker_place(None, None));
                e.time = at();
                line(&e)
            }),
        ] {
            let (old, new) = (json(v1), json(&l));
            let (old, new) = (old.as_object().expect("object"), new.as_object().expect("object"));
            for (k, v) in old {
                assert_eq!(new.get(k), Some(v), "{k}");
            }
            assert!(l.starts_with("{\"t\":\""));
        }
    }

    #[test]
    fn every_line_ends_with_the_version() {
        for source in [
            Source::Prim,
            Source::Hud,
            Source::ParcelMedia,
            Source::ParcelMusic,
            Source::Chat,
            Source::Im,
        ] {
            let l = line(&Entry::new(source, "https://a.example/"));
            assert!(l.ends_with(",\"v\":2}\n"), "{l}");
            assert_eq!(json(&l)["v"], 2);
        }
    }

    #[test]
    fn a_prim_line_names_its_linkset_and_owner_when_known() {
        let known = Known {
            name: "Club screen".into(),
            owner: id(0x0a),
        };
        let place = (Some("Sandbox".to_owned()), Some(Vec3::new(10.0, 20.0, 30.0)));
        let who = identity(Some(&known), None, None, Uuid::nil());
        let mut e = face_entry(&a_face(false), "https://tv.example/", "allowed", place.clone(), who);
        e.time = at();
        assert_eq!(
            line(&e),
            "{\"t\":\"2026-11-14T09:30:05Z\",\"source\":\"prim\",\"url\":\"*https://tv.example/\",\"host\":\"tv.example\",\
             \"state\":\"allowed\",\"region\":\"Sandbox\",\"pos\":[10,20,30],\"object\":\"00000000-0000-0000-0000-000000001234\",\
             \"face\":3,\"object_name\":\"Club screen\",\"owner\":\"00000000-0000-0000-0000-00000000000a\",\"v\":2}\n"
        );
        // only the owner its update carried (a sound, particles)
        let v = json(&line(&face_entry(
            &a_face(false),
            "https://tv.example/",
            "allowed",
            place.clone(),
            identity(None, None, None, id(0x0b)),
        )));
        assert_eq!(v["owner"], "00000000-0000-0000-0000-00000000000b");
        assert!(v.get("object_name").is_none());
        // nothing known: both left out
        let v = json(&line(&face_entry(
            &a_face(false),
            "https://tv.example/",
            "allowed",
            place,
            identity(None, None, None, Uuid::nil()),
        )));
        assert!(v.get("object_name").is_none() && v.get("owner").is_none());
    }

    #[test]
    fn a_hud_line_is_its_wearers_with_the_items_name() {
        let me = id(0x99);
        let place = (Some("Sandbox".to_owned()), Some(Vec3::new(1.0, 2.0, 3.0)));
        let v = json(&line(&face_entry(
            &a_face(true),
            "https://hud.example/",
            "allowed",
            place,
            identity(None, Some(me), Some("Radio HUD"), Uuid::nil()),
        )));
        assert_eq!(v["source"], "hud");
        assert_eq!(v["object_name"], "Radio HUD");
        assert_eq!(v["owner"], "00000000-0000-0000-0000-000000000099");
        assert_eq!(v["v"], 2);
    }

    #[test]
    fn the_name_and_owner_follow_what_is_surest() {
        let known = Known {
            name: "Prim name".into(),
            owner: id(2),
        };
        // the answer's name before the item's, the wearer before any owner
        assert_eq!(
            identity(Some(&known), Some(id(1)), Some("Item name"), id(3)),
            (Some("Prim name".into()), Some(id(1)))
        );
        assert_eq!(identity(Some(&known), None, None, id(3)), (Some("Prim name".into()), Some(id(2))));
        let unnamed = Known {
            name: String::new(),
            owner: Uuid::nil(),
        };
        assert_eq!(
            identity(Some(&unnamed), None, Some("Item name"), id(3)),
            (Some("Item name".into()), Some(id(3)))
        );
        assert_eq!(identity(None, None, Some(""), Uuid::nil()), (None, None));
    }

    #[test]
    fn a_chat_line_says_where_the_speaker_was() {
        let pos = Vec3::new(100.4, 50.6, 21.0);
        // an object we know of, then an avatar: the message's place
        for object in [true, false] {
            let place = speaker_place(Some(("Sandbox", Some(pos))), Some("Home".into()));
            let v = json(&line(&said_entry(Source::Chat, "https://a.example/", &a_speaker(object), place)));
            assert_eq!(v["region"], "Sandbox");
            assert_eq!(v["pos"], serde_json::json!([100, 51, 21]));
            assert_eq!(v["speaker_kind"], if object { "object" } else { "avatar" });
            assert_eq!(v.get("owner").is_some(), object);
        }
        // no region we know under it: our own region alone
        let v = json(&line(&said_entry(
            Source::Chat,
            "https://a.example/",
            &a_speaker(true),
            speaker_place(None, Some("Home".into())),
        )));
        assert_eq!(v["region"], "Home");
        assert!(v.get("pos").is_none());
        let v = json(&line(&said_entry(
            Source::Chat,
            "https://a.example/",
            &a_speaker(false),
            speaker_place(Some(("", Some(pos))), Some("Home".into())),
        )));
        assert_eq!(v["region"], "Home");
        assert!(v.get("pos").is_none());
    }

    #[test]
    fn an_im_line_has_the_place_an_object_gives() {
        let v = json(&line(&said_entry(
            Source::Im,
            "https://a.example/",
            &a_speaker(true),
            speaker_place(Some(("Far Away", None)), None),
        )));
        assert_eq!(v["region"], "Far Away");
        assert!(v.get("pos").is_none());
        let place = speaker_place(Some(("Far Away", Some(Vec3::new(1.0, 2.0, 3.0)))), None);
        let v = json(&line(&said_entry(Source::Im, "https://a.example/", &a_speaker(true), place)));
        assert_eq!(v["pos"], serde_json::json!([1, 2, 3]));
        // an avatar's IM: no place
        let v = json(&line(&said_entry(
            Source::Im,
            "https://a.example/",
            &a_speaker(false),
            speaker_place(None, None),
        )));
        assert!(v.get("region").is_none() && v.get("pos").is_none());
    }

    #[test]
    fn a_parcel_line_has_the_parcels_owner_when_known() {
        let parcel = ParcelInfo {
            name: "Beach".into(),
            owner_id: id(0x42),
            ..Default::default()
        };
        for source in [Source::ParcelMedia, Source::ParcelMusic] {
            let v = json(&line(&parcel_entry_of(
                source,
                "https://radio.example/",
                "Sandbox".into(),
                Vec3::ONE,
                Some(&parcel),
            )));
            assert_eq!(v["parcel"], "Beach");
            assert_eq!(v["owner"], "00000000-0000-0000-0000-000000000042");
            assert_eq!(v["v"], 2);
        }
        let unknown = ParcelInfo {
            name: "Beach".into(),
            ..Default::default()
        };
        let v = json(&line(&parcel_entry_of(
            Source::ParcelMedia,
            "https://a.example/",
            "Sandbox".into(),
            Vec3::ONE,
            Some(&unknown),
        )));
        assert!(v.get("owner").is_none());
        let v = json(&line(&parcel_entry_of(
            Source::ParcelMedia,
            "https://a.example/",
            "Sandbox".into(),
            Vec3::ONE,
            None,
        )));
        assert!(v.get("owner").is_none() && v.get("parcel").is_none());
        assert_eq!(v["region"], "Sandbox");
    }

    #[test]
    fn the_known_objects_are_kept_only_while_the_log_is_on() {
        // the switch is process-wide; no test turns it on
        let p = ObjectProps {
            object_id: id(0x77),
            name: "Screen".into(),
            owner_id: id(0x78),
            ..Default::default()
        };
        assert!(!enabled());
        on_properties(std::slice::from_ref(&p));
        assert!(!state().known.contains_key(&p.object_id));
    }

    #[test]
    fn names_and_addresses_are_escaped_on_one_line() {
        let mut e = Entry::new(Source::Im, "https://a.example/\"x\"\n\\y");
        e.time = at();
        e.speaker = Some(Uuid::from_u128(1));
        e.speaker_name = Some("Bad \"Name\"\nline\u{1}\ttab".into());
        e.speaker_kind = Some("object");
        e.owner = Some(Uuid::from_u128(2));
        let l = line(&e);
        assert_eq!(l.matches('\n').count(), 1);
        assert!(l.ends_with("}\n"));
        assert!(l.contains("\"url\":\"*https://a.example/\\\"x\\\"\\n\\\\y\""));
        assert!(l.contains("\"speaker_name\":\"Bad \\\"Name\\\"\\nline\\u0001\\ttab\""));
        let back: serde_json::Value = serde_json::from_str(l.trim_end()).expect("valid JSON");
        assert_eq!(back["speaker_name"], "Bad \"Name\"\nline\u{1}\ttab");
        assert_eq!(back["owner"], "00000000-0000-0000-0000-000000000002");
    }

    #[test]
    fn an_address_is_written_once_per_key() {
        let mut seen = Seen::default();
        let a = Uuid::from_u128(1);
        assert!(seen.first(Key::Face(a, 0), "https://a.example/"));
        assert!(!seen.first(Key::Face(a, 0), "https://a.example/"));
        assert!(seen.first(Key::Face(a, 1), "https://a.example/"));
        assert!(seen.first(Key::Face(a, 0), "https://b.example/"));
        assert!(seen.first(Key::Speaker(a), "https://a.example/"));
        assert!(seen.first(Key::ParcelMedia(Some(5), 2), "https://a.example/"));
        assert!(seen.first(Key::ParcelMusic(Some(5), 2), "https://a.example/"));
        assert!(!seen.first(Key::ParcelMusic(Some(5), 2), "https://a.example/"));
        assert!(seen.first(Key::ParcelMusic(Some(5), 3), "https://a.example/"));
    }

    #[test]
    fn the_once_per_session_set_is_bounded() {
        let mut seen = Seen::with_cap(2);
        let a = Uuid::from_u128(1);
        assert!(seen.first(Key::Speaker(a), "1"));
        assert!(seen.first(Key::Speaker(a), "2"));
        // full: a known address is still known
        assert!(!seen.first(Key::Speaker(a), "1"));
        // a new one empties the set first
        assert!(seen.first(Key::Speaker(a), "3"));
        assert_eq!(seen.hashes.len(), 1);
        assert!(seen.first(Key::Speaker(a), "1"));
    }

    #[test]
    fn file_names_and_the_thirty_days() {
        let today = day("2026-11-14");
        assert_eq!(file_name(today), "2026-11-14.jsonl");
        assert_eq!(file_name(day("2027-01-02")), "2027-01-02.jsonl");
        assert!(expired("2026-10-14.jsonl", today), "31 days");
        assert!(!expired("2026-10-15.jsonl", today), "30 days are kept");
        assert!(!expired("2026-11-14.jsonl", today));
        assert!(!expired("2026-12-01.jsonl", today), "a day to come");
        assert!(expired("2025-01-01.jsonl", today));
        for other in [
            "notes.jsonl",
            "2026-01-01.jsonl.bak",
            "2026-01-01.json",
            "2026-1-01.jsonl",
            "x2026-01-01.jsonl",
            "2026-01-01x.jsonl",
            "2026_01_01.jsonl",
            "2026-13-01.jsonl",
            ".jsonl",
            "2026-01-01",
        ] {
            assert!(!expired(other, today), "{other}");
        }
    }

    #[test]
    fn parcel_music_is_written_only_once_on() {
        let u = "http://radio.example/live";
        // off: the radio stopped or the channel off, or another stream
        assert_eq!(music_state(u, "", "", Verdict::Allow), None);
        assert_eq!(music_state(u, "https://other.example/", "", Verdict::Allow), None);
        // on: the filter's answer
        assert_eq!(music_state(u, u, u, Verdict::Allow), Some("allowed"));
        assert_eq!(music_state(u, u, "", Verdict::Ask), Some("held"));
        assert_eq!(music_state(u, u, "", Verdict::Deny), Some("blocked"));
        // never played: written anyway
        assert_eq!(music_state("rtsp://radio.example/", "", "", Verdict::Allow), Some("unsupported"));
        assert_eq!(music_state("HTTP://radio.example/", "", "", Verdict::Allow), Some("unsupported"));
    }

    #[test]
    fn web_addresses_of_a_chat_line() {
        let w = crate::ui::chat::web_addresses;
        assert_eq!(
            w("see https://a.example/x?y=1#z, and http://b.example."),
            ["https://a.example/x?y=1#z", "http://b.example"]
        );
        assert_eq!(w("[https://c.example/p?q=1 the shop] is open"), ["https://c.example/p?q=1"]);
        assert_eq!(w("(https://d.example/)!"), ["https://d.example/"]);
        assert!(w("secondlife:///app/agent/00000000-0000-0000-0000-000000000001/about").is_empty());
        assert!(w("ftp://e.example/ www.f.example hop://grid/Region/1/2/3").is_empty());
        assert!(w("no link here").is_empty());
    }
}
