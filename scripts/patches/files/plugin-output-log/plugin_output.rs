// aurora-local-patch: plugin-output-module
//! The media plugins' standard output and error, in the viewer's log with
//! repeats folded (local patch set, 79-plugin-output-log).
//!
//! Upstream starts SLPlugin (and the processes its engine starts below it)
//! with the viewer's own stdout and stderr. An engine that prints an error
//! for every video frame then floods whatever reads the viewer's output (a
//! terminal, the system log) with several lines per frame. Here each plugin
//! process gets one pipe for both streams, drained by a thread of its own
//! into the viewer's log under the target [`TARGET`]:
//!
//! - the first occurrence of a line is logged in full; the same line again
//!   (Chromium's varying `[pid:tid:time:` start ignored, its level, file and
//!   line kept) is only counted, and the count is logged once a minute
//!   ("repeated N times in the last M s"), when the table must make room,
//!   and when the plugin's pipe closes;
//! - bounded: [`TABLE_SIZE`] lines remembered (the least recently seen
//!   evicted), [`MAX_LINE`] bytes a line (the rest skipped, the read goes
//!   on), invalid UTF-8 and control characters replaced, and at most
//!   [`RATE`] lines a second written by all plugins together (a burst of
//!   [`BURST`] allowed, then "N lines dropped");
//! - logged at `info`, never as a warning or an error, so they stay out of
//!   the Dev Toolbox's warning lists; the crash and bug reports skip them
//!   ([`is_plugin_line`]): page addresses and console text never reach a
//!   report.
//!
//! The pipe: the thread is started before the plugin, so a thread that
//! cannot start leaves the plugin on the viewer's output as upstream. Only
//! the plugin and its descendants hold the write end once the `Command` is
//! dropped (std opens pipes close-on-exec, the child's stdout and stderr
//! are copies). The thread reads to the end, never stops early (a full pipe
//! would freeze the plugin), is never joined (an engine's helper processes
//! may keep the pipe open after SLPlugin has ended) and ends at the end of
//! the pipe; a viewer that exits does not wait for it.
//!
//! `AURORA_PLUGIN_OUTPUT=inherit` keeps upstream's behaviour: no pipe, no
//! thread. Unix only; elsewhere the plugins keep the viewer's output, as
//! upstream (a Windows plugin has no console to print to).

use std::collections::VecDeque;
use std::io::Read;
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, LazyLock, Mutex, PoisonError};
use std::time::{Duration, Instant};

/// The log target of the plugins' lines.
pub const TARGET: &str = "aurora_media::plugin";
/// The switch back to upstream's inherited output.
pub const VARIABLE: &str = "AURORA_PLUGIN_OUTPUT";
/// Longest line kept, in bytes.
const MAX_LINE: usize = 512;
/// Lines remembered for their repeats.
const TABLE_SIZE: usize = 64;
/// How often a line's repeats are reported.
const PERIOD: Duration = Duration::from_secs(60);
/// Lines a second written to the log by all plugins together, after a burst.
const RATE: f64 = 2.0;
const BURST: f64 = 120.0;
/// Characters of a line quoted in its repeat count.
const SHOWN: usize = 160;
/// Size of one read from the pipe.
const READ_CHUNK: usize = 8192;

static SHARED: LazyLock<Arc<Mutex<Collapser>>> = LazyLock::new(|| Arc::new(Mutex::new(Collapser::new(Instant::now()))));

/// Give `cmd`, about to start the plugin `plugin` (a library SLPlugin loads,
/// or a program), a pipe for its stdout and stderr, drained into the log.
/// Nothing changes with `AURORA_PLUGIN_OUTPUT=inherit`, off Unix, or when
/// the pipe or its thread cannot be made.
pub fn attach(cmd: &mut Command, plugin: &Path) {
    if !cfg!(unix) || !piped(std::env::var_os(VARIABLE).as_deref()) {
        return;
    }
    let label = label_of(plugin);
    if let Err(e) = connect(cmd, label, Arc::clone(&SHARED), |line| log::info!(target: TARGET, "{line}")) {
        log::warn!("media: the plugin keeps the viewer's output: {e}");
    }
}

/// Whether the switch asks for the pipe (anything but `inherit`).
fn piped(value: Option<&std::ffi::OsStr>) -> bool {
    !value.is_some_and(|v| v.to_string_lossy().trim().eq_ignore_ascii_case("inherit"))
}

/// The plugin's name in its lines: the file's stem without `lib`.
fn label_of(plugin: &Path) -> String {
    let stem = plugin.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    match stem.strip_prefix("lib") {
        Some(rest) if !rest.is_empty() => rest.to_owned(),
        _ => stem,
    }
}

/// The pipe and its thread; `cmd` gets the write end for both streams.
fn connect(
    cmd: &mut Command,
    label: String,
    table: Arc<Mutex<Collapser>>,
    mut emit: impl FnMut(String) + Send + 'static,
) -> std::io::Result<std::thread::JoinHandle<()>> {
    let (reader, writer) = std::io::pipe()?;
    let second = writer.try_clone()?;
    let thread = std::thread::Builder::new()
        .name("media plugin output".into())
        .spawn(move || drain(reader, &label, &table, &mut emit))?;
    cmd.stdout(writer).stderr(second);
    Ok(thread)
}

/// Read `reader` to its end, a line at a time, through the shared table.
fn drain(mut reader: impl Read, label: &str, table: &Mutex<Collapser>, emit: &mut dyn FnMut(String)) {
    let mut chunk = vec![0u8; READ_CHUNK];
    let mut line: Vec<u8> = Vec::with_capacity(MAX_LINE);
    let mut cut = false;
    loop {
        let n = match reader.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => {
                log::debug!("media: the output of {label} is no longer read: {e}");
                break;
            }
        };
        for piece in chunk[..n].split_inclusive(|&b| b == b'\n') {
            let (bytes, ends) = match piece.strip_suffix(b"\n") {
                Some(bytes) => (bytes, true),
                None => (piece, false),
            };
            let room = MAX_LINE - line.len();
            if bytes.len() > room {
                line.extend_from_slice(&bytes[..room]);
                cut = true;
            } else {
                line.extend_from_slice(bytes);
            }
            if ends {
                let out = lock(table).line(Instant::now(), label, &line, cut);
                out.into_iter().for_each(&mut *emit);
                line.clear();
                cut = false;
            }
        }
    }
    let mut table = lock(table);
    let mut out = if line.is_empty() {
        Vec::new()
    } else {
        table.line(Instant::now(), label, &line, cut)
    };
    out.extend(table.flush(Instant::now(), label));
    drop(table);
    out.into_iter().for_each(emit);
}

fn lock(table: &Mutex<Collapser>) -> std::sync::MutexGuard<'_, Collapser> {
    table.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A line of the viewer's log written for a plugin. The crash and bug
/// reports leave these out. The viewer's logger is env_logger's default
/// format with millisecond timestamps (aurora-viewer logging.rs `init`):
/// `[<time> <LEVEL padded to 5> <target>] <message>`, the target always
/// written for these records. Colour codes, which RUST_LOG_STYLE=always can
/// put around the brackets and the level, are skipped; a plugin's own text
/// holds none (control characters are replaced).
pub fn is_plugin_line(line: &str) -> bool {
    let plain;
    let line = if line.contains('\u{1b}') {
        plain = without_colours(line);
        plain.as_str()
    } else {
        line
    };
    line.strip_prefix('[')
        .and_then(|rest| rest.split_once("] "))
        .is_some_and(|(header, _)| header.split_whitespace().nth(2) == Some(TARGET))
}

/// `line` without its ANSI CSI sequences (`ESC [ … final byte`).
fn without_colours(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c != '\u{1b}' {
            out.push(c);
            continue;
        }
        if chars.next() == Some('[') {
            for c in chars.by_ref() {
                if ('\u{40}'..='\u{7e}').contains(&c) {
                    break;
                }
            }
        }
    }
    out
}

/// `text` without the plugins' lines.
pub fn without_plugin_lines(text: &str) -> String {
    text.split_inclusive('\n').filter(|line| !is_plugin_line(line)).collect()
}

struct Entry {
    key: String,
    label: String,
    text: String,
    since: Instant,
    last: Instant,
    repeats: u64,
}

/// The repeats of the plugins' lines and the rate of what is written: pure
/// logic, the time given by the caller. Returns the lines to log.
pub(crate) struct Collapser {
    /// The least recently seen first.
    entries: VecDeque<Entry>,
    tokens: f64,
    refilled: Instant,
    dropped: u64,
}

impl Collapser {
    pub(crate) fn new(now: Instant) -> Collapser {
        Collapser {
            entries: VecDeque::with_capacity(TABLE_SIZE),
            tokens: BURST,
            refilled: now,
            dropped: 0,
        }
    }

    /// One line of `label`'s output (`cut`: longer than [`MAX_LINE`]).
    pub(crate) fn line(&mut self, now: Instant, label: &str, raw: &[u8], cut: bool) -> Vec<String> {
        let mut out = Vec::new();
        self.refill(now);
        self.report_dropped(&mut out);
        self.sweep(now, &mut out);
        let text = clean(raw, cut);
        if text.is_empty() {
            return out;
        }
        let key = format!("{label}\u{1f}{}", key_of(&text));
        if let Some(at) = self.entries.iter().position(|e| e.key == key)
            && let Some(mut entry) = self.entries.remove(at)
        {
            entry.repeats += 1;
            entry.last = now;
            self.entries.push_back(entry);
            return out;
        }
        if self.entries.len() >= TABLE_SIZE
            && let Some(old) = self.entries.pop_front()
            && old.repeats > 0
        {
            let summary = summary(&old, now);
            self.admit(summary, &mut out);
        }
        self.admit(format!("{label}: {text}"), &mut out);
        self.entries.push_back(Entry {
            key,
            label: label.to_owned(),
            text,
            since: now,
            last: now,
            repeats: 0,
        });
        out
    }

    /// The repeats not yet reported of `label`'s lines (its pipe closed).
    pub(crate) fn flush(&mut self, now: Instant, label: &str) -> Vec<String> {
        let mut out = Vec::new();
        self.refill(now);
        self.report_dropped(&mut out);
        let mut summaries = Vec::new();
        self.entries.retain(|e| {
            if e.label != label {
                return true;
            }
            if e.repeats > 0 {
                summaries.push(summary(e, now));
            }
            false
        });
        for summary in summaries {
            self.admit(summary, &mut out);
        }
        out
    }

    /// Report the repeats of a period gone by; forget lines not seen for a
    /// period, so that they are shown in full when they come back.
    fn sweep(&mut self, now: Instant, out: &mut Vec<String>) {
        let mut summaries = Vec::new();
        self.entries.retain_mut(|e| {
            if now.saturating_duration_since(e.since) < PERIOD {
                return true;
            }
            if e.repeats > 0 {
                summaries.push(summary(e, now));
                e.repeats = 0;
                e.since = now;
                return true;
            }
            now.saturating_duration_since(e.last) < PERIOD
        });
        for summary in summaries {
            self.admit(summary, out);
        }
    }

    fn refill(&mut self, now: Instant) {
        let gained = now.saturating_duration_since(self.refilled).as_secs_f64() * RATE;
        self.tokens = (self.tokens + gained).min(BURST);
        self.refilled = now;
    }

    fn admit(&mut self, line: String, out: &mut Vec<String>) {
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            out.push(line);
        } else {
            self.dropped += 1;
        }
    }

    fn report_dropped(&mut self, out: &mut Vec<String>) {
        if self.dropped > 0 && self.tokens >= 1.0 {
            self.tokens -= 1.0;
            out.push(format!("media plugins: {} lines dropped (more than {RATE} a second)", self.dropped));
            self.dropped = 0;
        }
    }
}

fn summary(entry: &Entry, now: Instant) -> String {
    let seconds = now.saturating_duration_since(entry.since).as_secs().max(1);
    let shown: String = entry.text.chars().take(SHOWN).collect();
    let more = if shown.len() < entry.text.len() { "…" } else { "" };
    format!(
        "{}: repeated {} times in the last {seconds} s: {shown}{more}",
        entry.label, entry.repeats
    )
}

/// The line as text: invalid UTF-8 and control characters replaced, the
/// ends trimmed, `…` when it was cut.
fn clean(raw: &[u8], cut: bool) -> String {
    let lossy = String::from_utf8_lossy(raw);
    let mut text: String = lossy
        .trim()
        .chars()
        .map(|c| match c {
            '\t' => ' ',
            c if c.is_control() => '\u{fffd}',
            c => c,
        })
        .collect();
    if cut && !text.is_empty() {
        text.push('…');
    }
    text
}

/// What stays the same between repeats: Chromium's log lines start with
/// `[pid:tid:MMDD/HHMMSS.micros:LEVEL:file(line)]` (or `file:line`, with
/// some of the first fields left out); the fields before the level are
/// dropped. Any other line is its own key.
fn key_of(text: &str) -> &str {
    let Some(inside) = text
        .strip_prefix('[')
        .and_then(|rest| rest.split_once(']'))
        .map(|(inside, _)| inside)
    else {
        return text;
    };
    let mut offset = 0;
    for part in inside.split(':') {
        if is_level(part) {
            // text[0] is '[': the key starts at the level
            return if offset > 0 { &text[1 + offset..] } else { text };
        }
        if part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit() || b == b'/' || b == b'.') {
            break;
        }
        offset += part.len() + 1;
    }
    text
}

fn is_level(part: &str) -> bool {
    matches!(part, "FATAL" | "ERROR" | "WARNING" | "INFO" | "VERBOSE")
        || part
            .strip_prefix("VERBOSE")
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The four lines a video frame printed, once per frame, on a driver
    /// that refuses to map the engine's frame buffers.
    const FRAME: [&str; 4] = [
        "Failed to mmap dmabuf: 無効な引数です (22)",
        "Failed to map the buffer.",
        "ScopedMapping init failed.",
        "[12345:67:1126/195100.123456:ERROR:gpu/command_buffer/client/client_shared_image.cc:577] Unable to create ScopedMapping.   shared_image Info: { format: (Y_UV, 420, 8unorm, ExtSamplerOn) usage: Gles2Read|RasterRead|DisplayRead|CpuWriteOnly|CpuRead label: MediaGmbVideoFramePoolMappableSI }",
    ];

    fn frame_line(i: usize, frame: usize) -> String {
        if i < 3 || frame == 0 {
            return FRAME[i].to_owned();
        }
        // the start Chromium varies at every line: pid, thread, time
        FRAME[3].replace(
            "12345:67:1126/195100.123456",
            &format!(
                "{}:{}:1126/1951{:02}.{:06}",
                12345 + frame % 3,
                67 + frame % 5,
                frame % 60,
                frame * 43
            ),
        )
    }

    #[test]
    fn the_varying_start_of_a_chromium_line_is_not_part_of_its_key() {
        assert_eq!(
            key_of(&frame_line(3, 1)),
            "ERROR:gpu/command_buffer/client/client_shared_image.cc:577] Unable to create ScopedMapping.   shared_image Info: { format: (Y_UV, 420, 8unorm, ExtSamplerOn) usage: Gles2Read|RasterRead|DisplayRead|CpuWriteOnly|CpuRead label: MediaGmbVideoFramePoolMappableSI }"
        );
        assert_eq!(key_of(&frame_line(3, 1)), key_of(&frame_line(3, 977)));
        // the older form, with fewer fields and file(line)
        assert_eq!(key_of("[1126/195100.1:WARNING:foo.cc(12)] x"), "WARNING:foo.cc(12)] x");
        assert_eq!(key_of("[0102/030405.6:VERBOSE1:a.cc(1)] y"), "VERBOSE1:a.cc(1)] y");
        // other lines are their own key
        for text in FRAME
            .iter()
            .take(3)
            .copied()
            .chain(["[ERROR:x.cc(1)] z", "[abc:ERROR:x] z", "[1:2:3] w", "[", "]"])
        {
            assert_eq!(key_of(text), text);
        }
        assert_ne!(
            key_of("[1:1:0101/000000.0:ERROR:a.cc:1] one"),
            key_of("[1:1:0101/000000.0:ERROR:a.cc:2] one")
        );
    }

    #[test]
    fn a_video_flood_becomes_four_lines_then_a_count_a_minute() {
        let start = Instant::now();
        let mut c = Collapser::new(start);
        let mut out = Vec::new();
        // 23 frames a second for two minutes and a half
        let frames = 23 * 150;
        for frame in 0..frames {
            let now = start + Duration::from_millis(frame as u64 * 1000 / 23);
            for i in 0..4 {
                out.extend(c.line(now, "media_plugin_cef", frame_line(i, frame).as_bytes(), false));
            }
        }
        out.extend(c.flush(start + Duration::from_secs(150), "media_plugin_cef"));
        // the four lines in full, the non-ASCII text intact
        assert_eq!(out[0], format!("media_plugin_cef: {}", FRAME[0]));
        assert_eq!(out[1], format!("media_plugin_cef: {}", FRAME[1]));
        assert_eq!(out[2], format!("media_plugin_cef: {}", FRAME[2]));
        assert_eq!(out[3], format!("media_plugin_cef: {}", FRAME[3]));
        // then a count for each, at 60 s and 120 s, and the rest at the end
        let counts: Vec<&String> = out[4..].iter().collect();
        assert_eq!(counts.len(), 12, "{counts:#?}");
        assert!(counts.iter().all(|l| l.starts_with("media_plugin_cef: repeated ")), "{counts:#?}");
        assert!(
            counts[0].contains("in the last 60 s: Failed to mmap dmabuf: 無効な引数です (22)"),
            "{}",
            counts[0]
        );
        assert!(counts[3].contains(" s: [12345:67:1126/195100.123456:ERROR:"), "{}", counts[3]);
        assert!(counts[3].ends_with('…'), "{}", counts[3]);
        let total: u64 = counts
            .iter()
            .map(|l| {
                l["media_plugin_cef: repeated ".len()..]
                    .split(' ')
                    .next()
                    .unwrap()
                    .parse::<u64>()
                    .unwrap()
            })
            .sum();
        assert_eq!(total, 4 * (frames as u64 - 1), "every repeat counted once");
        assert!(c.entries.is_empty(), "the closed pipe's lines are forgotten");
    }

    #[test]
    fn a_line_unseen_for_a_period_is_shown_in_full_again() {
        let start = Instant::now();
        let mut c = Collapser::new(start);
        assert_eq!(c.line(start, "p", b"hello", false), ["p: hello"]);
        assert!(c.line(start + Duration::from_secs(1), "p", b"hello", false).is_empty());
        // the count, then a quiet period, then the line again
        let at = start + Duration::from_secs(61);
        assert_eq!(
            c.line(at, "p", b"other", false),
            ["p: repeated 1 times in the last 61 s: hello", "p: other"]
        );
        let later = at + Duration::from_secs(61);
        let out = c.line(later, "p", b"hello", false);
        assert_eq!(out, ["p: hello"]);
        // another plugin's same line is its own
        assert_eq!(c.line(later, "q", b"hello", false), ["q: hello"]);
    }

    #[test]
    fn the_table_is_bounded_and_an_evicted_line_reports_its_repeats() {
        let start = Instant::now();
        let mut c = Collapser::new(start);
        c.line(start, "p", b"first", false);
        c.line(start, "p", b"first", false);
        let mut out = Vec::new();
        for i in 0..TABLE_SIZE * 3 {
            out.extend(c.line(start, "p", format!("unique {i}").as_bytes(), false));
            assert!(c.entries.len() <= TABLE_SIZE);
        }
        assert!(out.contains(&"p: repeated 1 times in the last 1 s: first".to_owned()), "{out:?}");
    }

    #[test]
    fn the_rate_is_capped_for_all_plugins_and_the_drops_are_counted() {
        let start = Instant::now();
        let mut c = Collapser::new(start);
        let mut out = Vec::new();
        for i in 0..1000 {
            let label = if i % 2 == 0 { "a" } else { "b" };
            out.extend(c.line(start, label, format!("line {i}").as_bytes(), false));
        }
        assert_eq!(out.len(), BURST as usize);
        // ten seconds later: the drops first, then the new line
        let out = c.line(start + Duration::from_secs(10), "a", b"after", false);
        assert_eq!(
            out,
            [
                format!("media plugins: {} lines dropped (more than 2 a second)", 1000 - BURST as usize),
                "a: after".to_owned()
            ]
        );
        // a steady flood of new lines writes RATE lines a second
        let mut written = 0;
        for i in 0..10_000u64 {
            let now = start + Duration::from_secs(20) + Duration::from_millis(i);
            written += c.line(now, "a", format!("flood {i}").as_bytes(), false).len();
        }
        assert!(written <= BURST as usize + 2 * 10 + 1, "{written}");
    }

    #[test]
    fn text_is_cleaned_and_cut() {
        assert_eq!(clean(b"  a\tb\x1b[0m\r", false), "a b\u{fffd}[0m");
        assert_eq!(clean(b"bad \xff\xfe utf8", false), "bad \u{fffd}\u{fffd} utf8");
        assert_eq!(clean(b"long", true), "long…");
        assert_eq!(clean(b" \r", true), "");
        let mut c = Collapser::new(Instant::now());
        assert!(c.line(Instant::now(), "p", b"   ", false).is_empty());
    }

    #[test]
    fn the_switch_and_the_label() {
        assert!(piped(None));
        assert!(piped(Some("".as_ref())));
        assert!(piped(Some("log".as_ref())));
        assert!(!piped(Some("inherit".as_ref())));
        assert!(!piped(Some(" Inherit ".as_ref())));
        assert_eq!(label_of(Path::new("/p/llplugin/libmedia_plugin_cef.so")), "media_plugin_cef");
        assert_eq!(label_of(Path::new("/p/trust-host/aurora-trust-host")), "aurora-trust-host");
        assert_eq!(label_of(Path::new("/p/lib")), "lib");
    }

    #[test]
    fn the_reports_find_the_plugins_lines_by_their_target() {
        let log = "[2026-11-26T19:51:00.123Z INFO  aurora_media::plugin] media_plugin_cef: [1:2:3/4.5:ERROR:a.cc:1] GPU: x\n\
                   [2026-11-26T19:51:00.124Z WARN  aurora_viewer::media] keep ERROR me\n\
                   [2026-11-26T19:51:00.125Z INFO  aurora_media::plugin_output] the viewer's own line\n\
                   [2026-11-26T19:51:00.126Z INFO  aurora_media::plugin] media_plugin_cef: ] GPU: y\n\
                   not a record\n";
        assert!(is_plugin_line(log.lines().next().unwrap()));
        assert_eq!(
            without_plugin_lines(log),
            "[2026-11-26T19:51:00.124Z WARN  aurora_viewer::media] keep ERROR me\n\
             [2026-11-26T19:51:00.125Z INFO  aurora_media::plugin_output] the viewer's own line\n\
             not a record\n"
        );
    }

    /// The log as the viewer's own logger writes it: env_logger built as in
    /// aurora-viewer logging.rs `init` (millisecond timestamps, a pipe
    /// target), here not installed, its records written to a buffer.
    fn logged(style: env_logger::WriteStyle, records: &[(log::Level, &str, &str)]) -> String {
        use log::Log;
        #[derive(Clone)]
        struct Sink(Arc<Mutex<Vec<u8>>>);
        impl std::io::Write for Sink {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let sink = Sink(Arc::new(Mutex::new(Vec::new())));
        let logger = env_logger::Builder::new()
            .filter_level(log::LevelFilter::Trace)
            .format_timestamp_millis()
            .write_style(style)
            .target(env_logger::Target::Pipe(Box::new(sink.clone())))
            .build();
        for (level, target, text) in records {
            logger.log(
                &log::Record::builder()
                    .level(*level)
                    .target(target)
                    .args(format_args!("{text}"))
                    .build(),
            );
        }
        String::from_utf8(sink.0.lock().unwrap().clone()).unwrap()
    }

    #[test]
    fn the_reports_find_the_plugins_lines_in_the_viewers_own_format() {
        let records = [
            (log::Level::Info, TARGET, "media_plugin_cef: [1:2:3/4.5:ERROR:a.cc:1] GPU: x"),
            (log::Level::Warn, "aurora_viewer::media", "keep me"),
            (log::Level::Error, "aurora_media::plugin_output", "keep me too"),
            (log::Level::Info, TARGET, "media_plugin_cef: ] GPU: y"),
        ];
        for style in [env_logger::WriteStyle::Never, env_logger::WriteStyle::Always] {
            let log = logged(style, &records);
            assert_eq!(log.lines().count(), 4, "{log:?}");
            let kept = without_plugin_lines(&log);
            assert_eq!(kept.lines().count(), 2, "{style:?}: {kept:?}");
            assert!(kept.contains("keep me") && kept.contains("keep me too"));
            assert!(!kept.contains("GPU"), "{kept:?}");
        }
        // the plain format is the one the reports' other readers expect
        let line = logged(env_logger::WriteStyle::Never, &records[..1]);
        assert!(
            line.starts_with('[') && line.contains("Z INFO  aurora_media::plugin] media_plugin_cef: "),
            "{line:?}"
        );
    }

    /// The lines a test thread got from `connect`.
    #[cfg(unix)]
    fn collect() -> (Arc<Mutex<Vec<String>>>, impl FnMut(String) + Send + 'static) {
        let got = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&got);
        (got, move |line| sink.lock().unwrap().push(line))
    }

    #[cfg(unix)]
    fn shell(script: &str) -> Command {
        let program = Path::new("/bin/sh");
        let mut cmd = Command::new(program);
        cmd.arg("-c").arg(script);
        cmd
    }

    #[cfg(unix)]
    #[test]
    fn both_streams_are_drained_to_the_end() {
        let (got, sink) = collect();
        let table = Arc::new(Mutex::new(Collapser::new(Instant::now())));
        let mut cmd = shell("echo out; echo err >&2; printf 'no newline'");
        let thread = connect(&mut cmd, "sh".into(), table, sink).expect("pipe");
        let mut child = cmd.spawn().expect("sh");
        // the parent's write ends go with the command: the end of the pipe
        // comes when the child is done
        drop(cmd);
        child.wait().expect("wait");
        thread.join().expect("thread");
        let got = got.lock().unwrap();
        assert_eq!(*got, ["sh: out", "sh: err", "sh: no newline"]);
    }

    #[cfg(unix)]
    #[test]
    fn a_full_pipe_does_not_hold_the_plugin_and_long_lines_are_cut() {
        let (got, sink) = collect();
        let table = Arc::new(Mutex::new(Collapser::new(Instant::now())));
        // far more than a pipe holds (64 KiB), and one line of 100 000 bytes
        let mut cmd = shell(
            "i=0; while [ $i -lt 3000 ]; do echo \"the same line of a long flood\"; i=$((i+1)); done; head -c 100000 /dev/zero | tr '\\0' x; echo; echo end",
        );
        let thread = connect(&mut cmd, "sh".into(), table, sink).expect("pipe");
        let mut child = cmd.spawn().expect("sh");
        drop(cmd);
        let started = Instant::now();
        let status = child.wait().expect("wait");
        assert!(status.success());
        assert!(started.elapsed() < Duration::from_secs(30));
        thread.join().expect("thread");
        let got = got.lock().unwrap();
        let long = got.iter().find(|l| l.starts_with("sh: xxx")).expect("the long line");
        assert_eq!(long.len(), "sh: ".len() + MAX_LINE + '…'.len_utf8());
        assert!(got.len() <= BURST as usize + 2, "{}", got.len());
    }

    #[cfg(unix)]
    #[test]
    fn a_descendant_that_outlives_the_plugin_is_still_read() {
        let (got, sink) = collect();
        let table = Arc::new(Mutex::new(Collapser::new(Instant::now())));
        // the shell ends at once; its background child keeps the pipe open
        let mut cmd = shell("(sleep 1; echo late; echo late; echo late) & echo early");
        let thread = connect(&mut cmd, "sh".into(), table, sink).expect("pipe");
        let mut child = cmd.spawn().expect("sh");
        drop(cmd);
        child.wait().expect("wait");
        assert!(!thread.is_finished(), "the pipe is still open");
        // the thread ends by itself at the end of the pipe
        let until = Instant::now() + Duration::from_secs(20);
        while !thread.is_finished() && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(thread.is_finished());
        let got = got.lock().unwrap();
        assert_eq!(*got, ["sh: early", "sh: late", "sh: repeated 2 times in the last 1 s: late"]);
    }
}
