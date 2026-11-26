// aurora-local-patch: aurora-shader-host-config
//! The text formats of the aurora-shader folder: a strict subset of TOML,
//! read by hand like `looks/cube.rs` reads its tables (the `toml` crate is
//! not in Cargo.lock, and `build.sh` builds `--locked`).
//!
//! One grammar for `config.toml` and for the `//!` header of an effect:
//! `# comments`, `[table]` (`[a-z0-9_-]+`, no dotted names, no array
//! tables), `key = value` (bare keys `[A-Za-z0-9_-]+`) where a value is
//! `true`/`false`, a number, a `"string"` (escapes `\"` and `\\` only) or a
//! one-line array of numbers or of strings. Everything else is an error
//! naming its line. The subset is valid TOML: Python's `tomllib` reads the
//! same files (the aurora-shader repository's tests, a future GUI), so it
//! refuses what TOML refuses too (duplicate keys or tables, `05`, `.5`, `1.`).

use std::collections::HashMap;
use std::fmt;

/// A value of the subset. Numbers are kept as f64 (TOML's integers and
/// floats both end up in f32/i32/u32 shader members).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Value {
    Bool(bool),
    Num(f64),
    Str(String),
    /// Numbers or strings, never mixed (an empty array is either).
    Array(Vec<Value>),
}

impl Value {
    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Value::Bool(_) => "a boolean",
            Value::Num(_) => "a number",
            Value::Str(_) => "a string",
            Value::Array(_) => "an array",
        }
    }

    pub(crate) fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// The strings of an array of strings (an empty array included).
    pub(crate) fn as_strs(&self) -> Option<Vec<&str>> {
        match self {
            Value::Array(items) => items.iter().map(Value::as_str).collect(),
            _ => None,
        }
    }

    /// The numbers a shader member of `n` components takes from this value:
    /// a number fills a scalar, an array of exactly `n` numbers a vector,
    /// a boolean is 1 or 0. `None` for anything else.
    pub(crate) fn numbers(&self, n: usize) -> Option<Vec<f64>> {
        match self {
            Value::Num(v) if n == 1 => Some(vec![*v]),
            Value::Bool(b) if n == 1 => Some(vec![if *b { 1.0 } else { 0.0 }]),
            Value::Array(items) if items.len() == n => items
                .iter()
                .map(|v| match v {
                    Value::Num(x) => Some(*x),
                    Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
                    _ => None,
                })
                .collect(),
            _ => None,
        }
    }
}

/// `key = value` on line `line` (1-based, of the text given to [`parse`]).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Entry {
    pub(crate) key: String,
    pub(crate) value: Value,
    pub(crate) line: usize,
}

/// A table; the root (keys before the first `[table]`) is named "".
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Table {
    pub(crate) name: String,
    pub(crate) line: usize,
    pub(crate) entries: Vec<Entry>,
}

impl Table {
    pub(crate) fn get(&self, key: &str) -> Option<&Value> {
        self.entries.iter().find(|e| e.key == key).map(|e| &e.value)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Doc {
    /// The root first, then the tables in file order.
    pub(crate) tables: Vec<Table>,
}

impl Doc {
    pub(crate) fn table(&self, name: &str) -> Option<&Table> {
        self.tables.iter().find(|t| t.name == name)
    }

    pub(crate) fn root(&self) -> Option<&Table> {
        self.table("")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ParseError {
    pub(crate) line: usize,
    pub(crate) message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

fn is_key_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

/// A table name: an effect's file stem, so lower case only.
pub(crate) fn is_table_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
}

/// Parses `text`; `first_line` is the number of its first line in the file
/// it comes from (the header of an effect starts where the file does).
pub(crate) fn parse(text: &str, first_line: usize) -> Result<Doc, ParseError> {
    parse_lines(text.split('\n').enumerate().map(|(i, l)| (first_line + i, l)))
}

/// Parses lines that already carry their file line numbers (an effect's
/// `//!` lines, taken out of the WGSL around them).
pub(crate) fn parse_lines<'a>(lines: impl Iterator<Item = (usize, &'a str)>) -> Result<Doc, ParseError> {
    let mut doc = Doc {
        tables: vec![Table::default()],
    };
    let mut seen_tables: HashMap<String, usize> = HashMap::new();
    for (line, raw) in lines {
        let err = |message: String| ParseError { line, message };
        let text = raw.strip_suffix('\r').unwrap_or(raw);
        if text.chars().any(|c| (c.is_control() && c != '\t') || c == '\u{feff}') {
            return Err(err("control character or byte order mark".into()));
        }
        let mut cur = Cursor { s: text, pos: 0 };
        cur.skip_ws();
        if cur.at_end_or_comment() {
            continue;
        }
        if cur.eat('[') {
            if cur.eat('[') {
                return Err(err("array tables ([[...]]) are not supported".into()));
            }
            cur.skip_ws();
            let name = cur.take_while(is_key_char);
            cur.skip_ws();
            if !cur.eat(']') {
                return Err(err("expected `[name]` (dotted or quoted table names are not supported)".into()));
            }
            if !is_table_name(name) {
                return Err(err(format!("table name {name:?}: only a-z, 0-9, '_' and '-'")));
            }
            cur.skip_ws();
            if !cur.at_end_or_comment() {
                return Err(err("unexpected text after the table header".into()));
            }
            if let Some(first) = seen_tables.insert(name.to_owned(), line) {
                return Err(err(format!("table [{name}] already defined on line {first}")));
            }
            doc.tables.push(Table {
                name: name.to_owned(),
                line,
                entries: Vec::new(),
            });
            continue;
        }
        let key = cur.take_while(is_key_char);
        if key.is_empty() {
            return Err(err("expected `key = value`, `[table]` or a comment".into()));
        }
        cur.skip_ws();
        if !cur.eat('=') {
            return Err(err(format!("expected '=' after {key:?} (dotted or quoted keys are not supported)")));
        }
        cur.skip_ws();
        let value = cur.value().map_err(err)?;
        cur.skip_ws();
        if !cur.at_end_or_comment() {
            return Err(err("unexpected text after the value".into()));
        }
        // the root table always exists, so there is a last one
        let Some(table) = doc.tables.last_mut() else {
            return Err(err("internal: no table".into()));
        };
        if let Some(first) = table.entries.iter().find(|e| e.key == key) {
            return Err(err(format!("key {key:?} already set on line {}", first.line)));
        }
        table.entries.push(Entry {
            key: key.to_owned(),
            value,
            line,
        });
    }
    Ok(doc)
}

struct Cursor<'a> {
    s: &'a str,
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn rest(&self) -> &'a str {
        &self.s[self.pos..]
    }

    fn peek(&self) -> Option<char> {
        self.rest().chars().next()
    }

    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.pos += c.len_utf8();
            true
        } else {
            false
        }
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(' ' | '\t')) {
            self.pos += 1;
        }
    }

    fn at_end_or_comment(&self) -> bool {
        matches!(self.peek(), None | Some('#'))
    }

    fn take_while(&mut self, f: impl Fn(char) -> bool) -> &'a str {
        let start = self.pos;
        while let Some(c) = self.peek().filter(|&c| f(c)) {
            self.pos += c.len_utf8();
        }
        &self.s[start..self.pos]
    }

    fn value(&mut self) -> Result<Value, String> {
        match self.peek() {
            Some('"') => self.string().map(Value::Str),
            Some('[') => self.array(),
            Some(_) => {
                let word = self.take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.' | '_'));
                match word {
                    "true" => Ok(Value::Bool(true)),
                    "false" => Ok(Value::Bool(false)),
                    "" => Err("expected a value: true, false, a number, a \"string\" or an [array]".into()),
                    _ => number(word).map(Value::Num),
                }
            }
            None => Err("missing value".into()),
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.eat('"');
        let mut out = String::new();
        loop {
            let Some(c) = self.peek() else {
                return Err("unterminated string (strings are one line)".into());
            };
            self.pos += c.len_utf8();
            match c {
                '"' => return Ok(out),
                '\\' => match self.peek() {
                    Some(e @ ('"' | '\\')) => {
                        self.pos += 1;
                        out.push(e);
                    }
                    _ => return Err("only \\\" and \\\\ escapes are supported in strings".into()),
                },
                _ => out.push(c),
            }
        }
    }

    fn array(&mut self) -> Result<Value, String> {
        self.eat('[');
        let mut items = Vec::new();
        loop {
            self.skip_ws();
            if self.eat(']') {
                break;
            }
            if self.peek().is_none() {
                return Err("unterminated array (arrays are one line)".into());
            }
            let item = match self.peek() {
                Some('"') => Value::Str(self.string()?),
                Some('[') => return Err("nested arrays are not supported".into()),
                _ => match self.value()? {
                    v @ Value::Num(_) => v,
                    _ => return Err("arrays hold numbers or strings only".into()),
                },
            };
            if items
                .first()
                .is_some_and(|first: &Value| std::mem::discriminant(first) != std::mem::discriminant(&item))
            {
                return Err("an array mixes numbers and strings".into());
            }
            items.push(item);
            self.skip_ws();
            if self.eat(',') {
                continue;
            }
            self.skip_ws();
            if self.eat(']') {
                break;
            }
            if self.peek().is_none() {
                return Err("unterminated array (arrays are one line)".into());
            }
            return Err("expected ',' or ']' in the array".into());
        }
        Ok(Value::Array(items))
    }
}

/// A TOML decimal number without underscores, infinities or NaN:
/// `[+-]? (0 | [1-9][0-9]*) (. [0-9]+)? ([eE] [+-]? [0-9]+)?`.
fn number(word: &str) -> Result<f64, String> {
    let bad = || format!("{word:?} is not a value: expected true, false, a number (like 1, -0.5, 2e3) or a \"string\"");
    let b = word.as_bytes();
    let mut i = usize::from(matches!(b.first(), Some(b'+' | b'-')));
    let digits = |i: &mut usize| {
        let start = *i;
        while b.get(*i).is_some_and(u8::is_ascii_digit) {
            *i += 1;
        }
        *i - start
    };
    let int_start = i;
    let n = digits(&mut i);
    if n == 0 || (n > 1 && b[int_start] == b'0') {
        return Err(bad());
    }
    if b.get(i) == Some(&b'.') {
        i += 1;
        if digits(&mut i) == 0 {
            return Err(bad());
        }
    }
    if matches!(b.get(i), Some(b'e' | b'E')) {
        i += 1;
        if matches!(b.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        if digits(&mut i) == 0 {
            return Err(bad());
        }
    }
    if i != b.len() {
        return Err(bad());
    }
    match word.parse::<f64>() {
        Ok(v) if v.is_finite() => Ok(v),
        _ => Err(bad()),
    }
}

/// The `//!` header of an effect: the lines starting with `//!` before its
/// first line of code (blank lines and `//` comments may sit between them),
/// each without `//!` and one following space, with its file line number.
pub(crate) fn header_lines(source: &str) -> Vec<(usize, &str)> {
    let mut out = Vec::new();
    for (i, raw) in source.split('\n').enumerate() {
        let line = raw.strip_suffix('\r').unwrap_or(raw).trim_start();
        if let Some(rest) = line.strip_prefix("//!") {
            out.push((i + 1, rest.strip_prefix(' ').unwrap_or(rest)));
        } else if !(line.is_empty() || line.starts_with("//")) {
            break;
        }
    }
    out
}

// ------------------------------------------------------- editing in place

/// One change to `config.toml`: `[table] key = value`, `value` already a
/// token of the subset (see [`number_token`] and its neighbours).
#[derive(Clone, Debug, PartialEq)]
pub struct Edit {
    pub table: String,
    pub key: String,
    pub value: String,
}

/// A number as `config.toml` writes it: whole for an integer member, else
/// with a decimal point (`1.0`) and at most 6 decimals, never `-0.0`.
pub fn number_token(v: f64, integer: bool) -> String {
    let v = if v.is_finite() { v } else { 0.0 };
    if integer {
        let n = v.round();
        return if n == 0.0 { "0".to_owned() } else { format!("{n:.0}") };
    }
    let r = (v * 1e6).round() / 1e6;
    let r = if r == 0.0 { 0.0 } else { r };
    // Display of f64 never uses an exponent: always in the subset's grammar
    let s = format!("{r}");
    if s.contains('.') { s } else { format!("{s}.0") }
}

/// `[a, b, c]` of numbers.
pub fn numbers_token(values: &[f64], integer: bool) -> String {
    let items: Vec<String> = values.iter().map(|v| number_token(*v, integer)).collect();
    format!("[{}]", items.join(", "))
}

pub fn bool_token(b: bool) -> String {
    if b { "true" } else { "false" }.to_owned()
}

/// `["a", "b"]` of names (quotes and backslashes escaped).
pub fn names_token(names: &[String]) -> String {
    let items: Vec<String> = names
        .iter()
        .map(|n| format!("\"{}\"", n.replace('\\', "\\\\").replace('"', "\\\"")))
        .collect();
    format!("[{}]", items.join(", "))
}

/// Where the parts of a `key = value` line are (bytes of the line, `\r`
/// excluded): the key, the value, and the comment after it if any.
struct LineEntry<'a> {
    key: &'a str,
    value: std::ops::Range<usize>,
    comment: Option<usize>,
}

/// What a line of the subset is, for the editor (the file is known to
/// parse: anything odd is simply not an entry).
enum LineKind<'a> {
    Table(&'a str),
    Entry(LineEntry<'a>),
    Other,
}

fn line_kind(line: &str) -> LineKind<'_> {
    let mut cur = Cursor { s: line, pos: 0 };
    cur.skip_ws();
    if cur.eat('[') {
        cur.skip_ws();
        let name = cur.take_while(is_key_char);
        return LineKind::Table(name);
    }
    let key = cur.take_while(is_key_char);
    if key.is_empty() {
        return LineKind::Other;
    }
    cur.skip_ws();
    if !cur.eat('=') {
        return LineKind::Other;
    }
    cur.skip_ws();
    let start = cur.pos;
    if cur.value().is_err() {
        return LineKind::Other;
    }
    let end = cur.pos;
    cur.skip_ws();
    let comment = (cur.peek() == Some('#')).then_some(cur.pos);
    LineKind::Entry(LineEntry {
        key,
        value: start..end,
        comment,
    })
}

/// Applies `edits` to the text of `config.toml` in place: only the value
/// of an existing key changes (its comment, kept in its column when there
/// is room, the order and the blank lines stay); a missing key is added
/// after the last key of its table, a missing table at the end of the
/// file. CRLF files stay CRLF. A text the parser refuses is not touched:
/// the error says why, and nothing may be written over it.
pub fn edit(text: &str, edits: &[Edit]) -> Result<String, String> {
    parse(text, 1).map_err(|e| format!("config.toml line {}: {}", e.line, e.message))?;
    let crlf = text.contains("\r\n");
    let cr = if crlf { "\r" } else { "" };
    let mut lines: Vec<String> = text.split('\n').map(str::to_owned).collect();
    for e in edits {
        if !is_table_name(&e.table) || e.key.is_empty() || !e.key.chars().all(is_key_char) {
            return Err(format!("[{}] {}: not a table or key name of config.toml", e.table, e.key));
        }
        let mut table = String::new();
        let mut table_found = false;
        let mut last_in_table: Option<usize> = None;
        let mut done = false;
        for (i, line) in lines.iter_mut().enumerate() {
            let body = line.strip_suffix('\r').unwrap_or(line).to_owned();
            match line_kind(&body) {
                LineKind::Table(name) => {
                    table = name.to_owned();
                    if table == e.table {
                        table_found = true;
                        last_in_table = Some(i);
                    }
                }
                LineKind::Entry(entry) if table == e.table => {
                    if entry.key == e.key {
                        *line = format!("{}{cr}", replace_value(&body, &entry, &e.value));
                        done = true;
                        break;
                    }
                    last_in_table = Some(i);
                }
                _ => {}
            }
        }
        if done {
            continue;
        }
        let new_line = format!("{} = {}{cr}", e.key, e.value);
        if let (true, Some(last)) = (table_found, last_in_table) {
            // after the table's last key, or its header
            lines.insert(last + 1, new_line);
        } else {
            // a new table at the end: one blank line before it, and the file
            // keeps ending with a newline
            if lines.last().is_some_and(|l| l.is_empty()) {
                lines.pop();
            }
            if lines.iter().any(|l| !l.trim().is_empty()) {
                if lines.last().is_some_and(|l| !l.trim().is_empty()) {
                    lines.push(cr.to_owned());
                }
            } else {
                lines.clear();
            }
            lines.push(format!("[{}]{cr}", e.table));
            lines.push(new_line);
            lines.push(String::new());
        }
    }
    let out = lines.join("\n");
    // the result must read back as asked, or nothing is written
    let doc = parse(&out, 1).map_err(|e| format!("internal: the edited config.toml would not parse (line {}: {})", e.line, e.message))?;
    for (i, e) in edits.iter().enumerate() {
        if edits[i + 1..].iter().any(|later| later.table == e.table && later.key == e.key) {
            continue;
        }
        let want = parse(&format!("k = {}", e.value), 1)
            .ok()
            .and_then(|d| d.root().and_then(|r| r.get("k")).cloned())
            .ok_or_else(|| format!("[{}] {}: {:?} is not a value of config.toml", e.table, e.key, e.value))?;
        if doc.table(&e.table).and_then(|t| t.get(&e.key)) != Some(&want) {
            return Err(format!("internal: [{}] {} did not read back", e.table, e.key));
        }
    }
    Ok(out)
}

/// `line` with its value replaced; a comment after it keeps its column
/// while there is room (one space at least).
fn replace_value(line: &str, entry: &LineEntry<'_>, value: &str) -> String {
    let mut out = String::with_capacity(line.len() + value.len());
    out.push_str(&line[..entry.value.start]);
    out.push_str(value);
    match entry.comment {
        Some(at) => {
            let column = at - entry.value.start;
            let pad = column.saturating_sub(value.len()).max(1);
            out.extend(std::iter::repeat_n(' ', pad));
            out.push_str(&line[at..]);
        }
        None => out.push_str(&line[entry.value.end..]),
    }
    out
}
