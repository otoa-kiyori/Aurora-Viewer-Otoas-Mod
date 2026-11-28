// aurora-local-patch: constellation-builtin-module
//! Where the constellation data comes from: the data built into the viewer
//! (`assets/constellations/`, compiled in by build.rs, build_constellations.rs)
//! and the user's own folder on top of it (`constellations/` in the
//! configuration folder, or `AURORA_CONSTELLATION_DIR` for a launch).
//!
//! The data is cut into units, each taken whole from one layer:
//!
//! ```text
//! sets/<id>/       an art set              its folder in the user's data replaces the built-in one
//! cultures/<id>/   a culture               likewise
//! lines.json, names.json, drafts.json      the western culture without a folder: one unit,
//!                                          the user's as soon as it has one of the three
//! planets.json     the planets' names      the user's when it has the file
//! credits.json     the root's credits      never replaced: both are read
//! ```
//!
//! A unit of the user's replaces the built-in one whole, never figure by
//! figure: a set is drawn in one style (a figure the user's set lacks is
//! not filled in from the built-in set). A user unit counts once its folder
//! holds a file the scan reads (`.json`, `.webp`); an empty folder is not
//! one. A user `cultures/western/` is a culture folder like any other: the
//! scan then reads it instead of the root pair, whichever layer the pair
//! comes from. `AURORA_CONSTELLATION_BUILTIN=0` turns the built-in layer
//! off for a launch (the user's folder alone, as before the data shipped).
//!
//! The built-in bytes are static: nothing is read to show them, and the
//! JSON is parsed on the scan's thread like the user's (no work at start).
//! They go through the same checks and limits as the user's files.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

mod generated {
    include!(concat!(env!("OUT_DIR"), "/constellations_builtin.rs"));
}

/// Files of the data: (path inside the data folder, bytes), sorted by path.
pub type Table = &'static [(&'static str, &'static [u8])];

/// The data built into this executable (empty when it was built without).
pub fn table() -> Table {
    generated::FILES
}

/// `AURORA_CONSTELLATION_BUILTIN=0` (or `off`, `false`): the built-in
/// layer off for this launch.
pub fn builtin_on() -> bool {
    !matches!(
        std::env::var("AURORA_CONSTELLATION_BUILTIN").ok().as_deref().map(str::trim),
        Some("0" | "off" | "false")
    )
}

/// Where one file's bytes are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Bytes {
    /// In the user's folder, read when asked.
    File(PathBuf),
    /// Built into the viewer.
    Builtin(&'static [u8]),
}

impl Bytes {
    /// Its bytes, at most `max` (more is an error, as for a file: the
    /// built-in data passes the same limits).
    pub fn read(&self, max: u64) -> Result<Vec<u8>, String> {
        match self {
            Bytes::File(path) => super::catalogue::read_capped(path, max),
            Bytes::Builtin(b) if b.len() as u64 > max => Err(format!("larger than {} KiB", max >> 10)),
            Bytes::Builtin(b) => Ok(b.to_vec()),
        }
    }

    /// Its length (`None`: the file cannot be read).
    pub fn size(&self) -> Option<u64> {
        match self {
            Bytes::File(path) => std::fs::metadata(path).ok().filter(|m| m.is_file()).map(|m| m.len()),
            Bytes::Builtin(b) => Some(b.len() as u64),
        }
    }

    pub fn is_builtin(&self) -> bool {
        matches!(self, Bytes::Builtin(_))
    }

    /// For the log: its file, or that it is built in.
    pub fn describe(&self) -> String {
        match self {
            Bytes::File(path) => path.display().to_string(),
            Bytes::Builtin(_) => "built-in data".to_owned(),
        }
    }
}

/// The unit of the root pair (the western culture without a folder) and of
/// the planets' names.
pub const WESTERN_ROOT: &str = "western";
pub const PLANETS: &str = "planets";
const ROOT_PAIR: [&str; 3] = ["lines.json", "names.json", "drafts.json"];
const ROOT_CREDITS: &str = "credits.json";

/// The unit a data path belongs to; `None` for the root's credits (both
/// layers' are read).
pub fn unit_of(rel: &str) -> Option<String> {
    let mut parts = rel.splitn(3, '/');
    match (parts.next(), parts.next(), parts.next()) {
        (Some(top @ ("sets" | "cultures")), Some(id), Some(_)) => Some(format!("{top}/{id}")),
        (Some(super::planet_names::FILE), None, None) => Some(PLANETS.to_owned()),
        (Some(ROOT_CREDITS), None, None) => None,
        _ => Some(WESTERN_ROOT.to_owned()),
    }
}

/// The data as the scan sees it: every file by its path inside the data
/// folder, each unit from one layer.
#[derive(Clone, Debug, Default)]
pub struct Data {
    /// The user's folder (the window names it; it may not exist).
    pub dir: PathBuf,
    /// The built-in layer is used.
    pub builtin: bool,
    files: BTreeMap<String, Bytes>,
    /// `sets/<id>` and `cultures/<id>` folders, files or not (a user folder
    /// that reads nothing is still said by the scan).
    dirs: BTreeSet<String>,
    /// The root's `credits.json`: the built-in one first, then the user's.
    root_credits: Vec<Bytes>,
    /// The units taken from the user's folder.
    pub user_units: BTreeSet<String>,
}

/// The user folder's files the scan may read: the root's, and those of
/// each `cultures/<id>/` and `sets/<id>/` (`.json` and `.webp`), bounded
/// like the folder's signature (constellations_catalogue.rs `signature`).
fn user_files(dir: &Path) -> (BTreeMap<String, PathBuf>, BTreeSet<String>) {
    use super::catalogue::{MAX_SET_FIGURES, MAX_SETS};
    let wanted = |name: &str| name.ends_with(".json") || name.ends_with(".webp");
    let list = |d: &Path| -> Vec<(String, std::fs::Metadata)> {
        let mut out: Vec<(String, std::fs::Metadata)> = std::fs::read_dir(d)
            .map(|rd| {
                rd.filter_map(Result::ok)
                    .filter_map(|e| Some((e.file_name().into_string().ok()?, std::fs::metadata(e.path()).ok()?)))
                    .collect()
            })
            .unwrap_or_default();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    };
    let mut files = BTreeMap::new();
    let mut dirs = BTreeSet::new();
    for (name, m) in list(dir) {
        if m.is_file() && name.ends_with(".json") {
            files.insert(name.clone(), dir.join(&name));
        }
    }
    for top in ["cultures", "sets"] {
        for (sub, m) in list(&dir.join(top)).into_iter().take(MAX_SETS * 4) {
            if !m.is_dir() {
                continue;
            }
            dirs.insert(format!("{top}/{sub}"));
            let folder = dir.join(top).join(&sub);
            for (file, fm) in list(&folder).into_iter().take(MAX_SET_FIGURES * 4) {
                if fm.is_file() && wanted(&file) {
                    files.insert(format!("{top}/{sub}/{file}"), folder.join(&file));
                }
            }
        }
    }
    (files, dirs)
}

impl Data {
    /// The user's folder alone (the built-in layer off).
    pub fn folder(dir: &Path) -> Data {
        Data::layered(dir, None)
    }

    /// `table` (the built-in layer, `None`: off) under the user's folder
    /// `dir`: each unit the user's folder has replaces the table's whole.
    pub fn layered(dir: &Path, table: Option<Table>) -> Data {
        let (user, user_dirs) = user_files(dir);
        let present = |unit: &str, rel: &str| unit != WESTERN_ROOT || ROOT_PAIR.contains(&rel);
        let user_units: BTreeSet<String> = user.keys().filter_map(|rel| unit_of(rel).filter(|u| present(u, rel))).collect();
        let mut data = Data {
            dir: dir.to_path_buf(),
            builtin: table.is_some(),
            user_units,
            ..Data::default()
        };
        for &(rel, bytes) in table.unwrap_or_default() {
            match unit_of(rel) {
                None => data.root_credits.push(Bytes::Builtin(bytes)),
                Some(u) if !data.user_units.contains(&u) => {
                    data.files.insert(rel.to_owned(), Bytes::Builtin(bytes));
                }
                Some(_) => {}
            }
        }
        for (rel, path) in user {
            match unit_of(&rel) {
                None => data.root_credits.push(Bytes::File(path)),
                Some(u) if data.user_units.contains(&u) => {
                    data.files.insert(rel, Bytes::File(path));
                }
                Some(_) => {}
            }
        }
        let from_files: Vec<String> = data
            .files
            .keys()
            .filter_map(|rel| {
                let mut p = rel.splitn(3, '/');
                match (p.next(), p.next(), p.next()) {
                    (Some(top @ ("sets" | "cultures")), Some(id), Some(_)) => Some(format!("{top}/{id}")),
                    _ => None,
                }
            })
            .collect();
        data.dirs.extend(from_files);
        data.dirs.extend(user_dirs);
        data
    }

    /// The file at `rel` (a path inside the data folder), if the data has it.
    pub fn get(&self, rel: &str) -> Option<&Bytes> {
        self.files.get(rel)
    }

    /// Its bytes: `None` when absent, else as `Bytes::read`.
    pub fn read(&self, rel: &str, max: u64) -> Option<Result<Vec<u8>, String>> {
        self.get(rel).map(|b| b.read(max))
    }

    /// The folders under `top` (`sets`, `cultures`): their names, sorted.
    pub fn subdirs(&self, top: &str) -> Vec<String> {
        let prefix = format!("{top}/");
        self.dirs
            .iter()
            .filter_map(|d| d.strip_prefix(&prefix))
            .map(str::to_owned)
            .collect()
    }

    /// The files directly in the folder `dir` (`sets/<id>`): names, sorted.
    pub fn files_in(&self, dir: &str) -> Vec<String> {
        let prefix = format!("{dir}/");
        self.files
            .range(prefix.clone()..)
            .take_while(|(rel, _)| rel.starts_with(&prefix))
            .filter_map(|(rel, _)| rel.strip_prefix(&prefix).filter(|n| !n.contains('/')).map(str::to_owned))
            .collect()
    }

    /// The root's `credits.json` files, built-in first.
    pub fn root_credits(&self) -> &[Bytes] {
        &self.root_credits
    }

    /// Whether the unit (`sets/<id>`, `cultures/<id>`, `western`,
    /// `planets`) comes from the user's folder.
    pub fn is_user_unit(&self, unit: &str) -> bool {
        self.user_units.contains(unit)
    }

    /// Nothing at all in either layer.
    pub fn is_empty(&self) -> bool {
        self.files.is_empty() && self.root_credits.is_empty()
    }

    /// For the log: where a problem's file is (a problem starts with its
    /// path inside the data, `sets/<id>/…: …`), the user's folder or the
    /// built-in data.
    pub fn origin_of(&self, problem: &str) -> String {
        let rel = problem.split(':').next().unwrap_or_default();
        let user = match unit_of(rel) {
            Some(unit) => self.is_user_unit(&unit),
            None => !self.builtin,
        };
        if user || !self.builtin {
            self.dir.display().to_string()
        } else {
            "built-in data".to_owned()
        }
    }

    /// For the log: the layers read, and the units the user's folder gives.
    pub fn describe(&self) -> String {
        match (self.builtin, self.user_units.is_empty()) {
            (false, _) => self.dir.display().to_string(),
            (true, true) => "the built-in data".to_owned(),
            (true, false) => format!(
                "the built-in data and {} ({})",
                self.dir.display(),
                self.user_units.iter().map(String::as_str).collect::<Vec<_>>().join(", ")
            ),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::world::constellations::catalogue::tests::folder;
    use crate::world::constellations::catalogue::{self, Kind, MAX_FIGURE_FILE, MAX_WEBP};

    const LINES: &str = r#"{"Ori":{"name":"Orion","label":[88.0,22.8],"lines":[{"a":[78.6,-8.2],"b":[81.1,-2.4]}]},
        "Cyg":{"name":"Cygnus","label":[310.0,40.0],"lines":[{"a":[305.0,40.0],"b":[310.0,45.0]}]}}"#;
    const SET: &str = r#"{"name":"Built set","order":1}"#;

    /// A figure file with a drawing (`picture` in the scan) of `id`.
    fn figure(id: &str) -> String {
        format!(r#"{{"id":"{id}","kind":"modern","centre":[88.0,10.0],"size":[16,16],"texel":0.01,"origin":[-0.08,-0.08]}}"#)
    }

    fn leak(s: String) -> &'static [u8] {
        Box::leak(s.into_bytes().into_boxed_slice())
    }

    /// A small built-in table: the root pair, a two-figure set, a culture,
    /// the planets and the root's credits.
    fn builtin() -> Table {
        let rows: Vec<(&'static str, &'static [u8])> = vec![
            ("credits.json", br#"{"credits":[{"holder":"Built Library"}]}"#),
            ("cultures/other/culture.json", br#"{"name":"Other"}"#),
            (
                "cultures/other/lines.json",
                br#"{"Aa":{"name":"Aa","label":[10.0,10.0],"lines":[{"a":[10.0,10.0],"b":[11.0,11.0]}]}}"#,
            ),
            ("lines.json", LINES.as_bytes()),
            ("names.json", br#"{"names":{"fr":{"Ori":"Orion built"}}}"#),
            ("planets.json", br#"{"mars":{"name":"Built Mars"}}"#),
            ("sets/built/Cyg.json", leak(figure("Cyg"))),
            ("sets/built/Cyg.webp", b"RIFF-built-cyg"),
            ("sets/built/Ori.json", leak(figure("Ori"))),
            ("sets/built/Ori.webp", b"RIFF-built-ori"),
            ("sets/built/set.json", SET.as_bytes()),
        ];
        Box::leak(rows.into_boxed_slice())
    }

    fn write(root: &Path, rel: &str, bytes: &[u8]) {
        catalogue::tests::write(root, rel, bytes);
    }

    fn scan(data: &Data) -> (catalogue::Catalogue, Vec<String>) {
        let mut problems = Vec::new();
        let c = catalogue::scan_data(data, &mut problems);
        (c, problems)
    }

    #[test]
    fn units_of_paths() {
        assert_eq!(unit_of("sets/a/Ori.json").as_deref(), Some("sets/a"));
        assert_eq!(unit_of("cultures/b/lines.json").as_deref(), Some("cultures/b"));
        assert_eq!(unit_of("lines.json").as_deref(), Some(WESTERN_ROOT));
        assert_eq!(unit_of("drafts.json").as_deref(), Some(WESTERN_ROOT));
        assert_eq!(unit_of("planets.json").as_deref(), Some(PLANETS));
        assert_eq!(unit_of("credits.json"), None);
    }

    #[test]
    fn built_in_alone_and_an_empty_user_folder() {
        let root = folder("layers-builtin");
        // an empty set folder of the user's is not a unit: the built-in set stays
        std::fs::create_dir_all(root.join("sets/built")).expect("folder");
        let data = Data::layered(&root, Some(builtin()));
        assert!(data.user_units.is_empty());
        let (c, problems) = scan(&data);
        assert!(problems.is_empty(), "{problems:?}");
        assert_eq!(c.sets.len(), 1);
        assert_eq!(c.cultures.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(), ["western", "other"]);
        let ori = c.figures.iter().find(|f| f.id == "Ori").expect("Orion");
        assert_eq!(ori.drawings.len(), 1);
        assert_eq!(ori.drawings[0].image, catalogue::Image::Builtin(&b"RIFF-built-ori"[..]));
        assert_eq!(ori.names.get("fr").map(String::as_str), Some("Orion built"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_user_set_replaces_the_built_in_set_whole() {
        let root = folder("layers-set");
        // the user's set of the same id draws Orion only: Cygnus is not
        // filled in from the built-in set
        write(&root, "sets/built/set.json", br#"{"name":"My set","order":1}"#);
        write(&root, "sets/built/Ori.json", figure("Ori").as_bytes());
        write(&root, "sets/built/Ori.webp", b"RIFF-mine");
        let data = Data::layered(&root, Some(builtin()));
        assert_eq!(data.user_units.iter().collect::<Vec<_>>(), ["sets/built"]);
        assert!(data.is_user_unit("sets/built") && !data.is_user_unit(WESTERN_ROOT));
        let (c, problems) = scan(&data);
        assert!(problems.is_empty(), "{problems:?}");
        assert_eq!(c.sets[0].display_name("en"), "My set");
        let drawn = |id: &str| {
            c.figures
                .iter()
                .find(|f| f.id == id)
                .map(|f| f.drawings.clone())
                .unwrap_or_default()
        };
        assert!(matches!(&drawn("Ori")[0].image, catalogue::Image::File(p) if p.ends_with("Ori.webp")));
        assert!(drawn("Cyg").is_empty(), "no figure-by-figure fallback");
        // the lines still come from the built-in root pair
        assert!(c.figures.iter().any(|f| f.id == "Cyg" && !f.geom.lines.is_empty()));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_user_culture_and_root_pair_and_planets() {
        let root = folder("layers-root");
        write(&root, "cultures/other/culture.json", br#"{"name":"Mine"}"#);
        write(
            &root,
            "cultures/other/lines.json",
            br#"{"Bb":{"name":"Bb","label":[20.0,20.0],"lines":[{"a":[20.0,20.0],"b":[21.0,21.0]}]}}"#,
        );
        // names.json alone takes the whole root pair: no built-in lines.json
        write(&root, "names.json", br#"{"names":{"fr":{"Ori":"Orion mine"}}}"#);
        write(&root, "planets.json", br#"{"mars":{"name":"My Mars"}}"#);
        let data = Data::layered(&root, Some(builtin()));
        assert_eq!(
            data.user_units.iter().map(String::as_str).collect::<Vec<_>>(),
            ["cultures/other", PLANETS, WESTERN_ROOT]
        );
        assert!(data.get("lines.json").is_none(), "the root pair is the user's, whole");
        let (c, _) = scan(&data);
        let other: Vec<&str> = c
            .figures
            .iter()
            .filter(|f| c.cultures[f.culture].id == "other")
            .map(|f| f.id.as_str())
            .collect();
        assert_eq!(other, ["Bb"], "the user's culture, not the built-in figures");
        // the western names are the user's file's, and the root lines are
        // gone with the built-in pair: Orion is the built-in set's file only
        let ori = c.figures.iter().find(|f| f.id == "Ori").expect("Orion, from the built-in set");
        assert_eq!(ori.names.get("fr").map(String::as_str), Some("Orion mine"));
        assert!(ori.geom.lines.is_empty());
        let mut problems = Vec::new();
        let planets = super::super::planet_names::read_data(&data, &mut problems);
        let mars = crate::world::planets::ephemeris::Body::ALL
            .iter()
            .position(|b| b.id() == "mars")
            .expect("Mars");
        assert_eq!(planets[mars].as_ref().map(|n| n.name.as_str()), Some("My Mars"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_user_western_folder_wins_over_the_built_in_root_pair() {
        let root = folder("layers-western");
        write(
            &root,
            "cultures/western/lines.json",
            br#"{"Lyr":{"name":"Lyra","label":[280.0,36.0],"lines":[{"a":[279.0,38.0],"b":[281.0,33.0]}]}}"#,
        );
        let (c, _) = scan(&Data::layered(&root, Some(builtin())));
        let western: Vec<&str> = c.figures.iter().filter(|f| f.culture == 0).map(|f| f.id.as_str()).collect();
        // Lyra from the user's folder; Orion and Cygnus only through the
        // built-in set's files (their lines.json is the root pair's, unread)
        assert!(western.contains(&"Lyr"));
        assert!(
            !c.figures
                .iter()
                .any(|f| f.names.get("fr").map(String::as_str) == Some("Orion built"))
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn root_credits_add_up() {
        let root = folder("layers-credits");
        write(&root, "credits.json", br#"{"credits":[{"holder":"My Source"}]}"#);
        let data = Data::layered(&root, Some(builtin()));
        let mut problems = Vec::new();
        let groups = super::super::credits::read_data(&data, &[], &[], &mut problems);
        let holders: Vec<&str> = groups.iter().flat_map(|g| g.credits.iter().map(|c| c.holder.as_str())).collect();
        assert_eq!(holders, ["Built Library", "My Source"]);
        assert!(data.user_units.is_empty(), "the root's credits replace no unit");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn builtin_off_is_the_user_folder_alone() {
        let root = folder("layers-off");
        let (c, _) = scan(&Data::layered(&root, None));
        assert!(c.is_empty(), "no folder, no built-in layer: nothing, as before the data shipped");
        write(&root, "lines.json", LINES.as_bytes());
        let data = Data::layered(&root, None);
        let same = catalogue::scan(&root, &mut Vec::new());
        let (c, _) = scan(&data);
        assert_eq!(c.figures.len(), same.figures.len());
        assert!(c.figures.iter().all(|f| f.drawings.is_empty()));
        assert!(!data.builtin);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The units of the real built-in data, from its files.
    fn real_units() -> BTreeSet<String> {
        table().iter().filter_map(|(rel, _)| unit_of(rel)).collect()
    }

    /// The units `index.json` lists: its cultures' and sets' folders (the
    /// root pair and the planets' names are not folders).
    fn index_units(bytes: &[u8]) -> BTreeSet<String> {
        let v: serde_json::Value = serde_json::from_slice(bytes).expect("index.json reads");
        ["cultures", "sets"]
            .iter()
            .flat_map(|top| {
                v.get(top)
                    .and_then(serde_json::Value::as_array)
                    .unwrap_or_else(|| panic!("index.json lists its {top}"))
                    .iter()
                    .map(move |id| format!("{top}/{}", id.as_str().expect("an id")))
            })
            .collect()
    }

    /// The bytes of a built-in file.
    fn built(rel: &str) -> &'static [u8] {
        table()
            .iter()
            .find(|(r, _)| *r == rel)
            .unwrap_or_else(|| panic!("{rel} is built in"))
            .1
    }

    #[test]
    fn the_built_in_data_reads_whole_and_draws() {
        assert!(table().len() > 100, "the data is built in ({} files)", table().len());
        assert!(table().windows(2).all(|w| w[0].0 < w[1].0), "sorted by path, no duplicate");
        let root = folder("layers-real");
        let data = Data::layered(&root, Some(table()));
        let (c, problems) = scan(&data);
        assert!(problems.is_empty(), "{problems:?}");
        assert!(c.cultures.len() >= 7 && c.sets.len() >= 8, "{}", catalogue::summary(&c));
        assert!(c.figures.iter().filter(|f| f.culture == 0).count() >= 88);
        let drawings: Vec<&catalogue::Drawing> = c.figures.iter().flat_map(|f| &f.drawings).collect();
        assert!(drawings.len() > 100);
        assert!(drawings.iter().all(|d| d.image.is_builtin()));
        // the planets' names and every credits file read too
        let mut problems = Vec::new();
        let planets = super::super::planet_names::read_data(&data, &mut problems);
        assert!(planets.iter().all(Option::is_some), "every planet named");
        let cultures: Vec<(String, String)> = c.cultures.iter().map(|k| (k.id.clone(), k.id.clone())).collect();
        let sets: Vec<(String, String)> = c.sets.iter().map(|s| (s.id.clone(), s.id.clone())).collect();
        let groups = super::super::credits::read_data(&data, &cultures, &sets, &mut problems);
        assert!(problems.is_empty(), "{problems:?}");
        assert!(groups.len() >= 1 + 6 + 8, "{} credit groups", groups.len());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn built_in_pictures_decode_from_the_executable() {
        let (c, _) = scan(&Data::layered(&folder("layers-decode"), Some(table())));
        // the first drawing of each set, decoded as the overlay's worker does
        let mut seen = BTreeSet::new();
        for d in c.figures.iter().flat_map(|f| &f.drawings) {
            if !seen.insert(d.set) {
                continue;
            }
            let Bytes::Builtin(webp) = d.image else {
                panic!("a built-in drawing")
            };
            let mips = super::super::atlas::decode(webp, d.picture.size).expect("decodes");
            assert!(!mips.is_empty());
        }
        assert_eq!(seen.len(), c.sets.len(), "every set has a drawing");
    }

    /// Every text of every built-in `credits.json` reads whole: the window
    /// shows a credit as the loader keeps it, so a text over the loader's
    /// limits (`MAX_TEXT`, `MAX_URL`, `MAX_NOTE`) would be shown cut.
    #[test]
    fn every_built_in_credit_reads_whole() {
        use super::super::credits::{FILE, parse};
        let files: Vec<_> = table().iter().filter(|(rel, _)| rel.rsplit('/').next() == Some(FILE)).collect();
        assert!(files.len() >= 1 + 6 + 8, "{} credits files", files.len());
        let mut cut = Vec::new();
        for (rel, bytes) in files {
            let mut problems = Vec::new();
            let read = parse(bytes, &mut problems).unwrap_or_else(|e| panic!("{rel}: {e}"));
            assert!(problems.is_empty(), "{rel}: {problems:?}");
            let raw: serde_json::Value = serde_json::from_slice(bytes).expect("reads");
            let entries = raw["credits"].as_array().expect("a list");
            assert_eq!(read.len(), entries.len(), "{rel}: every entry");
            for (n, (c, e)) in read.iter().zip(entries).enumerate() {
                let fields: [(&str, &str); 5] = [
                    ("holder", &c.holder),
                    ("work", &c.work),
                    ("licence", &c.licence),
                    ("url", &c.url),
                    ("note", &c.note),
                ];
                for (key, kept) in fields {
                    let given = e[key].as_str().unwrap_or_default().trim();
                    if kept != given {
                        cut.push(format!(
                            "{rel} entry {} {key}: {} of {} characters",
                            n + 1,
                            kept.chars().count(),
                            given.chars().count()
                        ));
                    }
                }
            }
        }
        assert!(cut.is_empty(), "credit texts cut: {cut:#?}");
    }

    /// `index.json` lists exactly the built-in files, every unit has its
    /// credits and origin records, and NOTICE.md has one row per unit.
    #[test]
    fn the_index_the_folders_and_the_notice_agree() {
        let files: BTreeSet<String> = table().iter().map(|(rel, _)| (*rel).to_owned()).collect();
        let units = real_units();
        let folders: BTreeSet<String> = units.iter().filter(|u| u.contains('/')).cloned().collect();
        assert_eq!(
            index_units(built("index.json")),
            folders,
            "index.json and the data's folders differ"
        );
        assert!(units.contains(WESTERN_ROOT) && units.contains(PLANETS));
        for unit in &units {
            let at = |file: &str| match unit.as_str() {
                WESTERN_ROOT | PLANETS => file.to_owned(),
                u => format!("{u}/{file}"),
            };
            if unit != PLANETS {
                assert!(files.contains(&at("credits.json")), "{unit}: no credits.json");
            }
            assert!(files.contains(&at(super::super::credits::PROVENANCE)), "{unit}: no provenance.json");
        }
        let notice = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../NOTICE.md")).expect("NOTICE.md");
        let table_rows: Vec<&str> = notice
            .split("## Bundled assets")
            .nth(1)
            .and_then(|s| s.split("\n## ").next())
            .expect("the Bundled assets table")
            .lines()
            .filter(|l| l.contains("assets/constellations/"))
            .collect();
        let named: BTreeSet<String> = table_rows
            .iter()
            .filter_map(|row| {
                let loc = row.split("`crates/aurora-viewer/assets/constellations/").nth(1)?;
                let loc = loc.split('`').next()?;
                Some(match loc.trim_end_matches('/') {
                    "lines.json" => WESTERN_ROOT.to_owned(),
                    "planets.json" => PLANETS.to_owned(),
                    other => other.to_owned(),
                })
            })
            .collect();
        assert_eq!(table_rows.len(), named.len(), "one row per unit");
        assert_eq!(named, units, "NOTICE.md's rows and the data's units differ");
    }

    /// Every unit's `provenance.json` reads and covers each of its figures;
    /// no figure field is `unknown`. Names, readings and the sets' and
    /// cultures' own names may be (the data records which: counted here,
    /// not forced).
    #[test]
    fn every_built_in_unit_records_where_its_figures_come_from() {
        use super::super::credits::{Origin, PROVENANCE, parse_provenance};
        let mut unknown = BTreeMap::new();
        for unit in real_units().iter().filter(|u| u.as_str() != PLANETS) {
            let dir = if unit == WESTERN_ROOT { String::new() } else { format!("{unit}/") };
            let p = parse_provenance(built(&format!("{dir}{PROVENANCE}"))).unwrap_or_else(|e| panic!("{unit}: {e}"));
            // the unit's figures: its lines' ids, or a set's figure files
            let ids: BTreeSet<String> = if unit.starts_with("sets/") {
                table()
                    .iter()
                    .filter_map(|(rel, bytes)| {
                        let stem = rel.strip_prefix(dir.as_str())?.strip_suffix(".json")?;
                        let v: serde_json::Value = serde_json::from_slice(bytes).ok()?;
                        (v.get("kind").is_some() && v.get("id").and_then(|i| i.as_str()) == Some(stem)).then(|| stem.to_owned())
                    })
                    .collect()
            } else {
                let lines: BTreeMap<String, serde_json::Value> =
                    serde_json::from_slice(built(&format!("{dir}lines.json"))).expect("lines.json reads");
                lines.into_keys().collect()
            };
            assert!(!ids.is_empty(), "{unit}: no figure");
            let missing: Vec<&String> = ids.iter().filter(|id| !p.figures.contains_key(*id)).collect();
            assert!(missing.is_empty(), "{unit}: no record for {missing:?}");
            let bad: Vec<(&String, &String)> = p
                .figures
                .iter()
                .flat_map(|(id, f)| f.iter().filter(|(_, o)| **o == Origin::Unknown).map(move |(k, _)| (id, k)))
                .collect();
            assert!(bad.is_empty(), "{unit}: figure fields of unknown origin {bad:?}");
            let count = p
                .names
                .values()
                .chain(p.readings.values())
                .flat_map(|m| m.values())
                .chain(p.set_name.values())
                .chain(p.culture_name.values());
            let n = count.filter(|o| **o == Origin::Unknown).count();
            if n > 0 {
                unknown.insert(unit.clone(), n);
            }
        }
        eprintln!("names, readings and set or culture names of unknown origin: {unknown:?}");
    }

    #[test]
    fn oversize_built_in_entries_are_refused() {
        let big_json: &'static [u8] = Box::leak(vec![b' '; MAX_FIGURE_FILE as usize + 1].into_boxed_slice());
        let big_webp: &'static [u8] = Box::leak(vec![0u8; MAX_WEBP as usize + 1].into_boxed_slice());
        let rows: Vec<(&'static str, &'static [u8])> = vec![
            ("lines.json", LINES.as_bytes()),
            ("sets/big/Cyg.json", big_json),
            ("sets/big/Ori.json", leak(figure("Ori"))),
            ("sets/big/Ori.webp", big_webp),
            ("sets/big/set.json", SET.as_bytes()),
        ];
        let table: Table = Box::leak(rows.into_boxed_slice());
        let root = folder("layers-big");
        let (c, problems) = scan(&Data::layered(&root, Some(table)));
        assert!(
            problems.iter().any(|p| p.starts_with("sets/big/Cyg.json: larger than")),
            "{problems:?}"
        );
        assert!(
            problems.iter().any(|p| p.starts_with("sets/big/Ori.webp: larger than")),
            "{problems:?}"
        );
        assert!(c.figures.iter().all(|f| f.drawings.is_empty()));
        assert_eq!(c.figures.iter().filter(|f| f.kind == Kind::Modern).count(), 2);
        let _ = std::fs::remove_dir_all(&root);
    }
}
