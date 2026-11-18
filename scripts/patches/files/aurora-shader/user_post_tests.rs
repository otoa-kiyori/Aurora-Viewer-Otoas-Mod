// aurora-local-patch: aurora-shader-host-tests
//! Tests of the aurora-shader host: the TOML subset, the contract checks,
//! the parameters, the folder loader, the AA setting's mapping; and, with a
//! GPU (`-- --ignored`), the chain drawn end to end, its AA finish and the
//! two wgpu behaviours contract v1 relies on.

use crate::pipeline_cache::CachedPipelines;
use std::path::PathBuf;
use std::sync::Arc;

use super::config::{self, Value};
use super::library::{self, Compiled, Effect, Library, Loader, ScalarKind};

// ------------------------------------------------------------- TOML subset

#[test]
fn subset_reads_the_documented_config() {
    let text = "# aurora-shader — config format 1\r\n[host]\nenabled = true\norder = [\"aerial\", \"rim\", \"ink\", \"ca\"]   # run in this order\nphotos = true\nvr = false\n\n[aerial]\nenabled = false\ndensity = 0.5\nwarmth = -1\nshift = [0.08, 0, -2e-1,]\nname = \"a \\\"quoted\\\" \\\\ name\"\nempty = []\n";
    let doc = config::parse(text, 1).expect("valid");
    let host = doc.table("host").expect("host");
    assert_eq!(host.get("enabled"), Some(&Value::Bool(true)));
    assert_eq!(host.get("order").and_then(Value::as_strs), Some(vec!["aerial", "rim", "ink", "ca"]));
    let aerial = doc.table("aerial").expect("aerial");
    assert_eq!(aerial.line, 8);
    assert_eq!(aerial.get("warmth"), Some(&Value::Num(-1.0)));
    assert_eq!(aerial.get("shift").and_then(|v| v.numbers(3)), Some(vec![0.08, 0.0, -0.2]));
    assert_eq!(aerial.get("name").and_then(Value::as_str), Some("a \"quoted\" \\ name"));
    assert_eq!(aerial.get("empty").and_then(Value::as_strs), Some(vec![]));
    assert_eq!(doc.root().map(|r| r.entries.len()), Some(0));
}

#[test]
fn subset_refuses_what_toml_refuses_or_it_does_not_support() {
    let bad = [
        ("a = 1\na = 2", 2, "already set"),
        ("[t]\n[t]", 2, "already defined"),
        ("[[t]]", 1, "array tables"),
        ("a.b = 1", 1, "dotted"),
        ("[a.b]", 1, "dotted"),
        ("\"a\" = 1", 1, "expected `key = value`"),
        ("a = 05", 1, "not a value"),
        ("a = .5", 1, "not a value"),
        ("a = 1.", 1, "not a value"),
        ("a = 1_000", 1, "not a value"),
        ("a = nan", 1, "not a value"),
        ("a = inf", 1, "not a value"),
        ("a = 1e", 1, "not a value"),
        ("a = TRUE", 1, "not a value"),
        ("a = \"x\\n\"", 1, "escapes"),
        ("a = \"x", 1, "unterminated"),
        ("a = [1, \"x\"]", 1, "mixes"),
        ("a = [[1]]", 1, "nested"),
        ("a = [true]", 1, "numbers or strings"),
        ("a = [1, 2", 1, "unterminated array"),
        ("a = 1 2", 1, "after the value"),
        ("a =", 1, "missing value"),
        ("[Upper]", 1, "only a-z"),
        ("[t] x", 1, "after the table header"),
        ("\n\na = 'single'", 3, "expected a value"),
        ("a = \"\u{1}\"", 1, "control character"),
    ];
    for (text, line, needle) in bad {
        let err = config::parse(text, 1).expect_err(text);
        assert_eq!(err.line, line, "{text}: {err}");
        assert!(err.message.contains(needle), "{text}: {err}");
    }
}

#[test]
fn header_lines_are_the_leading_doc_comments() {
    let src = "// a comment\n//! api = 1\n\n//! name = \"Ink\"\r\n//![defaults]\n  //! strength = 1.0\nfn f() {}\n//! not = \"header\"\n";
    let lines = config::header_lines(src);
    assert_eq!(
        lines,
        vec![(2, "api = 1"), (4, "name = \"Ink\""), (5, "[defaults]"), (6, "strength = 1.0")]
    );
    let doc = config::parse_lines(lines.into_iter()).expect("header");
    assert_eq!(doc.table("defaults").and_then(|t| t.get("strength")), Some(&Value::Num(1.0)));
    assert!(doc.root().and_then(|r| r.get("not")).is_none());
}

// ------------------------------------------------------------- the shaders

fn validate(src: &str, caps: naga::valid::Capabilities) {
    let module = match naga::front::wgsl::parse_str(src) {
        Ok(m) => m,
        Err(e) => panic!("{}", e.emit_to_string(src)),
    };
    if let Err(e) = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), caps).validate(&module) {
        panic!("{}", e.emit_to_string(src));
    }
}

#[test]
fn host_shaders_are_valid() {
    validate(include_str!("shaders/user_post_host.wgsl"), naga::valid::Capabilities::empty());
}

/// The prelude is core WGSL: it validates with no capability, alone and
/// with the smallest effect.
#[test]
fn prelude_is_valid_core_wgsl() {
    let src = format!("{}\n{}", library::PRELUDE, PASSTHROUGH);
    validate(&src, naga::valid::Capabilities::empty());
}

const PASSTHROUGH: &str =
    "//! api = 1\n@fragment\nfn fs_main(in: FsIn) -> @location(0) vec4<f32> {\n    return textureLoad(color, pixel_of(in), 0);\n}\n";

/// Touches every helper of the prelude, `state` from both stages, and
/// every kind of Params member.
const FULL: &str = r#"//! api = 1
//! name = "Full"
//! [defaults]
//! gain = 2.0
//! tint = [0.1, 0.2, 0.3]
//! steps = 4
//! mask = true
//! [ranges]
//! gain = [0.0, 4.0]

struct Params {
    gain: f32,
    tint: vec3<f32>,
    steps: i32,
    mask: u32,
    offset: vec2<f32>,
}
@group(1) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn prepare(@builtin(local_invocation_index) i: u32) {
    if (i == 0u) {
        state[0] = vec4<f32>(0.25, 0.0, 0.0, 1.0);
    }
}

@fragment
fn fs_main(in: FsIn) -> @location(0) vec4<f32> {
    let px = pixel_of(in);
    let d = raw_depth(depth_pixel(px));
    var c = textureSampleLevel(color, linear_sampler, in.uv, 0.0);
    c += textureSampleLevel(color, point_sampler, whole_uv(in.uv), 0.0) * 0.0;
    let n = view_normal(px);
    var z = 0.0;
    if (!is_sky(d)) {
        z = view_z(d) + length(view_pos(in.uv, d)) * 0.0;
    }
    let extra = (optical_centre().x + pct_to_px(1.0) + ref_width() + luma(c.rgb) + f32(params.steps) + f32(params.mask)) * 0.0;
    return vec4<f32>(c.rgb * params.gain + state[0].xyz + n.xyz * 0.0 + vec3<f32>(z * 0.0 + extra), 1.0);
}
"#;

fn compiled(src: &str) -> Compiled {
    let mut warnings = Vec::new();
    let c = library::compile("t.wgsl", src, &mut warnings).unwrap_or_else(|e| panic!("{e:?}"));
    assert!(warnings.is_empty(), "{warnings:?}");
    c
}

#[test]
fn compile_reflects_params_and_uses() {
    let c = compiled(FULL);
    assert_eq!(c.title.as_deref(), Some("Full"));
    assert!(c.prepare);
    assert!(c.reads_normals);
    let layout: Vec<(&str, u32, ScalarKind, u32)> = c.params.iter().map(|m| (m.name.as_str(), m.offset, m.kind, m.components)).collect();
    // WGSL's uniform layout: vec3 aligned on 16, the i32 packed after it
    assert_eq!(
        layout,
        vec![
            ("gain", 0, ScalarKind::F32, 1),
            ("tint", 16, ScalarKind::F32, 3),
            ("steps", 28, ScalarKind::I32, 1),
            ("mask", 32, ScalarKind::U32, 1),
            ("offset", 40, ScalarKind::F32, 2),
        ]
    );
    let p = compiled(PASSTHROUGH);
    assert!(!p.prepare && !p.reads_normals && p.params.is_empty());
}

fn errors(src: &str) -> String {
    let mut warnings = Vec::new();
    match library::compile("t.wgsl", src, &mut warnings) {
        Ok(_) => panic!("accepted:\n{src}"),
        Err(e) => e.join("\n"),
    }
}

#[test]
fn compile_refuses_what_contract_v1_does_not_allow() {
    let fs = "@fragment fn fs_main(in: FsIn) -> @location(0) vec4<f32> { return vec4<f32>(1.0); }\n";
    let cases = [
        (fs.to_owned(), "no `//! api = <1 to 2>`"),
        (format!("//! api = 3\n{fs}"), "api = 3, this host runs api = 1 to 2"),
        (format!("//! api = 1.5\n{fs}"), "api = 1 to 2"),
        (format!("//! api = \"1\"\n{fs}"), "api = 1 to 2"),
        ("//! api = 1\n".to_owned(), "no `@fragment fn fs_main"),
        (format!("//! api = 1\n{fs}@compute @workgroup_size(1) fn other() {{}}\n"), "`other`"),
        (
            format!("//! api = 1\n{fs}@compute @workgroup_size(32, 16) fn prepare() {{}}\n"),
            "at most 256",
        ),
        (
            format!("//! api = 1\n{fs}@group(2) @binding(0) var<uniform> x: vec4<f32>;\n"),
            "@group(2)",
        ),
        (
            format!("//! api = 1\n{fs}@group(0) @binding(7) var<uniform> x: vec4<f32>;\n"),
            "@group(0) @binding(7)",
        ),
        // v2's binding is not part of v1, even under its name
        (
            format!("//! api = 1\n{fs}@group(0) @binding(7) var alpha_depth: texture_depth_2d;\n"),
            "@group(0) @binding(7) `alpha_depth` is not part of the contract",
        ),
        (
            format!("//! api = 2\n{fs}@group(0) @binding(8) var<uniform> x: vec4<f32>;\n"),
            "@group(0) @binding(8)",
        ),
        (
            format!("//! api = 1\n{fs}@group(1) @binding(1) var<uniform> x: vec4<f32>;\n"),
            "@group(1) @binding(1)",
        ),
        (
            format!("//! api = 1\n{fs}@group(1) @binding(0) var<uniform> x: vec4<f32>;\n"),
            "must be `var<uniform>",
        ),
        (
            format!("//! api = 1\nstruct P {{ m: mat4x4<f32> }}\n@group(1) @binding(0) var<uniform> params: P;\n{fs}"),
            "only f32, i32, u32",
        ),
        (
            format!("//! api = 1\nstruct P {{ a: array<vec4<f32>, 17> }}\n@group(1) @binding(0) var<uniform> params: P;\n{fs}"),
            "at most 256",
        ),
        (format!("//! api = 1\noverride k: f32 = 1.0;\n{fs}"), "override"),
        (format!("//! api = 1\n//! [defaults\n{fs}"), "t.wgsl:2: header"),
    ];
    for (src, needle) in cases {
        let e = errors(&src);
        assert!(e.contains(needle), "{needle:?} not in {e:?} for\n{src}");
    }
}

/// Errors point at the effect's own lines, not at prelude + effect.
#[test]
fn compile_errors_name_the_effect_line() {
    let src = "//! api = 1\n\n@fragment\nfn fs_main(in: FsIn) -> @location(0) vec4<f32> {\n    return vec4<f32>(nope);\n}\n";
    let e = errors(src);
    assert!(e.starts_with("t.wgsl:5:"), "{e}");
    assert!(e.contains("nope"), "{e}");
    // a validation (not parse) error, here without a span: the file alone
    let src = "//! api = 1\n\n@fragment\nfn fs_main(in: FsIn) -> vec4<f32> {\n    return vec4<f32>(1.0);\n}\n";
    let e = errors(src);
    assert!(e.starts_with("t.wgsl: Entry point fs_main") && !e.contains("prelude"), "{e}");
    // past the longer prelude of v2 too
    let src = "//! api = 2\n\n@fragment\nfn fs_main(in: FsIn) -> @location(0) vec4<f32> {\n    return vec4<f32>(nope);\n}\n";
    let e = errors(src);
    assert!(e.starts_with("t.wgsl:5:"), "{e}");
}

/// Contract v2: v1's prelude, then its additions; core WGSL too.
#[test]
fn prelude_v2_is_valid_core_wgsl() {
    let src = format!("{}\n{}", library::prelude(2), ALPHA_OUT);
    validate(&src, naga::valid::Capabilities::empty());
    assert!(library::prelude(2).starts_with(library::PRELUDE));
    assert_eq!(library::prelude(1), library::PRELUDE);
}

/// A v1 effect is compiled from the very text it was before v2 existed, so
/// it runs as before; a v2 effect gets v2's prelude and bindings.
#[test]
fn contract_v1_effects_are_compiled_as_before() {
    let v1 = compiled(PASSTHROUGH);
    assert_eq!(v1.api, 1);
    assert_eq!(&*v1.source, format!("{}\n{PASSTHROUGH}", library::PRELUDE));
    assert!(!v1.source.contains("alpha_depth"));
    let v2 = compiled(ALPHA_OUT);
    assert_eq!(v2.api, 2);
    assert_eq!(&*v2.source, format!("{}\n{}\n{ALPHA_OUT}", library::PRELUDE, library::PRELUDE_V2));
    // every v1 helper is still there under v2
    let full_v2 = compiled(&FULL.replacen("//! api = 1", "//! api = 2", 1));
    assert!(full_v2.prepare && full_v2.reads_normals && full_v2.params.len() == 5);
}

/// Reads contract v2's blended faces: red their raw depth, green whether one
/// is in front of the scene (3 %), blue `api` / 4, alpha FLAG_ALPHA_DEPTH.
const ALPHA_OUT: &str = "//! api = 2\n@fragment\nfn fs_main(in: FsIn) -> @location(0) vec4<f32> {\n    let px = pixel_of(in);\n    let front = alpha_in_front(px, view_z(raw_depth(px)), 0.03);\n    let flag = select(0.0, 1.0, (frame.flags & FLAG_ALPHA_DEPTH) != 0u);\n    return vec4<f32>(raw_alpha_depth(px), front, f32(frame.api) * 0.25, flag);\n}\n";

/// A v1 effect's view of the same frame: red v2's flag bit, green `api` / 4,
/// blue the scene depth.
const FLAGS_V1: &str = "//! api = 1\n@fragment\nfn fs_main(in: FsIn) -> @location(0) vec4<f32> {\n    let px = pixel_of(in);\n    return vec4<f32>(select(0.0, 1.0, (frame.flags & 32u) != 0u), f32(frame.api) * 0.25, raw_depth(px), 1.0);\n}\n";

#[test]
fn header_warnings_name_unknown_keys() {
    let mut warnings = Vec::new();
    let src = format!("//! api = 1\n//! colour = 1\n//! [defaults]\n//! nope = 1\n//! [extra]\n{PASSTHROUGH}");
    library::compile("t.wgsl", &src, &mut warnings).expect("runs");
    let all = warnings.join("\n");
    assert!(all.contains("t.wgsl:2: header key `colour`"), "{all}");
    assert!(all.contains("t.wgsl:4: [defaults] `nope`"), "{all}");
    assert!(all.contains("t.wgsl:5: header table [extra]"), "{all}");
}

// -------------------------------------------------------------- parameters

fn f32_at(bytes: &[u8], at: usize) -> f32 {
    f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

#[test]
fn params_come_from_config_then_defaults_then_zero() {
    let c = compiled(FULL);
    let doc = config::parse("[full]\nenabled = true\ngain = 3\ntint = [1, 2]\nnope = 1\nmask = false\n", 1).expect("config");
    let mut warnings = Vec::new();
    let bytes = library::resolve_params("full", &c, doc.table("full"), &mut warnings);
    assert_eq!(bytes.len(), library::PARAMS_SIZE);
    assert_eq!(f32_at(&bytes, 0), 3.0, "config wins");
    assert_eq!(
        [f32_at(&bytes, 16), f32_at(&bytes, 20), f32_at(&bytes, 24)],
        [0.1, 0.2, 0.3],
        "wrong length: the default"
    );
    assert_eq!(i32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]), 4, "default");
    assert_eq!(u32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]), 0, "false is 0");
    assert_eq!([f32_at(&bytes, 40), f32_at(&bytes, 44)], [0.0, 0.0], "no value at all");
    let all = warnings.join("\n");
    assert!(all.contains("config.toml:4: [full] `tint` wants an array of 3 numbers"), "{all}");
    assert!(
        all.contains("config.toml:5: [full] `nope` is not a parameter of t.wgsl (gain, tint, steps, mask, offset)"),
        "{all}"
    );
    assert!(all.contains("Params `offset` has no default"), "{all}");
    assert!(!all.contains("enabled"), "{all}");
}

// --------------------------------------------------------------- the folder

struct Folder(PathBuf);

impl Folder {
    fn new(name: &str) -> Folder {
        let dir = std::env::temp_dir().join(format!("aurora-shader-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        Folder(dir)
    }

    fn put(&self, file: &str, text: &str) {
        std::fs::write(self.0.join(file), text).expect("write");
    }
}

impl Drop for Folder {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn names(library: &Library) -> Vec<&str> {
    library.effects.iter().map(|e| e.name.as_str()).collect()
}

fn problems(report: &[(String, Vec<String>)]) -> String {
    report.iter().flat_map(|(_, p)| p.iter().cloned()).collect::<Vec<_>>().join("\n")
}

#[test]
fn loader_runs_the_order_and_reports_problems() {
    let f = Folder::new("order");
    f.put("a.wgsl", PASSTHROUGH);
    f.put("b.wgsl", FULL);
    f.put("off.wgsl", PASSTHROUGH);
    f.put(".hidden.wgsl", "junk");
    f.put(
        "config.toml",
        "[host]\norder = [\"b\", \"missing\", \"a\", \"a\", \"off\", \"Bad\"]\nphotos = false\nwhat = 1\n[off]\nenabled = false\n[b]\ngain = 5\n",
    );
    let mut loader = Loader::new(f.0.clone());
    let (library, report) = loader.load();
    assert_eq!(names(&library), vec!["b", "a"]);
    assert!(!library.photos && !library.vr);
    assert_eq!(f32_at(&library.effects[0].params, 0), 5.0);
    let all = problems(&report);
    assert!(all.contains("\"missing\" but missing.wgsl is missing"), "{all}");
    assert!(all.contains("lists \"a\" twice"), "{all}");
    assert!(all.contains("\"Bad\" is not a file stem"), "{all}");
    assert!(all.contains("[host] `what` is unknown"), "{all}");
    assert!(!all.contains("hidden"), "{all}");
}

#[test]
fn loader_keeps_the_last_good_versions() {
    let f = Folder::new("last-good");
    f.put("a.wgsl", PASSTHROUGH);
    f.put("config.toml", "[host]\norder = [\"a\"]\n");
    let mut loader = Loader::new(f.0.clone());
    let (good, _) = loader.load();
    let good_hash = good.effects[0].compiled.hash;

    // a typo while editing: the effect stays, as it was
    f.put("a.wgsl", "//! api = 1\nfn fs_main( {\n");
    let (library, report) = loader.load();
    assert_eq!(names(&library), vec!["a"]);
    assert_eq!(library.effects[0].compiled.hash, good_hash);
    let all = problems(&report);
    assert!(all.contains("a.wgsl:2:"), "{all}");
    assert!(all.contains("last version that worked keeps running"), "{all}");

    // a broken config.toml keeps the previous one
    f.put("config.toml", "[host\n");
    let (library, report) = loader.load();
    assert_eq!(names(&library), vec!["a"]);
    assert!(problems(&report).contains("config.toml:1:"), "{report:?}");
    assert!(problems(&report).contains("the previous configuration stays"), "{report:?}");

    // a new file that never worked is left out
    f.put("b.wgsl", "//! api = 1\n");
    f.put("config.toml", "[host]\norder = [\"a\", \"b\"]\n");
    let (library, report) = loader.load();
    assert_eq!(names(&library), vec!["a"]);
    assert!(problems(&report).contains("b.wgsl: skipped until it is fixed"), "{report:?}");

    // turned off
    f.put("config.toml", "[host]\nenabled = false\norder = [\"a\"]\n");
    assert!(loader.load().0.effects.is_empty());
}

#[test]
fn loader_without_folder_or_config_runs_nothing_quietly() {
    let f = Folder::new("empty");
    let mut loader = Loader::new(f.0.join("nope"));
    let (library, report) = loader.load();
    assert!(library.effects.is_empty() && report.is_empty());
    let mut loader = Loader::new(f.0.clone());
    let (library, report) = loader.load();
    assert!(library.effects.is_empty());
    assert!(problems(&report).contains("config.toml: missing"));
}

#[test]
fn loader_caps_the_chain() {
    let f = Folder::new("cap");
    let names: Vec<String> = (0..10).map(|i| format!("e{i}")).collect();
    for n in &names {
        f.put(&format!("{n}.wgsl"), PASSTHROUGH);
    }
    let order: Vec<String> = names.iter().map(|n| format!("\"{n}\"")).collect();
    f.put("config.toml", &format!("[host]\norder = [{}]\n", order.join(", ")));
    let (library, report) = Loader::new(f.0.clone()).load();
    assert_eq!(library.effects.len(), library::MAX_EFFECTS);
    assert!(problems(&report).contains("more than 8 effects"));
}

// ------------------------------------------------- the GUI: [ui] hints

const HINTED: &str = r#"//! api = 1
//! name = "Hinted"
//! [defaults]
//! gain = 2.0
//! tint = [0.1, 0.2, 0.3]
//! steps = 4
//! mask = 1
//! offset = [-0.5, 0.25]
//! [ranges]
//! steps = [0, 10]
//! [ui]
//! gain = [0.0, 4.0, 0.05]
//! gain-label = "Gain"
//! tint-kind = "slider"
//! mask-label = "Mask on"
//! offset-label = "Offset"
//! nope = [0.0, 1.0]
//! gain-what = 1
//! mask-kind = "color"
//! tint = [1.0, 0.0]
//! gain-group = "Levels"
//! steps-group = "Levels"
//! gain-ordered = true
//! steps-ordered = true
//! tint-ordered = true
//! offset-group = ""
"#;

fn hinted_body(header: &str) -> String {
    format!(
        "{header}struct Params {{ gain: f32, tint: vec3<f32>, steps: i32, mask: u32, offset: vec2<f32> }}\n@group(1) @binding(0) var<uniform> params: Params;\n@fragment\nfn fs_main(in: FsIn) -> @location(0) vec4<f32> {{\n    return vec4<f32>(params.tint * params.gain + vec3<f32>(f32(params.steps) + f32(params.mask) + params.offset.x), 1.0);\n}}\n"
    )
}

#[test]
fn ui_hints_give_each_param_its_control() {
    use super::Control;
    let mut warnings = Vec::new();
    let c = library::compile("h.wgsl", &hinted_body(HINTED), &mut warnings).expect("runs");
    let p: Vec<_> = c.ui.iter().map(|p| (p.name.as_str(), p)).collect();
    let get = |n: &str| p.iter().find(|(name, _)| *name == n).map(|(_, p)| *p).expect(n);
    let gain = get("gain");
    assert_eq!(
        (gain.label.as_str(), gain.control, gain.min, gain.max, gain.step),
        ("Gain", Control::Slider, 0.0, 4.0, Some(0.05))
    );
    assert_eq!(gain.default, vec![2.0]);
    let tint = get("tint");
    assert_eq!(
        (tint.control, tint.components, tint.label.as_str()),
        (Control::Slider, 3, "tint"),
        "kind = slider wins over the vec3 rule"
    );
    assert_eq!((tint.min, tint.max), (0.0, 1.0), "a bad range falls back to the rule");
    let steps = get("steps");
    assert_eq!(
        (steps.control, steps.integer, steps.min, steps.max, steps.step),
        (Control::Slider, true, 0.0, 10.0, Some(1.0)),
        "[ranges] of v1"
    );
    let mask = get("mask");
    assert_eq!(
        (mask.control, mask.label.as_str()),
        (Control::Flag, "Mask on"),
        "a u32 0/1 is a flag; colour does not fit a scalar"
    );
    let offset = get("offset");
    assert_eq!(
        (offset.control, offset.components, offset.min, offset.max),
        (Control::Slider, 2, -1.0, 1.0)
    );
    let all = warnings.join("\n");
    assert!(all.contains("h.wgsl:17: [ui] `nope`: `nope` is not a member"), "{all}");
    assert!(all.contains("h.wgsl:18: [ui] `gain-what` is unknown"), "{all}");
    assert!(all.contains("h.wgsl:19: [ui] `mask-kind` does not fit"), "{all}");
    assert!(all.contains("h.wgsl:20: [ui] `tint` wants [min, max]"), "{all}");
    assert!(all.contains("h.wgsl:25: [ui] `tint-ordered` is for a scalar member"), "{all}");
    assert!(all.contains("h.wgsl:26: [ui] `offset-group` wants a \"group name\""), "{all}");
    assert_eq!((gain.group.as_deref(), gain.ordered), (Some("Levels"), true));
    assert_eq!((steps.group.as_deref(), steps.ordered), (Some("Levels"), true));
    assert_eq!((tint.group.as_deref(), tint.ordered), (None, false), "ordered is for scalars");
    assert_eq!(offset.group, None);
    assert!(!all.contains("header table"), "[ui] is a known table: {all}");
}

#[test]
fn without_hints_the_rules_pick_the_controls() {
    use super::Control;
    let header =
        "//! api = 1\n//! [defaults]\n//! gain = 3.0\n//! tint = [0.1, 0.2, 0.3]\n//! steps = 4\n//! mask = 0\n//! offset = [-2.0, 0.0]\n";
    let c = compiled(&hinted_body(header));
    let by = |n: &str| c.ui.iter().find(|p| p.name == n).expect(n);
    assert_eq!((by("gain").min, by("gain").max), (0.0, 6.0), "0..2 x default");
    assert_eq!(by("tint").control, Control::Color);
    assert_eq!(
        (by("steps").control, by("steps").max),
        (Control::Slider, 8.0),
        "an integer that is not 0/1"
    );
    assert_eq!(by("mask").control, Control::Flag);
    assert_eq!(
        (by("offset").min, by("offset").max),
        (-4.0, 1.0),
        "negative defaults stay reachable"
    );
    assert_eq!(by("gain").label, "gain");
}

#[test]
fn loader_describes_effects_that_do_not_run() {
    let f = Folder::new("describe");
    f.put("a.wgsl", PASSTHROUGH);
    f.put("off.wgsl", &hinted_body(HINTED));
    f.put("bad.wgsl", "//! api = 1\nfn fs_main( {\n");
    f.put(
        "config.toml",
        "[host]\norder = [\"a\", \"off\", \"gone\", \"bad\"]\n[off]\nenabled = false\n",
    );
    let mut loader = Loader::new(f.0.clone());
    let (library, report) = loader.load();
    assert_eq!(names(&library), vec!["a"]);
    let ui: Vec<_> = library
        .ui
        .iter()
        .map(|e| (e.name.as_str(), e.title.as_str(), e.params.len(), e.problem.is_some()))
        .collect();
    assert_eq!(
        ui,
        vec![
            ("a", "a", 0, false),
            ("off", "Hinted", 5, false),
            ("gone", "gone", 0, true),
            ("bad", "bad", 0, true)
        ]
    );
    // the effect turned off does not report its header's warnings
    assert!(
        !problems(&report).contains("h.wgsl") && !problems(&report).contains("off.wgsl"),
        "{report:?}"
    );
    // the whole chain off: nothing runs, everything is described
    f.put("config.toml", "[host]\nenabled = false\norder = [\"a\", \"off\"]\n");
    let (library, report) = loader.load();
    assert!(library.effects.is_empty());
    assert_eq!(library.ui.len(), 2);
    assert!(problems(&report).is_empty(), "{report:?}");
}

#[test]
fn ordered_chains_push_their_neighbours() {
    let mut v = vec![5.0, 20.0, 80.0, 256.0];
    v[1] = 100.0;
    super::push_ordered(&mut v, 1);
    assert_eq!(v, vec![5.0, 100.0, 100.0, 256.0]);
    v[3] = 1.0;
    super::push_ordered(&mut v, 3);
    assert_eq!(v, vec![1.0, 1.0, 1.0, 1.0]);
    v[0] = 2.0;
    super::push_ordered(&mut v, 0);
    assert_eq!(v, vec![2.0, 2.0, 2.0, 2.0]);
    super::push_ordered(&mut v, 9);
}

#[test]
fn available_lists_the_folder_effects_by_stem() {
    let f = Folder::new("available");
    f.put("b.wgsl", &hinted_body(HINTED));
    f.put("a.wgsl", PASSTHROUGH);
    f.put("Bad.wgsl", PASSTHROUGH);
    f.put("notes.txt", "x");
    f.put("config.toml", "");
    assert_eq!(
        super::available(&f.0),
        vec![("a".to_owned(), "a".to_owned()), ("b".to_owned(), "Hinted".to_owned())]
    );
    assert!(super::available(&f.0.join("nope")).is_empty());
}

// ------------------------------------------- the GUI: editing config.toml

fn edit(table: &str, key: &str, value: &str) -> super::Edit {
    super::Edit {
        table: table.into(),
        key: key.into(),
        value: value.into(),
    }
}

const SHIPPED: &str = "# aurora-shader — config format 1\n#\n# A comment.\n\n[host]\nenabled = true                           # false: nothing runs\norder = [\"aerial\", \"rim\"]   # run in this order\n\n[aerial]\nenabled = true\ndensity = 0.5                            # 0..1, how much haze\nwarmth = 0.0\n\n# the rim\n[rim]\ncontrast = 1.25                          # 0.5..2 on the rim\n";

#[test]
fn edit_replaces_values_and_keeps_comments_and_layout() {
    let out = super::edit_config(
        SHIPPED,
        &[
            edit("aerial", "density", &super::number_token(0.25, false)),
            edit("host", "enabled", &super::bool_token(false)),
            edit("rim", "contrast", "1.5"),
        ],
    )
    .expect("edited");
    let want = SHIPPED
        .replace(
            "density = 0.5                            #",
            "density = 0.25                           #",
        )
        .replace(
            "enabled = true                           # false",
            "enabled = false                          # false",
        )
        .replace(
            "contrast = 1.25                          #",
            "contrast = 1.5                           #",
        );
    assert_eq!(out, want);
    // a longer value than the comment's room keeps one space
    let out = super::edit_config("[a]\nk = 1 # c\n", &[edit("a", "k", "123456.0")]).expect("edited");
    assert_eq!(out, "[a]\nk = 123456.0 # c\n");
}

#[test]
fn edit_writes_arrays_and_names() {
    let order = vec!["rim".to_owned(), "aerial".to_owned()];
    let out = super::edit_config(
        SHIPPED,
        &[
            edit("host", "order", &super::names_token(&order)),
            edit("aerial", "warmth", &super::numbers_token(&[0.1, -0.2], false)),
        ],
    )
    .expect("edited");
    assert!(out.contains("order = [\"rim\", \"aerial\"]   # run in this order\n"), "{out}");
    assert!(out.contains("warmth = [0.1, -0.2]\n"), "{out}");
    let view = super::ConfigView::parse(&out).expect("parses");
    assert_eq!(view.order(), order);
    assert_eq!(view.numbers("aerial", "warmth", 2), Some(vec![0.1, -0.2]));
}

#[test]
fn edit_adds_missing_keys_and_tables() {
    let out = super::edit_config(
        SHIPPED,
        &[
            edit("aerial", "radial", "1"),
            edit("host", "photos", "false"),
            edit("ink", "strength", "0.5"),
            edit("rim", "p1", "2.0"),
        ],
    )
    .expect("edited");
    assert!(
        out.contains("warmth = 0.0\nradial = 1\n\n# the rim\n"),
        "after the table's last key: {out}"
    );
    assert!(
        out.contains("order = [\"aerial\", \"rim\"]   # run in this order\nphotos = false\n"),
        "{out}"
    );
    assert!(
        out.ends_with("contrast = 1.25                          # 0.5..2 on the rim\np1 = 2.0\n\n[ink]\nstrength = 0.5\n"),
        "{out}"
    );
    // into an empty table, and into an empty file
    assert_eq!(super::edit_config("[a]\n", &[edit("a", "k", "1")]).expect("edited"), "[a]\nk = 1\n");
    assert_eq!(super::edit_config("", &[edit("a", "k", "1")]).expect("edited"), "[a]\nk = 1\n");
    assert_eq!(
        super::edit_config("# only\n", &[edit("a", "k", "1")]).expect("edited"),
        "# only\n\n[a]\nk = 1\n"
    );
    assert_eq!(
        super::edit_config("[b]\nx = 1", &[edit("a", "k", "1")]).expect("edited"),
        "[b]\nx = 1\n\n[a]\nk = 1\n"
    );
}

#[test]
fn edit_keeps_crlf() {
    let text = SHIPPED.replace('\n', "\r\n");
    let out = super::edit_config(
        &text,
        &[edit("aerial", "density", "0.75"), edit("aerial", "new", "1.0"), edit("t", "k", "1")],
    )
    .expect("edited");
    assert!(!out.replace("\r\n", "").contains('\n'), "every line ends in CRLF: {out:?}");
    assert!(
        out.contains("density = 0.75                           # 0..1, how much haze\r\n"),
        "{out:?}"
    );
    assert!(out.contains("warmth = 0.0\r\nnew = 1.0\r\n"), "{out:?}");
    assert!(out.ends_with("\r\n\r\n[t]\r\nk = 1\r\n"), "{out:?}");
}

#[test]
fn edit_refuses_a_file_the_parser_refuses() {
    let broken = "[host]\nenabled = yes\n";
    let e = super::edit_config(broken, &[edit("host", "enabled", "false")]).expect_err("refused");
    assert!(e.contains("line 2"), "{e}");
    assert!(super::edit_config(SHIPPED, &[edit("Host", "x", "1")]).is_err(), "not a table name");
    assert!(super::edit_config(SHIPPED, &[edit("host", "x", "nope")]).is_err(), "not a value");
}

#[test]
fn number_tokens_stay_in_the_subset() {
    use super::number_token as n;
    assert_eq!(n(1.0, false), "1.0");
    assert_eq!(n(0.1 + 0.2, false), "0.3");
    assert_eq!(n(-0.0000001, false), "0.0");
    assert_eq!(n(1e-6, false), "0.000001");
    assert_eq!(n(1024.0, false), "1024.0");
    assert_eq!(n(2.6, true), "3");
    assert_eq!(n(-0.4, true), "0");
    assert_eq!(n(f64::NAN, false), "0.0");
    for v in [0.0, -1.5, 1e21, 123.456789, 0.25] {
        let t = n(v, false);
        assert!(config::parse(&format!("k = {t}"), 1).is_ok(), "{t}");
    }
}

#[test]
fn save_config_writes_atomically_over_the_file_on_disk() {
    let f = Folder::new("save");
    let path = f.0.join("config.toml");
    f.put("config.toml", SHIPPED);
    // a hand edit since the GUI read the file stays
    f.put("config.toml", &SHIPPED.replace("warmth = 0.0", "warmth = 0.5 # by hand"));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).expect("chmod");
    }
    let out = super::save_config(&path, &[edit("aerial", "density", "0.1")]).expect("saved");
    assert_eq!(std::fs::read_to_string(&path).expect("read"), out);
    assert!(out.contains("warmth = 0.5 # by hand") && out.contains("density = 0.1 "), "{out}");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&path).expect("meta").permissions().mode() & 0o777;
        assert_eq!(mode, 0o640, "the file keeps its permissions");
    }
    let left: Vec<_> = std::fs::read_dir(&f.0).expect("dir").flatten().map(|e| e.file_name()).collect();
    assert_eq!(left.len(), 1, "no temporary file left: {left:?}");
    // a broken file is left alone
    f.put("config.toml", "[host\n");
    assert!(super::save_config(&path, &[edit("aerial", "density", "0.2")]).is_err());
    assert_eq!(std::fs::read_to_string(&path).expect("read"), "[host\n");
    // the loader reads what was written, without a warning
    f.put(
        "aerial.wgsl",
        &hinted_body(HINTED)
            .replace("Params { gain", "Params { density: f32, gain")
            .replace("//! gain = 2.0", "//! gain = 2.0\n//! density = 0.5"),
    );
    f.put(
        "config.toml",
        SHIPPED
            .replace(", \"rim\"]", "]")
            .replace("warmth = 0.0\n", "")
            .replace("[rim]\ncontrast = 1.25", "[rim]")
            .as_str(),
    );
    let saved = super::save_config(&path, &[edit("aerial", "density", "0.3")]).expect("saved");
    let mut loader = Loader::new(f.0.clone());
    let (library, report) = loader.load();
    assert_eq!(f32_at(&library.effects[0].params, 0), 0.3);
    let config_problems: Vec<_> = report
        .iter()
        .filter(|(f, _)| f == "config.toml")
        .flat_map(|(_, p)| p.clone())
        .collect();
    assert!(config_problems.is_empty(), "{config_problems:?}\n{saved}");
    assert!(
        saved.contains("density = 0.3                            # 0..1, how much haze"),
        "{saved}"
    );
}

// ------------------------------------------------------- the GPU timer

#[test]
fn timestamps_read_from_misaligned_bytes() {
    // a mapped view 4 bytes off an 8-byte boundary, as a driver may hand it
    let values = [1u64, 0x0123_4567_89ab_cdef, u64::MAX];
    let mut backing = vec![0u64; values.len() + 1];
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut backing);
    for (i, v) in values.iter().enumerate() {
        bytes[4 + 8 * i..12 + 8 * i].copy_from_slice(&v.to_ne_bytes());
    }
    let view = &bytes[4..4 + 8 * values.len()];
    assert!(bytemuck::try_cast_slice::<u8, u64>(view).is_err(), "the view must be misaligned");
    assert_eq!(super::timestamps(view), values);
    // a length that is not a multiple of 8 keeps the whole values only
    assert_eq!(super::timestamps(&view[..12]), [1]);
    assert!(super::timestamps(&[]).is_empty());
}

// ------------------------------------------------------------- with a GPU

fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
}

/// Contract v1's `state` is `read_write` in the prelude, so `fs_main` sees
/// a writable storage buffer it only reads, and the pipelines are built on
/// a worker thread inside error scopes. Draws FULL end to end: `prepare`
/// writes `state`, `fs_main` reads it with the input, the params and the
/// normals rebuilt from a flat depth; nothing reaches the uncaptured error
/// handler. Needs a GPU:
/// `cargo test --workspace user_post::tests::chain_draws -- --ignored`.
#[test]
#[ignore]
fn chain_draws_end_to_end_on_the_gpu() {
    let _gpu = crate::test_gpu::lock();
    let Some((device, queue)) = gpu() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let uncaught = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let seen = uncaught.clone();
    device.on_uncaptured_error(Arc::new(move |e| {
        eprintln!("uncaptured: {e}");
        seen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }));
    let size = (64, 64);
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let c = Arc::new(compiled(FULL));
    let mut warnings = Vec::new();
    let params = library::resolve_params("full", &c, None, &mut warnings);
    let mut chain = super::UserPost::default();
    chain.set_library(Arc::new(Library {
        effects: vec![Effect {
            name: "full".into(),
            compiled: c,
            params,
        }],
        photos: true,
        vr: false,
        ui: Arc::default(),
    }));
    let context = super::Context::default();
    // the first version builds on a worker: the chain joins once it is ready
    let started = std::time::Instant::now();
    while !chain.prepare_with(&device, format, size, context) {
        let failed = chain.gpu.as_ref().and_then(|g| g.effects.first()).and_then(|e| e.failed.as_ref());
        assert!(failed.is_none(), "the GPU refused the effect: {failed:?}");
        assert!(started.elapsed().as_secs() < 20, "the pipelines never came");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let tex = |format, usage| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    let depth = tex(
        wgpu::TextureFormat::Depth32Float,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    );
    let depth_view = depth.create_view(&Default::default());
    let out = tex(format, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
    let out_view = out.create_view(&Default::default());
    let mut encoder = device.create_command_encoder(&Default::default());
    // the scene: a flat depth of 0.5, an input image of (0.25, 0.25, 0.25)
    let clear = |encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView| {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.25,
                        g: 0.25,
                        b: 0.25,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    };
    clear(&mut encoder, chain.input().expect("input"));
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: None,
        color_attachments: &[],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(0.5),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    let frame = super::FrameInputs {
        depth: &depth_view,
        mesh_normals: None,
        water: None,
        alpha_depth: None,
        proj: glam::camera::rh::proj::directx::perspective_infinite_reverse(1.0, 1.0, 0.5),
        view: glam::Mat4::IDENTITY,
        draw_distance: 256.0,
        time: 1.0,
        tile: [0.0, 0.0, 1.0, 1.0],
        pixel_scale: 1.0,
        depth_jitter: glam::Vec2::ZERO,
        output_size: size,
        taa: false,
        frame_index: 0,
        motion: None,
        reproject: glam::Mat4::IDENTITY,
        cut: false,
    };
    let passes = chain.encode(&device, &queue, &mut encoder, &frame, &out_view);
    assert_eq!(passes, 3, "normals, prepare, fs_main");
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: crate::readback::readback_size(256 * u64::from(size.1)),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        out.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    readback.map_async(wgpu::MapMode::Read, .., |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let view = readback.get_mapped_range(..).expect("mapped");
    let px = &view[32 * 256 + 32 * 4..32 * 256 + 32 * 4 + 4];
    // 0.25 * gain 2 + state 0.25 in red, 0.5 in green and blue
    assert!((i32::from(px[0]) - 191).abs() <= 2, "{px:?}");
    assert!((i32::from(px[1]) - 128).abs() <= 2, "{px:?}");
    assert!((i32::from(px[2]) - 128).abs() <= 2, "{px:?}");
    assert_eq!(uncaught.load(std::sync::atomic::Ordering::Relaxed), 0);
}

/// Writes `values[x]` as the depth of `width` x 1 pixels (a Depth32Float
/// image takes no buffer copy, so it is drawn).
fn depth_row(device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, values: &[f32]) -> wgpu::Texture {
    let list: Vec<String> = values.iter().map(|v| format!("{v:?}")).collect();
    let src = format!(
        "const V = array<f32, {n}>({list});\n\
         @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {{\n\
         let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));\n\
         return vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);\n}}\n\
         @fragment fn fs(@builtin(position) p: vec4<f32>) -> @builtin(frag_depth) f32 {{\n\
         var v = V;\nreturn v[u32(p.x)];\n}}\n",
        n = values.len(),
        list = list.join(", ")
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let pipeline = device.create_render_pipeline_cached(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: Default::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[],
        }),
        multiview_mask: None,
        cache: None,
    });
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: values.len() as u32,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: None,
        color_attachments: &[],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(0.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(&pipeline);
    pass.draw(0..3, 0..1);
    drop(pass);
    texture
}

/// Writes the effects' depth in red; reads the normals so their pass runs
/// with the water too (w only: rebuilt normals of a 1-row image are NaN).
const DEPTH_OUT: &str = "//! api = 1\n@fragment\nfn fs_main(in: FsIn) -> @location(0) vec4<f32> {\n    let px = pixel_of(in);\n    return vec4<f32>(raw_depth(px) + view_normal(px).w * 0.0, 0.0, 0.0, 1.0);\n}\n";

/// The water depth pass (`fs_depth`): above water the effects' depth is the
/// nearer of the scene and the water surface, the surface's depth is
/// near / its distance (1 inside the near plane), and an exclusion face or
/// no water keeps the scene's. Without water the pass does not run and
/// its image is not made. Needs a GPU:
/// `cargo test --workspace user_post::tests::water_depth -- --ignored`.
#[test]
#[ignore]
fn water_depth_merges_the_surface_on_the_gpu() {
    let _gpu = crate::test_gpu::lock();
    let Some((device, queue)) = gpu() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let uncaught = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let seen = uncaught.clone();
    device.on_uncaptured_error(Arc::new(move |e| {
        eprintln!("uncaptured: {e}");
        seen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }));
    let size = (5u32, 1u32);
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let mut chain = ready_chain(&device, DEPTH_OUT, super::ChainAa::Off, format, size);
    // near 0.5: water at 10 m is 0.05; at 0.1 m, inside the near plane, 1
    let at = |dist: f32| 1.0 / (1.0 + dist);
    // sea to the sky, ground without water, a hull over the sea, land in
    // front of the sea, the surface inside the near plane
    let scene = [0.0, 0.5, 0.0, 0.25, 0.0];
    let interface = [at(10.0), 0.0, at(10.0), at(10.0), at(0.1)];
    let mask = [255u8, 255, 0, 255, 255];
    let mut encoder = device.create_command_encoder(&Default::default());
    let scene_tex = depth_row(&device, &mut encoder, &scene);
    let interface_tex = depth_row(&device, &mut encoder, &interface);
    queue.submit([encoder.finish()]);
    let mask_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        mask_tex.as_image_copy(),
        &mask,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size.0),
            rows_per_image: None,
        },
        mask_tex.size(),
    );
    let scene_view = scene_tex.create_view(&Default::default());
    let interface_view = interface_tex.create_view(&Default::default());
    let mask_view = mask_tex.create_view(&Default::default());
    // runs the chain and reads the red channel of its output back
    let run = |chain: &mut super::UserPost, water: Option<super::WaterInputs<'_>>| -> (u32, Vec<u8>) {
        let frame = super::FrameInputs {
            depth: &scene_view,
            mesh_normals: None,
            water,
            alpha_depth: None,
            proj: glam::camera::rh::proj::directx::perspective_infinite_reverse(1.0, 1.0, 0.5),
            view: glam::Mat4::IDENTITY,
            draw_distance: 256.0,
            time: 1.0,
            tile: [0.0, 0.0, 1.0, 1.0],
            pixel_scale: 1.0,
            depth_jitter: glam::Vec2::ZERO,
            output_size: size,
            taa: false,
            frame_index: 0,
            motion: None,
            reproject: glam::Mat4::IDENTITY,
            cut: false,
        };
        let out = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let out_view = out.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        let passes = chain.encode(&device, &queue, &mut encoder, &frame, &out_view);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: crate::readback::readback_size(256),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            out.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: None,
                },
            },
            out.size(),
        );
        queue.submit([encoder.finish()]);
        readback.map_async(wgpu::MapMode::Read, .., |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let view = readback.get_mapped_range(..).expect("mapped");
        let red = (0..size.0 as usize).map(|x| view[4 * x]).collect();
        (passes, red)
    };
    let near = |got: &[u8], want: &[u8]| got.iter().zip(want).all(|(g, w)| (i32::from(*g) - i32::from(*w)).abs() <= 1);

    let (passes, red) = run(&mut chain, None);
    assert_eq!(passes, 2, "normals, fs_main");
    assert!(near(&red, &[0, 128, 0, 64, 0]), "no water: the scene's depth, {red:?}");
    let targets = chain.gpu.as_ref().and_then(|g| g.targets.as_ref());
    assert!(targets.is_some_and(|t| t.depth.is_none()), "no water: no water depth image");

    let water = super::WaterInputs {
        interface: &interface_view,
        mask: &mask_view,
    };
    let (passes, red) = run(&mut chain, Some(water));
    assert_eq!(passes, 3, "depth, normals, fs_main");
    // 0.05 * 255 = 12.75
    assert!(near(&red, &[13, 128, 0, 64, 255]), "with water: {red:?}");
    assert_eq!(uncaught.load(std::sync::atomic::Ordering::Relaxed), 0);
}

/// Contract v2's `alpha_depth`: given, a v2 effect reads it with
/// FLAG_ALPHA_DEPTH and `alpha_in_front` finds a face nearer than the scene
/// beyond its tolerance only (the sky included); not given, it reads 0 and
/// nothing is in front; a v1 effect sees neither the flag nor api 2. Needs
/// a GPU: `cargo test --workspace user_post::tests::alpha_depth -- --ignored`.
#[test]
#[ignore]
fn alpha_depth_reaches_v2_effects_only_on_the_gpu() {
    let _gpu = crate::test_gpu::lock();
    let Some((device, queue)) = gpu() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let uncaught = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let seen = uncaught.clone();
    device.on_uncaptured_error(Arc::new(move |e| {
        eprintln!("uncaptured: {e}");
        seen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }));
    let size = (4u32, 1u32);
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let mut v2 = ready_chain(&device, ALPHA_OUT, super::ChainAa::Off, format, size);
    let mut v1 = ready_chain(&device, FLAGS_V1, super::ChainAa::Off, format, size);
    // near 0.5: the scene at 10 m (0.05), the sky in the last pixel; a pane
    // at 5 m (0.1), the scene's own face, one 1 % nearer (inside the 3 %),
    // a pane against the sky
    let scene = [0.05, 0.05, 0.05, 0.0];
    let alpha = [0.1, 0.05, 0.0505, 0.1];
    let mut encoder = device.create_command_encoder(&Default::default());
    let scene_tex = depth_row(&device, &mut encoder, &scene);
    let alpha_tex = depth_row(&device, &mut encoder, &alpha);
    queue.submit([encoder.finish()]);
    let scene_view = scene_tex.create_view(&Default::default());
    let alpha_view = alpha_tex.create_view(&Default::default());
    // runs a chain and reads its output back, RGBA per pixel
    let run = |chain: &mut super::UserPost, alpha_depth: Option<&wgpu::TextureView>| -> Vec<[u8; 4]> {
        let frame = super::FrameInputs {
            depth: &scene_view,
            mesh_normals: None,
            water: None,
            alpha_depth,
            proj: glam::camera::rh::proj::directx::perspective_infinite_reverse(1.0, 1.0, 0.5),
            view: glam::Mat4::IDENTITY,
            draw_distance: 256.0,
            time: 1.0,
            tile: [0.0, 0.0, 1.0, 1.0],
            pixel_scale: 1.0,
            depth_jitter: glam::Vec2::ZERO,
            output_size: size,
            taa: false,
            frame_index: 0,
            motion: None,
            reproject: glam::Mat4::IDENTITY,
            cut: false,
        };
        let out = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let out_view = out.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        chain.encode(&device, &queue, &mut encoder, &frame, &out_view);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: crate::readback::readback_size(256),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            out.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: None,
                },
            },
            out.size(),
        );
        queue.submit([encoder.finish()]);
        readback.map_async(wgpu::MapMode::Read, .., |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let view = readback.get_mapped_range(..).expect("mapped");
        (0..size.0 as usize)
            .map(|x| [view[4 * x], view[4 * x + 1], view[4 * x + 2], view[4 * x + 3]])
            .collect()
    };
    let near = |got: &[[u8; 4]], want: &[[u8; 4]]| {
        got.iter()
            .flatten()
            .zip(want.iter().flatten())
            .all(|(g, w)| (i32::from(*g) - i32::from(*w)).abs() <= 1)
    };

    // 0.1 * 255 = 25.5, 0.05 * 255 = 12.75; api 2 / 4 = 127.5
    let got = run(&mut v2, Some(&alpha_view));
    let want = [[26, 255, 128, 255], [13, 0, 128, 255], [13, 0, 128, 255], [26, 255, 128, 255]];
    assert!(near(&got, &want), "v2 with the blended faces: {got:?}");
    let got = run(&mut v2, None);
    assert!(near(&got, &[[0, 0, 128, 0]; 4]), "v2 without them: {got:?}");
    // api 1 / 4 = 63.75
    let got = run(&mut v1, Some(&alpha_view));
    assert!(
        near(&got, &[[0, 64, 13, 255], [0, 64, 13, 255], [0, 64, 13, 255], [0, 64, 0, 255]]),
        "v1: {got:?}"
    );
    assert_eq!(uncaught.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn chain_aa_follows_the_setting() {
    use super::ChainAa;
    use crate::types::AntiAliasing as Aa;
    let cases = [
        (Aa::None, 1, false, ChainAa::Off),
        (Aa::Smaa, 1, false, ChainAa::Off),
        (Aa::Smaa, 1, true, ChainAa::Off),
        (Aa::Fxaa, 1, false, ChainAa::Fxaa),
        (Aa::Fxaa, 1, true, ChainAa::Fxaa),
        // the effects read the jittered depth: resolved over the frames
        (Aa::Taa, 1, false, ChainAa::Temporal),
        // a photo's TAA history starts over: its scene is not resolved
        (Aa::Taa, 1, true, ChainAa::Fxaa),
        // FSR in the TAA pass's place, at any quality
        (Aa::Fsr(crate::types::FsrQuality::NativeAa), 1, false, ChainAa::Temporal),
        (Aa::Fsr(crate::types::FsrQuality::Performance), 1, false, ChainAa::Temporal),
        (Aa::Fsr(crate::types::FsrQuality::Quality), 1, true, ChainAa::Fxaa),
        (Aa::Msaa(4), 4, false, ChainAa::FxaaMasked),
        (Aa::Msaa(4), 4, true, ChainAa::FxaaMasked),
        // MSAA asked, none on this GPU: the scene is aliased, so is the chain
        (Aa::Msaa(4), 1, false, ChainAa::Off),
    ];
    for (aa, samples, photo, want) in cases {
        let got = ChainAa::for_setting_with(aa, samples, photo, super::TemporalMode::Auto);
        assert_eq!(got, want, "{aa:?} x{samples} photo {photo}");
    }
}

#[test]
fn temporal_finish_follows_its_variable() {
    use super::{ChainAa, TemporalMode};
    use crate::types::AntiAliasing as Aa;
    for (v, want) in [
        (None, TemporalMode::Auto),
        (Some("1"), TemporalMode::Auto),
        (Some(" 0 "), TemporalMode::Off),
        (Some("off"), TemporalMode::Off),
        (Some("always"), TemporalMode::Always),
        (Some("yes"), TemporalMode::Auto),
    ] {
        assert_eq!(TemporalMode::parse(v), want, "{v:?}");
    }
    let fsr = Aa::Fsr(crate::types::FsrQuality::Quality);
    let cases = [
        // 0: the masked FXAA under TAA and FSR, as before
        (Aa::Taa, 1, false, TemporalMode::Off, ChainAa::FxaaMasked),
        (fsr, 1, false, TemporalMode::Off, ChainAa::FxaaMasked),
        (Aa::Taa, 1, true, TemporalMode::Off, ChainAa::Fxaa),
        // always: every AA but a photo's
        (Aa::None, 1, false, TemporalMode::Always, ChainAa::Temporal),
        (Aa::Smaa, 1, false, TemporalMode::Always, ChainAa::Temporal),
        (Aa::Fxaa, 1, false, TemporalMode::Always, ChainAa::Temporal),
        (Aa::Msaa(4), 4, false, TemporalMode::Always, ChainAa::Temporal),
        (Aa::Msaa(4), 1, false, TemporalMode::Always, ChainAa::Temporal),
        (Aa::Taa, 1, false, TemporalMode::Always, ChainAa::Temporal),
        (Aa::Taa, 1, true, TemporalMode::Always, ChainAa::Fxaa),
        (Aa::Smaa, 1, true, TemporalMode::Always, ChainAa::Off),
        (Aa::Msaa(4), 4, true, TemporalMode::Always, ChainAa::FxaaMasked),
        // auto: the others as before
        (Aa::None, 1, false, TemporalMode::Auto, ChainAa::Off),
        (Aa::Msaa(8), 8, false, TemporalMode::Auto, ChainAa::FxaaMasked),
    ];
    for (aa, samples, photo, mode, want) in cases {
        assert_eq!(
            ChainAa::for_setting_with(aa, samples, photo, mode),
            want,
            "{aa:?} x{samples} photo {photo} {mode:?}"
        );
    }
}

#[test]
fn the_chain_draws_once_at_its_slot_or_at_its_look() {
    use super::{Context, Place, place};
    let view = Context::default();
    let photo = Context {
        photo: true,
        ..Default::default()
    };
    let eye = Context {
        vr_eye: true,
        ..Default::default()
    };
    // (effects, [host] photos, [host] vr, frame, the look in the stack)
    let cases = [
        // the normal view and the photo mode's live view without the look
        (true, false, false, view, false, Place::Normal),
        (true, true, false, view, false, Place::Normal),
        // photos without the look: [host] photos decides
        (true, false, false, photo, false, Place::Off),
        (true, true, false, photo, false, Place::Normal),
        // the look in the stack: only there, on the live view and on the
        // photo, whatever [host] photos says
        (true, false, false, view, true, Place::Layer),
        (true, true, false, view, true, Place::Layer),
        (true, false, false, photo, true, Place::Layer),
        (true, true, false, photo, true, Place::Layer),
        // the master switch (or no effect): nowhere, the look draws nothing
        (false, true, true, view, false, Place::Off),
        (false, true, true, photo, true, Place::Off),
        (false, true, true, view, true, Place::Off),
        // a VR eye without [host] vr
        (true, true, false, eye, false, Place::Off),
        (true, true, false, eye, true, Place::Off),
        (true, true, true, eye, false, Place::Normal),
    ];
    for (effects, photos, vr, context, layer, want) in cases {
        assert_eq!(
            place(effects, photos, vr, context, layer),
            want,
            "effects {effects} photos {photos} vr {vr} {context:?} layer {layer}"
        );
    }
}

#[test]
fn chain_aa_at_its_look_never_smooths_the_image_twice() {
    use super::{ChainAa, TemporalMode};
    use crate::types::AntiAliasing as Aa;
    let fsr = Aa::Fsr(crate::types::FsrQuality::Quality);
    let cases = [
        // the frame's AA has run before the looks: only the chain's edges
        (Aa::Smaa, 1, false, TemporalMode::Auto, ChainAa::FxaaMasked),
        (Aa::Fxaa, 1, false, TemporalMode::Auto, ChainAa::FxaaMasked),
        (Aa::Msaa(4), 4, false, TemporalMode::Auto, ChainAa::FxaaMasked),
        (Aa::Smaa, 1, true, TemporalMode::Auto, ChainAa::FxaaMasked),
        // the jittered depth: resolved over the frames on the live view
        (Aa::Taa, 1, false, TemporalMode::Auto, ChainAa::Temporal),
        (fsr, 1, false, TemporalMode::Auto, ChainAa::Temporal),
        // a photo is one frame
        (Aa::Taa, 1, true, TemporalMode::Auto, ChainAa::FxaaMasked),
        (fsr, 1, true, TemporalMode::Always, ChainAa::FxaaMasked),
        // AURORA_SHADER_TEMPORAL=0 and always
        (Aa::Taa, 1, false, TemporalMode::Off, ChainAa::FxaaMasked),
        (Aa::Smaa, 1, false, TemporalMode::Always, ChainAa::Temporal),
        // an aliased scene: an aliased chain
        (Aa::None, 1, false, TemporalMode::Auto, ChainAa::Off),
        (Aa::None, 1, true, TemporalMode::Auto, ChainAa::Off),
        (Aa::Msaa(4), 1, false, TemporalMode::Auto, ChainAa::Off),
    ];
    for (aa, samples, photo, mode, want) in cases {
        assert_eq!(
            ChainAa::for_layer_with(aa, samples, photo, mode),
            want,
            "{aa:?} x{samples} photo {photo} {mode:?}"
        );
    }
}

/// Black right of the diagonal, white left of it: a hard staircase.
const DIAGONAL: &str = "//! api = 1\n@fragment\nfn fs_main(in: FsIn) -> @location(0) vec4<f32> {\n    return select(vec4<f32>(1.0), vec4<f32>(0.0, 0.0, 0.0, 1.0), in.pos.x > in.pos.y);\n}\n";

/// A chain of one effect, its pipelines built.
fn ready_chain(device: &wgpu::Device, src: &str, aa: super::ChainAa, format: wgpu::TextureFormat, size: (u32, u32)) -> super::UserPost {
    let c = Arc::new(compiled(src));
    let mut warnings = Vec::new();
    let params = library::resolve_params("e", &c, None, &mut warnings);
    let mut chain = super::UserPost::default();
    chain.set_library(Arc::new(Library {
        effects: vec![Effect {
            name: "e".into(),
            compiled: c,
            params,
        }],
        photos: true,
        vr: false,
        ui: Arc::default(),
    }));
    let context = super::Context { aa, ..Default::default() };
    let started = std::time::Instant::now();
    while !chain.prepare_with(device, format, size, context) {
        let failed = chain.gpu.as_ref().and_then(|g| g.effects.first()).and_then(|e| e.failed.as_ref());
        assert!(failed.is_none(), "the GPU refused the effect: {failed:?}");
        assert!(started.elapsed().as_secs() < 20, "the pipelines never came");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    chain
}

/// The AA finish (`ChainAa`): FXAA smooths the edges a chain draws, masked
/// or not, and the masked one leaves the edges of its input (the resolved
/// scene) as they are; without it the chain's staircase stays. Needs a GPU:
/// `cargo test --workspace user_post::tests::aa_finish -- --ignored`.
#[test]
#[ignore]
fn aa_finish_smooths_only_the_chain_edges_on_the_gpu() {
    use super::ChainAa;
    let _gpu = crate::test_gpu::lock();
    let Some((device, queue)) = gpu() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let uncaught = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let seen = uncaught.clone();
    device.on_uncaptured_error(Arc::new(move |e| {
        eprintln!("uncaptured: {e}");
        seen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }));
    let size = (64u32, 64u32);
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let tex = |format, usage| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    let depth = tex(
        wgpu::TextureFormat::Depth32Float,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    );
    let depth_view = depth.create_view(&Default::default());
    let frame = super::FrameInputs {
        depth: &depth_view,
        mesh_normals: None,
        water: None,
        alpha_depth: None,
        proj: glam::camera::rh::proj::directx::perspective_infinite_reverse(1.0, 1.0, 0.5),
        view: glam::Mat4::IDENTITY,
        draw_distance: 256.0,
        time: 1.0,
        tile: [0.0, 0.0, 1.0, 1.0],
        pixel_scale: 1.0,
        depth_jitter: glam::Vec2::ZERO,
        output_size: size,
        taa: false,
        frame_index: 0,
        motion: None,
        reproject: glam::Mat4::IDENTITY,
        cut: false,
    };
    // runs `chain` (after `scene`, which draws its input, if any) and reads
    // the red channel of its output back
    let run = |chain: &mut super::UserPost, scene: Option<&mut super::UserPost>| -> (u32, Vec<u8>) {
        let out = tex(format, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
        let out_view = out.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        match scene {
            Some(scene) => {
                let input = chain.input().expect("input");
                assert_eq!(scene.encode(&device, &queue, &mut encoder, &frame, input), 1);
            }
            None => {
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: chain.input().expect("input"),
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
            }
        }
        let passes = chain.encode(&device, &queue, &mut encoder, &frame, &out_view);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: crate::readback::readback_size(256 * u64::from(size.1)),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            out.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        readback.map_async(wgpu::MapMode::Read, .., |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let view = readback.get_mapped_range(..).expect("mapped");
        (passes, view.chunks(4).map(|px| px[0]).collect())
    };
    // pixels neither black nor white: the staircase was smoothed
    let soft = |red: &[u8]| red.iter().filter(|&&r| (16..240).contains(&r)).count();
    let staircase: Vec<u8> = (0..size.1)
        .flat_map(|y| (0..size.0).map(move |x| if x > y { 0 } else { 255 }))
        .collect();

    // AA off: the chain's own staircase, untouched, drawn straight to out
    let mut chain = ready_chain(&device, DIAGONAL, ChainAa::Off, format, size);
    let (passes, red) = run(&mut chain, None);
    assert_eq!(passes, 1);
    assert_eq!(red, staircase);

    // FXAA (whole image) and masked FXAA: the chain drew the edge, both smooth it
    for aa in [ChainAa::Fxaa, ChainAa::FxaaMasked] {
        let mut chain = ready_chain(&device, DIAGONAL, aa, format, size);
        let (passes, red) = run(&mut chain, None);
        assert_eq!(passes, 2, "{aa:?}: the effect, the finish");
        assert!(soft(&red) >= 32, "{aa:?}: {} soft pixels", soft(&red));
        assert_eq!(red[50 * 64 + 10], 255, "{aa:?}: away from the edge");
        assert_eq!(red[10 * 64 + 50], 0, "{aa:?}: away from the edge");
    }

    // the edge is in the input (a resolved scene), the chain passes it
    // through: masked FXAA leaves it, whole-image FXAA smooths it
    let mut scene = ready_chain(&device, DIAGONAL, ChainAa::Off, format, size);
    let mut masked = ready_chain(&device, PASSTHROUGH, ChainAa::FxaaMasked, format, size);
    let (_, red) = run(&mut masked, Some(&mut scene));
    assert_eq!(red, staircase, "masked FXAA touched the scene");
    let mut whole = ready_chain(&device, PASSTHROUGH, ChainAa::Fxaa, format, size);
    let (_, red) = run(&mut whole, Some(&mut scene));
    assert!(soft(&red) >= 32, "{} soft pixels", soft(&red));
    assert_eq!(uncaught.load(std::sync::atomic::Ordering::Relaxed), 0);
}

/// wgpu 30's error scopes are per thread (its `std` feature): an error on a
/// build worker lands in the worker's scope, never in one the frame thread
/// has open, and the other way round. Needs a GPU:
/// `cargo test --workspace user_post::tests::error_scopes -- --ignored`.
#[test]
#[ignore]
fn error_scopes_are_per_thread() {
    let _gpu = crate::test_gpu::lock();
    let Some((device, _queue)) = gpu() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let bad = |device: &wgpu::Device| {
        // MAP_READ with MAP_WRITE: invalid without MAPPABLE_PRIMARY_BUFFERS
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("invalid on purpose"),
            size: crate::readback::readback_size(16),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE,
            mapped_at_creation: false,
        })
    };
    let main_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let worker = device.clone();
    let caught = std::thread::spawn(move || {
        let scope = worker.push_error_scope(wgpu::ErrorFilter::Validation);
        let _b = bad(&worker);
        pollster::block_on(scope.pop()).is_some()
    })
    .join()
    .expect("worker");
    assert!(caught, "the worker's own scope caught its error");
    assert!(
        pollster::block_on(main_scope.pop()).is_none(),
        "the main thread's scope saw nothing"
    );
}

/// A line one pixel wide that lands on column 8 or 9 as the jitter moves,
/// as an ink line drawn from the jittered depth does.
const JITTERED_LINE: &str = "//! api = 1\n@fragment\nfn fs_main(in: FsIn) -> @location(0) vec4<f32> {\n    let px = pixel_of(in);\n    let at = select(8, 9, frame.jitter.x > 0.0);\n    return select(textureLoad(color, px, 0), vec4<f32>(0.0, 0.0, 0.0, 1.0), px.x == at);\n}\n";

/// The temporal finish (`ChainAa::Temporal`): a line the jitter moves by a
/// pixel each frame flickers black and white under the masked FXAA; resolved
/// over the frames it settles to a steady grey over both columns, within the
/// clamp, and a cut (or the first frame) starts over from the current one.
/// Needs a GPU: `cargo test --workspace user_post::tests::temporal_finish -- --ignored`.
#[test]
#[ignore]
fn temporal_finish_steadies_a_jittered_line_on_the_gpu() {
    use super::ChainAa;
    let _gpu = crate::test_gpu::lock();
    let Some((device, queue)) = gpu() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let uncaught = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let seen = uncaught.clone();
    device.on_uncaptured_error(Arc::new(move |e| {
        eprintln!("uncaptured: {e}");
        seen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }));
    let size = (16u32, 4u32);
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let tex = |format, usage| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    let depth = tex(
        wgpu::TextureFormat::Depth32Float,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    );
    let depth_view = depth.create_view(&Default::default());
    // runs one frame (jitter > 0: the line on column 9) and reads row 1 back
    let run = |chain: &mut super::UserPost, aa: ChainAa, jitter: f32, cut: bool| -> Vec<u8> {
        let context = super::Context { aa, ..Default::default() };
        assert!(chain.prepare_with(&device, format, size, context));
        let frame = super::FrameInputs {
            depth: &depth_view,
            mesh_normals: None,
            water: None,
            alpha_depth: None,
            proj: glam::camera::rh::proj::directx::perspective_infinite_reverse(1.0, 1.0, 0.5),
            view: glam::Mat4::IDENTITY,
            draw_distance: 256.0,
            time: 1.0,
            tile: [0.0, 0.0, 1.0, 1.0],
            pixel_scale: 1.0,
            depth_jitter: glam::Vec2::new(jitter, 0.0),
            output_size: size,
            taa: true,
            frame_index: 0,
            motion: None,
            reproject: glam::Mat4::IDENTITY,
            cut,
        };
        let out = tex(format, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
        let out_view = out.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        // the scene: white, and the sky's depth
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: chain.input().expect("input"),
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        assert_eq!(
            chain.encode(&device, &queue, &mut encoder, &frame, &out_view),
            2,
            "the effect, the finish"
        );
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: crate::readback::readback_size(256 * u64::from(size.1)),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            out.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: None,
                },
            },
            out.size(),
        );
        queue.submit([encoder.finish()]);
        readback.map_async(wgpu::MapMode::Read, .., |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let view = readback.get_mapped_range(..).expect("mapped");
        (0..size.0 as usize).map(|x| view[256 + 4 * x]).collect()
    };
    let jitter = |i: usize| if i.is_multiple_of(2) { -0.25 } else { 0.25 };

    // the masked FXAA: the line jumps a column each frame
    let mut masked = ready_chain(&device, JITTERED_LINE, ChainAa::FxaaMasked, format, size);
    let a = run(&mut masked, ChainAa::FxaaMasked, jitter(0), false);
    let b = run(&mut masked, ChainAa::FxaaMasked, jitter(1), false);
    assert!(a[8].abs_diff(b[8]) > 60, "masked: {a:?} then {b:?}");

    // temporal: the first frame is the current one, then both columns
    // settle to grey, a few levels apart from one frame to the next
    let mut temporal = ready_chain(&device, JITTERED_LINE, ChainAa::Temporal, format, size);
    let first = run(&mut temporal, ChainAa::Temporal, jitter(0), false);
    assert_eq!((first[8], first[9], first[2]), (0, 255, 255), "the first frame: {first:?}");
    let mut rows = Vec::new();
    for i in 1..80 {
        rows.push(run(&mut temporal, ChainAa::Temporal, jitter(i), false));
    }
    let (a, b) = (&rows[rows.len() - 2], &rows[rows.len() - 1]);
    for x in [8, 9] {
        assert!((100..160).contains(&a[x]), "column {x} grey: {a:?}");
        assert!(a[x].abs_diff(b[x]) <= 20, "column {x} steady: {a:?} then {b:?}");
    }
    assert_eq!((a[2], a[13]), (255, 255), "away from the line: {a:?}");
    // a cut starts over from the current frame
    let cut = run(&mut temporal, ChainAa::Temporal, jitter(0), true);
    assert_eq!((cut[8], cut[9]), (0, 255), "after a cut: {cut:?}");
    assert_eq!(uncaught.load(std::sync::atomic::Ordering::Relaxed), 0);
}
