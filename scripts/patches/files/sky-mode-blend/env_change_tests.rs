//! Whole-frame tests of an environment change (local patch set,
//! scripts/patches/123-sky-mode-blend.toml): everything the
//! selector and the environment hand to the renderer, compared field by
//! field from the frames' own Debug forms, so that a field added later is
//! checked too.
//!
//! For each pair of skies and waters, a change through the selector:
//! (a) the first frame at the change's full progress equals the target as
//!     it shows on its own on the next tick;
//! (b) the first frame of the change equals the source as it showed;
//! (c) between two frames no value moves by more than the change's fastest
//!     pace (pi / 2 times the linear one: patch 999's ease; a linear change
//!     keeps under it) allows, times its whole
//!     excursion over the change (and a margin for the values that are not
//!     linear in it): no single-frame step anywhere.
//!
//! What may legitimately switch is named in `switches`, with its reason.
// aurora-local-patch: sky-mode-blend-tests

use super::eep::{Settings, SkyFrame, WaterFrame};
use super::eep_env::{EnvSelector, FrameInput};
use super::env::Environment;
use super::env_select::SettingsKind;
use aurora_render::Tonemapper;
use aurora_render::sky_mode;
use glam::{Quat, Vec2, Vec3};
use std::collections::BTreeMap;
use std::sync::Arc;
use uuid::Uuid;

/// A value of the Debug form: a number, or a word (bool, enum, id).
#[derive(Debug, Clone, PartialEq)]
enum Leaf {
    Num(f64),
    Word(String),
}

/// The leaves of a Debug form by their path (field names, and the index of
/// a value within its field).
fn leaves(debug: &str) -> BTreeMap<String, Leaf> {
    let mut out = BTreeMap::new();
    let mut stack: Vec<(String, usize)> = vec![("frame".into(), 0)];
    let mut word = String::new();
    let mut pending: Option<String> = None;
    let flush = |word: &mut String, stack: &mut Vec<(String, usize)>, out: &mut BTreeMap<String, Leaf>| {
        if word.is_empty() {
            return;
        }
        let w = std::mem::take(word);
        let path = path_of(stack);
        if let Some(top) = stack.last_mut() {
            let leaf = match w.parse::<f64>() {
                Ok(v) => Leaf::Num(v),
                Err(_) => Leaf::Word(w),
            };
            out.insert(format!("{path}#{}", top.1), leaf);
            top.1 += 1;
        }
    };
    for c in debug.chars() {
        match c {
            ':' => {
                pending = Some(std::mem::take(&mut word));
            }
            '{' | '(' | '[' => {
                // `Name {`: the type name is no value
                word.clear();
                let name = pending.take().unwrap_or_else(|| {
                    let top = stack.last().map_or(0, |t| t.1);
                    if let Some(t) = stack.last_mut() {
                        t.1 += 1;
                    }
                    format!("{top}")
                });
                stack.push((name, 0));
            }
            '}' | ')' | ']' => {
                if let Some(name) = pending.take() {
                    stack.push((name, 0));
                    flush(&mut word, &mut stack, &mut out);
                    stack.pop();
                } else {
                    flush(&mut word, &mut stack, &mut out);
                }
                stack.pop();
            }
            ',' => {
                if let Some(name) = pending.take() {
                    stack.push((name, 0));
                    flush(&mut word, &mut stack, &mut out);
                    stack.pop();
                } else {
                    flush(&mut word, &mut stack, &mut out);
                }
            }
            c if c.is_whitespace() => {}
            c => word.push(c),
        }
    }
    out
}

fn path_of(stack: &[(String, usize)]) -> String {
    stack.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(".")
}

/// Everything a frame hands over: the blended frames, the environment the
/// renderer gets and what the renderer derives from its sky (tone mapping,
/// exposure, the sky rays' moon factor).
fn frame_leaves(sky: &SkyFrame, water: &WaterFrame) -> BTreeMap<String, Leaf> {
    let env = Environment::from_eep(sky, water, Vec2::ZERO, 96.0).with_colored_ambient(sky, false);
    let s = &env.sky;
    let ll = if s.classic {
        1.0
    } else if s.no_post {
        2.0
    } else {
        0.0
    };
    // (each as it acts: the mix of a frame without a tone curve, and the
    // dark scene term of an exposure without a range, do nothing)
    let post = |hdr: bool, mapper: Tonemapper| {
        let mut p = sky_mode::post(s, hdr, mapper, 0.7);
        if p.tonemapper > 1.5 {
            p.tonemap_mix = 0.0;
        }
        p
    };
    let range = sky_mode::exposure_range(s, true);
    let derived = (
        post(true, Tonemapper::KhronosNeutral),
        post(true, Tonemapper::Aces),
        post(false, Tonemapper::KhronosNeutral),
        range,
        if range.1 > 1.0 { sky_mode::exposure_dark(s) } else { 0.0 },
        sky_mode::rays_moon(s, ll),
        ll,
    );
    let lit = lit_levels(&env, ll);
    leaves(&format!("{env:?} {derived:?} {lit:?}"))
}

/// The level of a white surface as the frame's lighting branch shades it,
/// before the exposure: edge-on to the light, facing it, and facing up.
/// The classic and linear branches as common.wgsl writes them
/// (ll_atmos_core with the classic gains, ll_classic_sun_ambient,
/// ll_final_scale; no shadow, occlusion, haze, specular nor local lights),
/// the HDR branch for a face without a PBR material (shade_pbr's `legacy`:
/// the sun colour / 3 x n.l, plus the hemisphere for the probes'
/// irradiance).
fn lit_levels(env: &Environment, ll: f32) -> [f32; 3] {
    lit_colours(env, ll).map(|c| c.max(Vec3::ZERO).dot(Vec3::new(0.2126, 0.7152, 0.0722)))
}

/// The colours behind [`lit_levels`].
fn lit_colours(env: &Environment, ll: f32) -> [Vec3; 3] {
    let lum = |c: Vec3| c.max(Vec3::ZERO).dot(Vec3::new(0.2126, 0.7152, 0.0722));
    let s2l = |c: Vec3| {
        let f = |x: f32| {
            let x = x.max(0.0);
            if x <= 0.04045 { x / 12.92 } else { ((x + 0.055) / 1.055).powf(2.4) }
        };
        Vec3::new(f(c.x), f(c.y), f(c.z))
    };
    let l2s = |x: f32| {
        let x = x.max(0.0);
        if x <= 0.0031308 {
            x * 12.92
        } else {
            1.055 * x.powf(1.0 / 2.4) - 0.055
        }
    };
    let up = env.sun_dir.z.max(0.0);
    let normals = [
        (0.0f32, 1.0f32),
        (1.0, 0.75),
        (up, 1.0 - (0.5 * env.sun_dir.z.abs().min(1.0)).powi(2)),
    ];
    let s = &env.sky;
    if ll < 0.5 {
        return normals.map(|(nl, _)| env.sun_color / 3.0 * nl + env.zenith);
    }
    let [gs, ga] = sky_mode::classic_gains(s);
    let atten = (s.blue_density + Vec3::splat(s.haze_density * 0.25)) * (s.density_multiplier * s.max_y);
    let k = 1.0 / s.light_norm.z.max(1e-6);
    let sunlit = s.object_sunlight * Vec3::new((-atten.x * k).exp(), (-atten.y * k).exp(), (-atten.z * k).exp()) * gs;
    let tmp = (s.ambient + (Vec3::ONE - s.ambient) * s.cloud_shadow * 0.5).max(Vec3::ZERO);
    let amblit = if s.colored_ambient { tmp * 0.7 } else { tmp.powf(0.9) * 0.57 } * ga;
    normals.map(|(nl, al)| {
        if ll < 1.5 {
            1.1 * s2l(amblit * al * 0.9 + sunlit * (l2s(nl.powf(1.2)) * 1.35 * 0.7))
        } else {
            // the linear branch's ambient is grey
            Vec3::splat(lum(s2l(amblit * al))) + nl * s2l(sunlit)
        }
    })
}

/// What may switch in one frame, and why. None: nothing.
fn switches(path: &str) -> Option<&'static str> {
    const NAMED: [(&str, &str); 13] = [
        // patch 998: the texture layers. Each layer's place and size belong
        // to it and show at its share only (`blend`, checked); a slot and the
        // ids are taken over by the next layer when the fade ends, at share 1.
        (".fade.", "a texture layer's own values show at its share, which is checked"),
        // (the first cloud layer's own place and size, in the sky's fields:
        // they show at 1 - its share, and become the next layer's at share 1)
        ("frame.0.sky.cloud_scale#", "the first cloud layer's own size (patch 998)"),
        ("frame.0.sky.cloud_variance#", "the first cloud layer's own size (patch 998)"),
        ("frame.0.sky.cloud_pos_density1#0", "the first cloud layer's own place (patch 998)"),
        ("frame.0.sky.cloud_pos_density1#1", "the first cloud layer's own place (patch 998)"),
        ("frame.0.sky.cloud_pos_density2#0", "the first cloud layer's own place (patch 998)"),
        ("frame.0.sky.cloud_pos_density2#1", "the first cloud layer's own place (patch 998)"),
        ("frame.0.textures", "a texture id is handed to the next layer at share 1"),
        ("frame.0.next_textures", "a texture id is handed to the next layer at share 1"),
        // patch 123: the shaders' lighting branch cannot be mixed; it takes
        // the modern model at a modern share of 1 only. Its flags are words;
        // what they make is checked instead: the colours it reads are
        // blended, and the level of a lit surface (`lit_levels`, frame.2,
        // under the classic gains) keeps the pace, and FLIP_STOPS where the
        // model switches
        (
            "frame.0.sky.classic#",
            "the lighting model's flag; its lit level is checked (patch 123)",
        ),
        (
            "frame.0.sky.no_post#",
            "the lighting model's flag; its lit level is checked (patch 123)",
        ),
        ("frame.1#6", "the lighting mode number; its lit level is checked (patch 123)"),
        // the post pass's tone mapper id switches where the curve's mix is 0
        // (checked: the mix), both curves at mix 0 being the classic clamp
        (".tonemapper#", "the tone curve's id, switched at mix 0 (patch 123)"),
    ];
    NAMED.iter().find(|(p, _)| path.contains(p)).map(|(_, why)| *why)
}

/// The share of a texture layer, the one value of patch 998's layers that
/// must move continuously.
fn is_share(path: &str) -> bool {
    path.contains(".fade.") && path.contains(".blend#")
}

/// A share's own step: when a fade ends, its next layer, within one frame's
/// pace (`step`) of a share of 1, becomes the first and only one (share 0
/// of nothing).
fn layer_handed_over(path: &str, from: f64, to: f64, step: f64) -> bool {
    is_share(path) && from >= 1.0 - step && to == 0.0
}

fn input(now: f64) -> FrameInput {
    FrameInput {
        main_region: None,
        eep_region: None,
        legacy_sun: Vec3::Z,
        unix_secs: 1000.0,
        now,
        altitude: 25.0,
    }
}

const DT: f64 = 1.0 / 60.0;

/// The frames of a change from (sky, water) `a` to `b` over `duration`
/// seconds: the steady source, every frame of the change with its linear
/// progress, then two ticks of the steady target.
fn change(a: (SkyFrame, WaterFrame), b: (SkyFrame, WaterFrame), duration: f32) -> Vec<(f32, SkyFrame, WaterFrame)> {
    let ids = [
        Uuid::from_u128(0xe1),
        Uuid::from_u128(0xe2),
        Uuid::from_u128(0xe3),
        Uuid::from_u128(0xe4),
    ];
    let mut sel = EnvSelector::default();
    sel.request_local(ids[0], SettingsKind::Sky);
    sel.on_settings(ids[0], Some(Arc::new(Settings::Sky(a.0))));
    sel.request_local(ids[1], SettingsKind::Water);
    sel.on_settings(ids[1], Some(Arc::new(Settings::Water(a.1))));
    let (s, w) = sel.frame(&input(0.0));
    let mut out = vec![(0.0, s, w)];
    sel.set_manual_transition(duration);
    sel.request_local(ids[2], SettingsKind::Sky);
    sel.on_settings(ids[2], Some(Arc::new(Settings::Sky(b.0))));
    sel.request_local(ids[3], SettingsKind::Water);
    sel.on_settings(ids[3], Some(Arc::new(Settings::Water(b.1))));
    let start = 1.0;
    let mut now = start;
    let mut after = 0;
    while after < 3 {
        let p = (((now - start) as f32) / duration).min(1.0);
        let (s, w) = sel.frame(&input(now));
        out.push((p, s, w));
        if p >= 1.0 {
            after += 1;
        }
        now += DT;
    }
    out
}

fn num(l: &Leaf) -> Option<f64> {
    match l {
        Leaf::Num(v) => Some(*v),
        Leaf::Word(_) => None,
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-5 * a.abs().max(b.abs()).max(1.0) || (a.is_nan() && b.is_nan())
}

/// Checks (a), (b) and (c) for one pair; `extra` names what else may
/// switch in this pair, with why.
fn check(name: &str, a: (SkyFrame, WaterFrame), b: (SkyFrame, WaterFrame), duration: f32, extra: &[(&str, &str)]) {
    let frames = change(a, b, duration);
    let all: Vec<_> = frames.iter().map(|(p, s, w)| (*p, frame_leaves(s, w))).collect();
    let allowed = |path: &str| switches(path).is_some() || extra.iter().any(|(p, _)| path.contains(p));
    // (b) the change's first frame is the source as it showed
    let (source, first) = (&all[0].1, &all[1].1);
    assert_eq!(all[1].0, 0.0, "{name}: the change's first frame");
    for (path, v) in source {
        if allowed(path) && !is_share(path) {
            continue;
        }
        let w = first.get(path).unwrap_or_else(|| panic!("{name}: {path} gone at the start"));
        match (num(v), num(w)) {
            (Some(x), Some(y)) => assert!(close(x, y), "{name}: start, {path}: {x} -> {y}"),
            _ => assert_eq!(v, w, "{name}: start, {path}"),
        }
    }
    // (a) the first frame at full progress is the target as it shows on the
    // next tick
    let end = all.iter().position(|(p, _)| *p >= 1.0).expect("the change ends");
    let (last, next) = (&all[end].1, &all[end + 1].1);
    for (path, v) in next {
        let w = last.get(path).unwrap_or_else(|| panic!("{name}: {path} missing at the end"));
        match (num(v), num(w)) {
            (Some(x), Some(y)) => assert!(close(x, y), "{name}: end, {path}: {y} -> {x}"),
            _ => assert_eq!(v, w, "{name}: end, {path}"),
        }
    }
    // (c) no single-frame step: each value's per-frame move against its
    // excursion over the whole change
    let mut lo: BTreeMap<&String, f64> = BTreeMap::new();
    let mut hi: BTreeMap<&String, f64> = BTreeMap::new();
    for (_, l) in &all {
        for (path, v) in l {
            if let Some(x) = num(v).filter(|x| x.is_finite()) {
                let e = lo.entry(path).or_insert(x);
                *e = e.min(x);
                let e = hi.entry(path).or_insert(x);
                *e = e.max(x);
            }
        }
    }
    // the eased progress's fastest pace (patch 999) is pi / 2 the linear
    // one's, a linear change's is the linear one; the
    // values that are not linear in the progress (exponentials, the sRGB
    // curve, the geometric brightness scale) get a margin of 6
    let step = std::f64::consts::FRAC_PI_2 * DT / f64::from(duration) * 6.0;
    for pair in all.windows(2) {
        let ((p0, l0), (p1, l1)) = (&pair[0], &pair[1]);
        for (path, v1) in l1 {
            let Some(v0) = l0.get(path) else {
                assert!(allowed(path), "{name}: {path} appears at {p1}");
                continue;
            };
            if allowed(path) && !is_share(path) {
                continue;
            }
            // the lit level where the lighting model switches: bounded in
            // stops (patch 123's classic gains), at the pace elsewhere
            let mode = |l: &BTreeMap<String, Leaf>| l.get("frame.1#6").and_then(num);
            if path.starts_with("frame.2#") && mode(l0) != mode(l1) {
                let (x, y) = (num(v0).unwrap_or(0.0).max(1e-6), num(v1).unwrap_or(0.0).max(1e-6));
                let stops = (y / x).log2().abs();
                assert!(
                    stops <= f64::from(FLIP_STOPS),
                    "{name}: {path} where the lighting switches, {p0} -> {p1}: {x} -> {y} ({stops} stops)"
                );
                continue;
            }
            match (num(v0), num(v1)) {
                (Some(x), Some(y)) if layer_handed_over(path, x, y, step) => {}
                (Some(x), Some(y)) if x.is_finite() && y.is_finite() => {
                    let range = hi[path] - lo[path];
                    let bound = step * range + 1e-6 * x.abs().max(y.abs()).max(1.0);
                    assert!(
                        (x - y).abs() <= bound,
                        "{name}: a step in {path} between progress {p0} and {p1}: {x} -> {y} (whole excursion {range})"
                    );
                }
                _ => assert_eq!(v0, v1, "{name}: {path} switches between progress {p0} and {p1}"),
            }
        }
    }
}

fn id(n: u128) -> Uuid {
    Uuid::from_u128(n)
}

/// A sky with the sun 40 degrees up and the moon 60 degrees up, every value
/// set from `k` (0 dark, 1 bright).
fn sky(k: f32, modern: bool) -> SkyFrame {
    let l = |a: f32, b: f32| a + (b - a) * k;
    let v = |a: f32, b: f32| Vec3::splat(l(a, b));
    SkyFrame {
        sun_rotation: Quat::from_rotation_y(-40f32.to_radians()),
        moon_rotation: Quat::from_rotation_arc(Vec3::X, Vec3::new(-0.3, 0.4, 0.866).normalize()),
        sunlight: Vec3::new(l(0.05, 0.9), l(0.05, 0.85), l(0.1, 0.8)),
        ambient: Vec3::new(l(0.02, 0.4), l(0.02, 0.4), l(0.05, 0.45)),
        blue_horizon: v(0.05, 0.5),
        blue_density: Vec3::new(l(0.05, 0.25), l(0.08, 0.45), l(0.12, 0.75)),
        haze_density: l(0.2, 0.9),
        haze_horizon: l(0.05, 0.3),
        density_multiplier: l(0.00005, 0.0003),
        distance_multiplier: l(0.3, 1.2),
        max_y: l(900.0, 1900.0),
        glow: Vec3::new(l(2.0, 6.0), 0.001, l(-0.2, -0.6)),
        cloud_color: v(0.1, 0.6),
        cloud_pos_density1: Vec3::new(l(0.5, 1.5), l(0.3, 0.7), l(0.4, 1.2)),
        cloud_pos_density2: Vec3::new(l(0.6, 1.4), l(0.2, 0.8), l(0.3, 1.1)),
        cloud_scale: l(0.2, 0.6),
        cloud_scroll_rate: Vec2::new(l(0.1, 0.4), l(0.0, 0.1)),
        cloud_shadow: l(0.1, 0.5),
        cloud_variance: l(0.0, 0.3),
        sun_scale: l(0.6, 1.4),
        moon_scale: l(0.7, 1.3),
        moon_brightness: l(0.2, 0.9),
        star_brightness: l(50.0, 400.0),
        gamma: l(0.7, 1.6),
        moisture_level: l(0.1, 0.6),
        droplet_radius: l(300.0, 900.0),
        ice_level: l(0.1, 0.5),
        probe_ambiance: if modern { l(0.3, 1.0) } else { 0.0 },
        can_auto_adjust: !modern,
        ..Default::default()
    }
}

fn water(k: f32) -> WaterFrame {
    let l = |a: f32, b: f32| a + (b - a) * k;
    WaterFrame {
        fog_color: Vec3::new(l(0.02, 0.2), l(0.05, 0.4), l(0.08, 0.5)),
        fog_density: l(2.0, 8.0),
        underwater_fog_mod: l(0.1, 0.5),
        fresnel_scale: l(0.3, 0.8),
        fresnel_offset: l(0.2, 0.6),
        blur_multiplier: l(0.01, 0.08),
        normal_scale: Vec3::new(l(1.0, 3.0), l(1.0, 3.0), l(0.5, 1.5)),
        wave1: Vec2::new(l(0.5, 1.5), l(-0.5, 0.2)),
        wave2: Vec2::new(l(-1.0, 0.4), l(0.3, 1.2)),
        ..Default::default()
    }
}

/// Dark to bright and bright to dark, for each pair of kinds (modern and
/// classic), every value different; over the server's 5 s and the manual
/// 29 s.
#[test]
fn a_change_between_any_two_skies_has_no_step_and_meets_both_ends() {
    for duration in [5.0, 29.0] {
        for (ma, mb) in [(true, true), (false, false), (true, false), (false, true)] {
            for (ka, kb) in [(0.0, 1.0), (1.0, 0.0)] {
                let name = format!("{} {ka} -> {} {kb} in {duration} s", kind(ma), kind(mb));
                check(&name, (sky(ka, ma), water(ka)), (sky(kb, mb), water(kb)), duration, &[]);
            }
        }
    }
}

fn kind(modern: bool) -> &'static str {
    if modern { "modern" } else { "classic" }
}

/// A pair that differs in one discrete field at a time: each texture id
/// (patch 998 fades them), a texture to none (the stand-in), the rainbow
/// and the halo coming and going, the kind of sky, the linear kind.
#[test]
fn a_change_of_each_discrete_field_alone_has_no_step() {
    let base = (sky(0.5, true), water(0.5));
    let mut cases: Vec<(&str, (SkyFrame, WaterFrame))> = Vec::new();
    cases.push((
        "cloud id",
        (
            SkyFrame {
                cloud_id: id(0x11),
                ..base.0
            },
            base.1,
        ),
    ));
    cases.push((
        "sun id",
        (
            SkyFrame {
                sun_id: id(0x12),
                ..base.0
            },
            base.1,
        ),
    ));
    cases.push((
        "moon id",
        (
            SkyFrame {
                moon_id: id(0x13),
                ..base.0
            },
            base.1,
        ),
    ));
    cases.push((
        "no moon",
        (
            SkyFrame {
                moon_id: Uuid::nil(),
                ..base.0
            },
            base.1,
        ),
    ));
    cases.push((
        "rainbow id",
        (
            SkyFrame {
                rainbow_id: id(0x14),
                ..base.0
            },
            base.1,
        ),
    ));
    cases.push((
        "halo id",
        (
            SkyFrame {
                halo_id: id(0x15),
                ..base.0
            },
            base.1,
        ),
    ));
    cases.push((
        "no rainbow",
        (
            SkyFrame {
                moisture_level: 0.0,
                ..base.0
            },
            base.1,
        ),
    ));
    cases.push(("no halo", (SkyFrame { ice_level: 0.0, ..base.0 }, base.1)));
    cases.push((
        "normal map",
        (
            base.0,
            WaterFrame {
                normal_map: id(0x16),
                ..base.1
            },
        ),
    ));
    cases.push((
        "no normal map",
        (
            base.0,
            WaterFrame {
                normal_map: Uuid::nil(),
                ..base.1
            },
        ),
    ));
    cases.push(("classic", (sky(0.5, false), base.1)));
    cases.push((
        "linear",
        (
            SkyFrame {
                can_auto_adjust: false,
                ..sky(0.5, false)
            },
            base.1,
        ),
    ));
    for (what, other) in cases {
        for duration in [5.0, 29.0] {
            check(&format!("{what}, there, in {duration} s"), base, other, duration, &[]);
            check(&format!("{what}, back, in {duration} s"), other, base, duration, &[]);
        }
    }
}

/// A change begun over a running one starts from the picture shown, with
/// its own progress from 0: no value jumps at the interruption.
#[test]
fn a_change_begun_over_another_starts_from_the_picture_shown() {
    let ids = [Uuid::from_u128(0xf1), Uuid::from_u128(0xf2), Uuid::from_u128(0xf3)];
    let skies = [sky(0.0, true), sky(1.0, false), sky(0.5, true)];
    let mut sel = EnvSelector::default();
    sel.request_local(ids[0], SettingsKind::Sky);
    sel.on_settings(ids[0], Some(Arc::new(Settings::Sky(skies[0]))));
    sel.frame(&input(0.0));
    sel.set_manual_transition(5.0);
    sel.request_local(ids[1], SettingsKind::Sky);
    sel.on_settings(ids[1], Some(Arc::new(Settings::Sky(skies[1]))));
    let mut now = 1.0;
    let mut shown = sel.frame(&input(now));
    while now < 3.4 {
        now += DT;
        shown = sel.frame(&input(now));
    }
    let before = frame_leaves(&shown.0, &shown.1);
    sel.request_local(ids[2], SettingsKind::Sky);
    sel.on_settings(ids[2], Some(Arc::new(Settings::Sky(skies[2]))));
    let next = sel.frame(&input(now + DT));
    let after = frame_leaves(&next.0, &next.1);
    for (path, v) in &before {
        if switches(path).is_some() {
            continue;
        }
        let w = &after[path];
        match (num(v), num(w)) {
            (Some(x), Some(y)) => assert!(close(x, y), "{path}: {x} -> {y} at the interruption"),
            _ => assert_eq!(v, w, "{path}"),
        }
    }
}

// ------------------------------------------------------------ lit level

/// Two skies from a viewer's log: a classic night (the sun
/// straight down) and a modern day.
fn night_and_day() -> (SkyFrame, SkyFrame) {
    let night = SkyFrame {
        sun_rotation: Quat::from_rotation_y(90f32.to_radians()),
        sunlight: Vec3::new(0.349, 0.356, 0.660),
        ambient: Vec3::new(0.204, 0.242, 0.330),
        gamma: 1.0,
        probe_ambiance: 0.0,
        can_auto_adjust: true,
        ..Default::default()
    };
    let day = SkyFrame {
        sun_rotation: Quat::from_rotation_y(-61.5f32.to_radians()),
        sunlight: Vec3::new(1.741, 1.741, 1.929),
        gamma: 1.0,
        probe_ambiance: 1.0,
        can_auto_adjust: false,
        ..Default::default()
    };
    (night, day)
}

/// The pairs of skies of different kinds: dark to bright and bright to
/// dark for each order of the kinds, and the night and the day above.
fn kind_pairs() -> Vec<(String, SkyFrame, SkyFrame)> {
    let mut out = Vec::new();
    for (ma, mb) in [(true, false), (false, true)] {
        for (ka, kb) in [(0.0, 1.0), (1.0, 0.0)] {
            out.push((format!("{} {ka} -> {} {kb}", kind(ma), kind(mb)), sky(ka, ma), sky(kb, mb)));
        }
    }
    let (night, day) = night_and_day();
    out.push(("night -> day".into(), night, day));
    out.push(("day -> night".into(), day, night));
    out
}

/// Over a change through the selector: each frame's lighting mode, its lit
/// levels and its sky's brightness (the haze colour at the brightness
/// scale), before the exposure.
fn light_track(a: SkyFrame, b: SkyFrame, duration: f32) -> Vec<(f32, [f32; 3], f32, [Vec3; 3])> {
    change((a, WaterFrame::default()), (b, WaterFrame::default()), duration)
        .iter()
        .map(|(_, s, w)| {
            let env = Environment::from_eep(s, w, Vec2::ZERO, 96.0).with_colored_ambient(s, false);
            let ll = if env.sky.classic {
                1.0
            } else if env.sky.no_post {
                2.0
            } else {
                0.0
            };
            let lum = |c: Vec3| c.max(Vec3::ZERO).dot(Vec3::new(0.2126, 0.7152, 0.0722));
            (
                ll,
                lit_levels(&env, ll),
                lum(env.fog_color) * env.sky.hdr_scale,
                lit_colours(&env, ll),
            )
        })
        .collect()
}

/// Where the lighting model switches (one frame, at a modern share of 1),
/// a lit white surface keeps its level, before the exposure: the classic
/// gains match the two models' levels edge-on to the light, and hold the
/// faces from the one facing up to the one facing the light within
/// FLIP_STOPS (the check of every frame, `check`, holds the lit levels to
/// the same bound at the switch and to the pace elsewhere). Reached: 0.028
/// stop for the night and the day, 0.12 to 0.16 stop for the other pairs
/// (their sun 40 degrees up). The colour may differ, the classic lighting
/// summing its sun and ambient in sRGB where the HDR one sums them
/// linearly: the chromaticity (r, g, b over their sum) moves by 0.006 for
/// the night and the day, up to 0.09 for the other pairs.
const FLIP_STOPS: f32 = 0.2;

#[test]
fn the_lit_surface_keeps_its_level_where_the_lighting_switches() {
    for duration in [5.0, 29.0] {
        for (name, a, b) in kind_pairs() {
            let track = light_track(a, b, duration);
            let mut flips = 0;
            for w in track.windows(2) {
                if w[0].0 == w[1].0 {
                    continue;
                }
                flips += 1;
                for k in 0..3 {
                    let (x, y) = (w[0].1[k].max(1e-6), w[1].1[k].max(1e-6));
                    let stops = (y / x).log2().abs();
                    let limit = if k == 0 { 0.02 } else { FLIP_STOPS };
                    assert!(stops <= limit, "{name} in {duration} s, normal {k}: {x} -> {y} ({stops} stops)");
                }
            }
            assert!(flips <= 1, "{name}: {flips} switches");
        }
    }
}

/// No flash, before the exposure: over a change between the kinds, the lit
/// surface facing up and the sky never pass the brighter end by more than
/// LIGHT_MARGIN of the distance between the ends, nor fall below the darker
/// one by more (the rule: never brighter than the brighter end, never darker
/// than the darker end). Reached: the sky 0; the surface 0.021 for the night
/// and the day, 0.127 (0.16 stop, the switch's residue above) for a bright
/// modern sky to a dark classic one and back; nothing below the darker end.
/// What the automatic exposure adds on top is not in this test (it eases on
/// the GPU; see the patch's header).
const LIGHT_MARGIN: f32 = 0.15;

fn envelope(values: &[f32]) -> (f32, f32) {
    let (a, b) = (values[0], *values.last().expect("values"));
    let (lo, hi) = (a.min(b), a.max(b));
    let d = (hi - lo).max(1e-6);
    let over = values.iter().map(|v| (v - hi) / d).fold(0.0, f32::max);
    let under = values.iter().map(|v| (lo - v) / d).fold(0.0, f32::max);
    (over, under)
}

#[test]
fn a_change_between_the_kinds_is_never_brighter_or_darker_than_its_ends() {
    for duration in [5.0, 29.0] {
        for (name, a, b) in kind_pairs() {
            let track = light_track(a, b, duration);
            let surface: Vec<f32> = track.iter().map(|t| t.1[2]).collect();
            let sky: Vec<f32> = track.iter().map(|t| t.2).collect();
            for (what, v) in [("surface", &surface), ("sky", &sky)] {
                let (over, under) = envelope(v);
                assert!(
                    over <= LIGHT_MARGIN && under <= LIGHT_MARGIN,
                    "{name} in {duration} s, {what}: over {over:.3}, under {under:.3}"
                );
            }
        }
    }
}

/// The numbers of the two tests above (run with --ignored --nocapture).
#[test]
#[ignore]
fn lit_level_report() {
    for duration in [5.0, 29.0] {
        for (name, a, b) in kind_pairs() {
            let track = light_track(a, b, duration);
            let mut flip = [0.0f32; 3];
            let mut hue = 0.0f32;
            let chroma = |c: Vec3| c.max(Vec3::ZERO) / c.max(Vec3::ZERO).element_sum().max(1e-9);
            for w in track.windows(2) {
                if w[0].0 != w[1].0 {
                    for k in 0..3 {
                        flip[k] = flip[k].max((w[1].1[k].max(1e-6) / w[0].1[k].max(1e-6)).log2().abs());
                        hue = hue.max((chroma(w[1].3[k]) - chroma(w[0].3[k])).abs().max_element());
                    }
                }
            }
            let surface: Vec<f32> = track.iter().map(|t| t.1[2]).collect();
            let sky: Vec<f32> = track.iter().map(|t| t.2).collect();
            let (so, su) = envelope(&surface);
            let (ko, ku) = envelope(&sky);
            eprintln!(
                "LIT {name:>24} {duration:>4} s: flip stops edge {:.3} facing {:.3} up {:.3} chroma {hue:.4} | surface over {so:.3} under {su:.3} | sky over {ko:.3} under {ku:.3}",
                flip[0], flip[1], flip[2]
            );
        }
    }
}
