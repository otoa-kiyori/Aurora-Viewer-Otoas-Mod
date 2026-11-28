//! How a sky is presented while it blends between the two kinds of sky
//! (local patch set, scripts/patches/123-sky-mode-blend.toml).
//!
//! A modern sky (one with a reflection probe ambiance) is tone mapped,
//! exposed automatically and drawn at its HDR brightness scale; a classic
//! one (probe ambiance 0) gets none of these (LLPipeline::tonemap's no_post,
//! LLPipeline::generateExposure, LLSettingsVOSky::applySpecial's
//! sky_hdr_scale; indra/newview/pipeline.cpp, llvosky.cpp). While an
//! environment change, or a day cycle between two keyframes, blends a sky of
//! one kind into one of the other, [`SkyParams::modern`] is the modern
//! side's share, strictly between 0 and 1, and each of these follows it:
//! - the brightness scale goes from 1 to the modern sky's, geometrically
//!   ([`hdr_scale`]);
//! - the tone curve chosen in Preferences is applied at RenderTonemapMix x
//!   the share: at mix 0 both curves give the classic clamp ([`post`]);
//! - the legacy gamma goes from the sky's own gamma to 1, linearly;
//! - the automatic exposure's range [1 / scale, scale] follows the scale,
//!   and its dark scene term weighs the share ([`exposure_dark`]);
//! - the classic lighting, which surfaces keep until the share reaches 1,
//!   takes gains on its sun and its ambient light that bring its level to
//!   the HDR lighting's as the share rises ([`classic_gains`]), so that a
//!   lit surface keeps its brightness where the lighting model switches.
//!
//! A sky of one kind (share exactly 0 or 1) takes the values it had before,
//! to the bit. The Legacy Viewer switches at once to the next sky's kind
//! (LLSettingsSky::blend, indra/llinventory/llsettingssky.cpp, which takes
//! the next sky's canAutoAdjust): it steps, as Aurora did at one end of the
//! change.
// aurora-local-patch: sky-mode-blend-render-module

use crate::types::{SkyParams, Tonemapper};
use glam::Vec3;

/// The modern share of a sky between the two kinds, strictly between 0
/// and 1; None for a sky of one kind.
pub fn blending(sky: &SkyParams) -> Option<f32> {
    (sky.modern > 0.0 && sky.modern < 1.0).then_some(sky.modern)
}

/// The brightness scale at a modern share: `modern` (the modern sky's,
/// `sqrt(gamma) x 2`) at 1, 1 at 0, and between them `modern ^ share`. The
/// displayed brightness is the scale times the automatic exposure, whose
/// range is [1 / scale, scale]: a geometric blend moves their product, and
/// the range, at an even pace in log units, the way brightness is seen. A
/// linear blend of the scale would front-load its change and, against the
/// tone curve fading out (which brightens the dark values), leave a bump
/// of about 5 % in the dark values near the classic end; the geometric one
/// stays within half a percent there. A scale of 0 (gamma 0) blends
/// linearly, as no power reaches it.
pub fn hdr_scale(modern: f32, share: f32) -> f32 {
    if share >= 1.0 {
        modern
    } else if share <= 0.0 {
        1.0
    } else if modern > 0.0 {
        modern.powf(share)
    } else {
        1.0 - share
    }
}

/// The post pass's tone mapping inputs for a sky (post.wgsl PostParams).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Post {
    /// 0 Khronos PBR Neutral, 1 ACES, 2 none.
    pub tonemapper: f32,
    pub legacy_gamma: f32,
    pub tonemap_mix: f32,
}

/// The tone mapper, the legacy gamma and the tone map mix of a frame, "HDR
/// and Emissive" being `hdr`, Preferences' tone mapper `tonemapper` and
/// RenderTonemapMix `tonemap_mix`.
pub fn post(sky: &SkyParams, hdr: bool, tonemapper: Tonemapper, tonemap_mix: f32) -> Post {
    let curve = match tonemapper {
        Tonemapper::KhronosNeutral => 0.0,
        Tonemapper::Aces => 1.0,
    };
    if let Some(m) = blending(sky) {
        // the classic side's legacy gamma (its sky's gamma) to the modern
        // side's (1); the curve fades out at the share (both curves at mix 0
        // are the classic clamp, `display`)
        let gamma = if sky.gamma.is_finite() { sky.gamma.max(0.0) } else { 1.0 };
        return Post {
            tonemapper: if hdr { curve } else { 2.0 },
            legacy_gamma: gamma + (1.0 - gamma) * m,
            tonemap_mix: if hdr { tonemap_mix.clamp(0.0, 1.0) * m } else { 0.0 },
        };
    }
    Post {
        // classic skies are not tone mapped (no_post + legacy gamma)
        tonemapper: if sky.no_post || !hdr { 2.0 } else { curve },
        // legacyGamma (postDeferredTonemap / CASF LEGACY_GAMMA): same
        // condition as no_post in LLPipeline::tonemap / applyCAS
        legacy_gamma: if sky.no_post && sky.gamma.is_finite() {
            sky.gamma.max(0.0)
        } else {
            1.0
        },
        // RenderTonemapMix on skies that are not classic; a classic sky
        // tone mapped all the same (probe ambiance above 0) is not mixed
        // (LLSettingsSky::getTonemapMix, LLSettingsVOSky::applySpecial)
        tonemap_mix: if sky.classic || !hdr { 0.0 } else { tonemap_mix.clamp(0.0, 1.0) },
    }
}

/// Whether the sky is tone mapped (before "HDR and Emissive"): a modern
/// sky, or one between the two kinds. The automatic exposure has a range
/// then, and Preferences' tone mapper choice is not greyed out.
pub fn tone_mapped(sky: &SkyParams) -> bool {
    !sky.no_post || blending(sky).is_some()
}

/// The weight of the automatic exposure's dark scene term
/// (shaders/exposure.wgsl): 1 on a sky of one kind, the modern share
/// between the two, so that it fades in and out with the range instead of
/// switching where the range leaves 1.
pub fn exposure_dark(sky: &SkyParams) -> f32 {
    blending(sky).unwrap_or(1.0)
}

/// The automatic exposure's range under this sky.
pub fn exposure_range(sky: &SkyParams, hdr: bool) -> (f32, f32) {
    crate::exposure::range(tone_mapped(sky) && hdr, sky.hdr_scale)
}

/// The weight of the HDR lighting's moon factor in the sky rays (patch
/// 114, `frame.sky_optics.w`): with the moon as the light, 1 under the HDR
/// lighting (`ll_mode` 0), the modern share while the sky blends between
/// the kinds (the classic lighting's moon colour already holds its
/// brightness, the HDR one's does not), 0 otherwise.
pub fn rays_moon(sky: &SkyParams, ll_mode: f32) -> f32 {
    let moon = sky.sun_dir.z < 0.0;
    if !moon {
        0.0
    } else if ll_mode == 0.0 {
        1.0
    } else {
        blending(sky).unwrap_or(0.0)
    }
}

/// The classic lighting's gains on its sun and its ambient light
/// (common.wgsl `ll_atmos_core`, `frame.sky_optics_tex.zw`): exactly 1 on a
/// sky of one kind; between the kinds, `(hdr / classic) ^ share` of each,
/// where `hdr / classic` is the gain that makes the classic lighting of the
/// frame's light (the branch the surfaces take below a share of 1) as
/// bright as the HDR lighting's (taken at a share of 1) on a white surface:
/// the ambient gain for a surface edge-on to the light (the ambient alone),
/// the sun gain for the faces between the one facing up and the one facing
/// the light (the two together), the one whose largest difference over them
/// is the least. The HDR side is a
/// face without a PBR material (shade_pbr's `legacy`: the sun colour / 3
/// times n.l, and the hemisphere, which the probes' irradiance follows);
/// the classic side softenLightF's legacy branch, classic or linear, as
/// common.wgsl `ll_classic_sun_ambient` writes it. Levels are matched,
/// colours are not.
pub fn classic_gains(sky: &SkyParams) -> [f32; 2] {
    let Some(share) = blending(sky) else {
        return [1.0, 1.0];
    };
    let (sun, ambient) = classic_match(sky);
    [(sun.ln() * share).exp(), (ambient.ln() * share).exp()]
}

/// The gains at a share of 1 ([`classic_gains`]), each within e^-3..e^3.
fn classic_match(s: &SkyParams) -> (f32, f32) {
    let lum = |c: Vec3| c.max(Vec3::ZERO).dot(Vec3::new(0.2126, 0.7152, 0.0722));
    let lin = |c: Vec3| srgb_to_linear(c.max(Vec3::ZERO));
    // ll_atmos_core: the objects' light along the light, and the ambient
    let atten = (s.blue_density + Vec3::splat(s.haze_density * 0.25)) * (s.density_multiplier * s.max_y);
    let along = 1.0 / s.light_norm.z.max(1e-6);
    let e = |x: f32| (-x * along).exp();
    let sunlit = s.object_sunlight * Vec3::new(e(atten.x), e(atten.y), e(atten.z));
    let tmp = (s.ambient + (Vec3::ONE - s.ambient) * s.cloud_shadow * 0.5).max(Vec3::ZERO);
    let amblit = if s.colored_ambient { tmp * 0.7 } else { tmp.powf(0.9) * 0.57 };
    // the HDR lighting of the same light (world/env.rs: the sun colour is the
    // objects' light made linear, x 3; the hemisphere amblit made linear)
    let hdr_ambient = lum(lin(amblit));
    let hdr_sun = lum(lin(sunlit));
    // the classic lighting with the gains for a face at `nl` to the light
    // (ambientLighting 1 - (nl / 2)^2): edge-on (n.l 0, the ambient alone)
    // up to facing it (n.l 1)
    let classic_mode = s.classic;
    let classic = |nl: f32, ga: f32, gs: f32| {
        let al = 1.0 - (0.5 * nl).powi(2);
        if classic_mode {
            let sun = linear_to_srgb(Vec3::splat(nl.powf(1.2))).x * 1.35 * 0.7;
            1.1 * lum(lin(amblit * (0.9 * al * ga) + sunlit * (sun * gs)))
        } else {
            lum(lin(amblit * (al * ga))) + nl * lum(lin(sunlit * gs))
        }
    };
    let ambient = solve(|g| classic(0.0, g, 1.0), hdr_ambient);
    // the sun gain that keeps the faces from the one facing up (the ground)
    // to the one facing the light nearest the HDR level: the largest ratio
    // in log units over their n.l, the least it can be (each ratio grows
    // with the gain, so the largest is lowest where it turns: a ternary
    // search). The faces nearer edge-on, where the classic sun's sRGB curve
    // rises steeply, are left further off (up to about 0.3 stop).
    let up = s.sun_dir.z.clamp(0.0, 1.0);
    let sun = if hdr_sun > 1e-6 && lum(sunlit) > 1e-6 {
        let worst = |lg: f32| {
            (0..=16)
                .map(|k| {
                    let nl = (up + (1.0 - up) * k as f32 / 16.0).max(1.0 / 64.0);
                    (classic(nl, ambient, lg.exp()).max(1e-9) / (hdr_sun * nl + hdr_ambient)).ln().abs()
                })
                .fold(0.0, f32::max)
        };
        let (mut lo, mut hi) = (-3.0f32, 3.0f32);
        for _ in 0..60 {
            let (a, b) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
            if worst(a) <= worst(b) {
                hi = b;
            } else {
                lo = a;
            }
        }
        (0.5 * (lo + hi)).exp()
    } else {
        ambient
    };
    (sun, ambient)
}

/// The gain in e^-3..e^3 at which the increasing `f` meets `target`
/// (bisection in log units; 1 when there is nothing to match).
fn solve(f: impl Fn(f32) -> f32, target: f32) -> f32 {
    if target <= 1e-9 || f(1.0) <= 1e-9 {
        return 1.0;
    }
    let (mut lo, mut hi) = (-3.0f32, 3.0f32);
    if f(lo.exp()) >= target {
        return lo.exp();
    }
    if f(hi.exp()) <= target {
        return hi.exp();
    }
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if f(mid.exp()) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (0.5 * (lo + hi)).exp()
}

/// A copy of post.wgsl's `tonemap` and `legacy_gamma` for one colour: the
/// colour displayed (linear, before the output encoding) for a scene colour
/// `c` at the exposure `exposure`.
pub fn display(post: &Post, exposure: f32, c: Vec3) -> Vec3 {
    let e = (c * exposure).max(Vec3::ZERO);
    let m = if post.tonemapper > 1.5 {
        e
    } else {
        let t = if post.tonemapper > 0.5 { aces(e) } else { neutral(e) };
        e.lerp(t, post.tonemap_mix)
    };
    let m = m.clamp(Vec3::ZERO, Vec3::ONE);
    if (post.legacy_gamma - 1.0).abs() < 1e-4 {
        return m;
    }
    let c = Vec3::ONE - linear_to_srgb(m).clamp(Vec3::ZERO, Vec3::ONE);
    srgb_to_linear(Vec3::ONE - c.powf(post.legacy_gamma))
}

/// post.wgsl `neutral` (Khronos PBR Neutral).
fn neutral(color_in: Vec3) -> Vec3 {
    let start = 0.8 - 0.04;
    let desat = 0.15;
    let x = color_in.min_element();
    let offset = if x < 0.08 { x - 6.25 * x * x } else { 0.04 };
    let color = color_in - Vec3::splat(offset);
    let peak = color.max_element();
    if peak < start {
        return color;
    }
    let d = 1.0 - start;
    let new_peak = 1.0 - d * d / (peak + d - start);
    let color = color * (new_peak / peak);
    let g = 1.0 - 1.0 / (desat * (peak - new_peak) + 1.0);
    color.lerp(Vec3::splat(new_peak), g)
}

/// post.wgsl `aces` (Stephen Hill's fit).
fn aces(x: Vec3) -> Vec3 {
    let input = glam::Mat3::from_cols(
        Vec3::new(0.59719, 0.07600, 0.02840),
        Vec3::new(0.35458, 0.90834, 0.13383),
        Vec3::new(0.04823, 0.01566, 0.83777),
    );
    let output = glam::Mat3::from_cols(
        Vec3::new(1.60475, -0.10208, -0.00327),
        Vec3::new(-0.53108, 1.10813, -0.07276),
        Vec3::new(-0.07367, -0.00605, 1.07602),
    );
    let c = input * x;
    let a = c * (c + Vec3::splat(0.0245786)) - Vec3::splat(0.000090537);
    let b = c * (c * 0.983729 + Vec3::splat(0.432_951)) + Vec3::splat(0.238081);
    (output * (a / b)).clamp(Vec3::ZERO, Vec3::ONE)
}

fn linear_to_srgb(c: Vec3) -> Vec3 {
    let f = |x: f32| {
        if x <= 0.0031308 {
            x * 12.92
        } else {
            1.055 * x.powf(1.0 / 2.4) - 0.055
        }
    };
    Vec3::new(f(c.x), f(c.y), f(c.z))
}

fn srgb_to_linear(c: Vec3) -> Vec3 {
    let f = |x: f32| if x <= 0.04045 { x / 12.92 } else { ((x + 0.055) / 1.055).powf(2.4) };
    Vec3::new(f(c.x), f(c.y), f(c.z))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sky(modern: f32, classic: bool, no_post: bool, gamma: f32, hdr_scale: f32) -> SkyParams {
        SkyParams {
            modern,
            classic,
            no_post,
            gamma,
            hdr_scale,
            ..Default::default()
        }
    }

    /// The values the renderer wrote before the blend, for a sky of one kind.
    fn upstream(s: &SkyParams, hdr: bool, curve: f32, mix: f32) -> Post {
        Post {
            tonemapper: if s.no_post || !hdr { 2.0 } else { curve },
            legacy_gamma: if s.no_post && s.gamma.is_finite() { s.gamma.max(0.0) } else { 1.0 },
            tonemap_mix: if s.classic || !hdr { 0.0 } else { mix.clamp(0.0, 1.0) },
        }
    }

    fn colours() -> Vec<Vec3> {
        let mut v = Vec::new();
        for i in 0..=60 {
            let x = 0.0005 * 1.18f32.powi(i);
            v.push(Vec3::splat(x));
            v.push(Vec3::new(x, x * 0.6, x * 0.25));
            v.push(Vec3::new(x * 0.2, x * 0.5, x));
        }
        v
    }

    #[test]
    fn every_tone_curve_at_mix_zero_is_the_classic_clamp() {
        let classic = Post {
            tonemapper: 2.0,
            legacy_gamma: 1.0,
            tonemap_mix: 0.0,
        };
        for curve in [0.0, 1.0] {
            let faded = Post {
                tonemapper: curve,
                ..classic
            };
            for c in colours() {
                for exposure in [0.5, 1.0, 2.0] {
                    assert_eq!(display(&faded, exposure, c), display(&classic, exposure, c), "{curve} {c}");
                    // and it moves no more than its mix away from it
                    let near = Post {
                        tonemap_mix: 0.01,
                        ..faded
                    };
                    let d = (display(&near, exposure, c) - display(&classic, exposure, c)).abs().max_element();
                    assert!(d <= 0.01 + 1e-6, "{curve} {c} {d}");
                }
            }
        }
    }

    #[test]
    fn the_brightness_scale_blends_in_log_units_and_keeps_its_ends() {
        assert_eq!(hdr_scale(2.0, 1.0), 2.0);
        assert_eq!(hdr_scale(2.0, 0.0), 1.0);
        assert!((hdr_scale(4.0, 0.5) - 2.0).abs() < 1e-6);
        let mut last = 1.0;
        for i in 1..=1000 {
            let m = i as f32 / 1000.0;
            let h = hdr_scale(2.0, m);
            assert!(h >= last && h - last <= 2.0f32.ln() * 2.0 / 1000.0 + 1e-6, "{m}");
            last = h;
        }
        assert!((last - 2.0).abs() < 1e-6);
        // no power reaches a scale of 0: linear there
        assert!((hdr_scale(0.0, 0.25) - 0.75).abs() < 1e-6);
        assert!((hdr_scale(0.0, 1e-6) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn a_sky_of_one_kind_keeps_its_post_values_to_the_bit() {
        let skies = [
            // modern, classic, linear (a probe ambiance key at 0), and a
            // classic sky tone mapped all the same
            sky(1.0, false, false, 1.3, 2.28),
            sky(0.0, true, true, 1.3, 1.0),
            sky(0.0, false, true, 0.8, 1.0),
            sky(1.0, true, false, 1.0, 2.0),
        ];
        for s in skies {
            for hdr in [true, false] {
                for (mapper, curve) in [(Tonemapper::KhronosNeutral, 0.0), (Tonemapper::Aces, 1.0)] {
                    assert_eq!(post(&s, hdr, mapper, 0.7), upstream(&s, hdr, curve, 0.7), "{hdr} {curve}");
                }
                assert_eq!(exposure_range(&s, hdr), crate::exposure::range(!s.no_post && hdr, s.hdr_scale));
                assert_eq!(exposure_dark(&s), 1.0);
            }
        }
    }

    #[test]
    fn between_the_kinds_the_post_values_meet_each_side() {
        let gamma: f32 = 1.4;
        let modern_scale = gamma.sqrt() * 2.0;
        let classic = sky(0.0, true, true, gamma, 1.0);
        let modern = sky(1.0, false, false, gamma, modern_scale);
        for (mapper, curve) in [(Tonemapper::KhronosNeutral, 0.0), (Tonemapper::Aces, 1.0)] {
            let near = |m: f32| {
                // (the lighting side's flags, as world/sky_kind.rs sets them)
                let s = sky(m, m < 1.0, m < 1.0, gamma, hdr_scale(modern_scale, m));
                (post(&s, true, mapper, 0.7), exposure_range(&s, true), exposure_dark(&s))
            };
            let ends = [
                (1e-4, post(&classic, true, mapper, 0.7), exposure_range(&classic, true), 0.0),
                (1.0 - 1e-4, post(&modern, true, mapper, 0.7), exposure_range(&modern, true), 1.0),
            ];
            for (m, end, range, dark) in ends {
                let (p, r, d) = near(m);
                assert!((p.legacy_gamma - end.legacy_gamma).abs() < 1e-3);
                assert!((p.tonemap_mix - end.tonemap_mix).abs() < 1e-3);
                assert!((r.0 - range.0).abs() < 1e-3 && (r.1 - range.1).abs() < 1e-3, "{r:?} {range:?}");
                assert!((d - dark).abs() < 1e-3);
                for c in colours() {
                    for exposure in [r.0, 1.0, r.1] {
                        let a = display(&p, exposure, c);
                        let b = display(&end, exposure, c);
                        assert!((a - b).abs().max_element() < 2e-3, "{curve} {m} {c}: {a} {b}");
                    }
                }
            }
        }
    }

    #[test]
    fn the_exposure_target_fades_its_dark_term_with_the_range() {
        use crate::exposure::{range, target, target_with};
        for average in [0.0005, 0.002, 0.008, 0.02, 0.2, 1.0] {
            // one kind: as before
            assert_eq!(target_with(average, range(true, 2.0), 1.0), target(average, range(true, 2.0)));
            assert_eq!(target_with(average, range(false, 2.0), 1.0), 1.0);
            // between: from 1 at the classic end to the modern target
            let mut last = target_with(average, range(true, hdr_scale(2.0, 0.0)), 0.0);
            assert_eq!(last, 1.0);
            for i in 1..=1000 {
                let m = i as f32 / 1000.0;
                let t = target_with(average, range(true, hdr_scale(2.0, m)), m);
                assert!((t - last).abs() < 0.01, "{average} {m}: {last} -> {t}");
                last = t;
            }
            assert!((last - target(average, range(true, 2.0))).abs() < 1e-5);
        }
    }

    /// A blending sky's light, for the gains: `classic` its lighting side.
    fn lit(modern: f32, classic: bool, sun_z: f32, sunlight: Vec3, ambient: Vec3) -> SkyParams {
        let light = Vec3::new((1.0 - sun_z * sun_z).max(0.0).sqrt(), 0.0, sun_z);
        SkyParams {
            modern,
            classic,
            no_post: true,
            object_sunlight: sunlight,
            light_norm: light,
            sun_dir: light,
            ambient,
            ..Default::default()
        }
    }

    /// The classic lighting of a white surface (common.wgsl
    /// ll_classic_sun_ambient, ll_final_scale) at n.l `nl` and
    /// ambientLighting `al`, with the gains, written out from the shader.
    fn classic_level(s: &SkyParams, nl: f32, al: f32) -> f32 {
        let [gs, ga] = classic_gains(s);
        let lum = |c: Vec3| c.max(Vec3::ZERO).dot(Vec3::new(0.2126, 0.7152, 0.0722));
        let atten = (s.blue_density + Vec3::splat(s.haze_density * 0.25)) * (s.density_multiplier * s.max_y);
        let k = 1.0 / s.light_norm.z.max(1e-6);
        let sunlit = s.object_sunlight * Vec3::new((-atten.x * k).exp(), (-atten.y * k).exp(), (-atten.z * k).exp()) * gs;
        let tmp = s.ambient + (Vec3::ONE - s.ambient) * s.cloud_shadow * 0.5;
        let amblit = tmp.max(Vec3::ZERO).powf(0.9) * 0.57 * ga;
        if s.classic {
            let sun = linear_to_srgb(Vec3::splat(nl.powf(1.2)));
            1.1 * lum(srgb_to_linear(amblit * al * 0.9 + sun * sunlit * 1.35 * 0.7))
        } else {
            lum(srgb_to_linear(amblit * al)) + nl * lum(srgb_to_linear(sunlit))
        }
    }

    /// The HDR lighting of the same white surface without a PBR material:
    /// the sun colour / 3 x n.l, and the hemisphere.
    fn hdr_level(s: &SkyParams, nl: f32) -> f32 {
        let lum = |c: Vec3| c.max(Vec3::ZERO).dot(Vec3::new(0.2126, 0.7152, 0.0722));
        let atten = (s.blue_density + Vec3::splat(s.haze_density * 0.25)) * (s.density_multiplier * s.max_y);
        let k = 1.0 / s.light_norm.z.max(1e-6);
        let sunlit = s.object_sunlight * Vec3::new((-atten.x * k).exp(), (-atten.y * k).exp(), (-atten.z * k).exp());
        let tmp = s.ambient + (Vec3::ONE - s.ambient) * s.cloud_shadow * 0.5;
        let amblit = tmp.max(Vec3::ZERO).powf(0.9) * 0.57;
        nl * lum(srgb_to_linear(sunlit)) + lum(srgb_to_linear(amblit))
    }

    #[test]
    fn a_sky_of_one_kind_takes_no_classic_gain() {
        for modern in [0.0, 1.0] {
            for classic in [false, true] {
                let s = lit(modern, classic, 0.5, Vec3::new(0.9, 0.9, 1.0), Vec3::splat(0.3));
                assert_eq!(classic_gains(&s), [1.0, 1.0]);
            }
        }
        assert_eq!(classic_gains(&SkyParams::default()), [1.0, 1.0]);
    }

    /// Just below a share of 1, where the surfaces switch to the HDR
    /// lighting, the classic lighting with the gains is as bright as the HDR
    /// one for a surface edge-on to the light (within 0.5 %), within
    /// FIT_STOPS of it for every face between the one facing up and the one
    /// facing the light, and within GRAZING_STOPS for the faces nearer
    /// edge-on (the classic sun's pow 1.2 and sRGB sum against the HDR
    /// lighting's linear n.l: two gains cannot meet a curve); between, the
    /// gains move with the share, from 1 at 0.
    const FIT_STOPS: f32 = 0.32;
    const GRAZING_STOPS: f32 = 0.55;

    #[test]
    fn the_classic_gains_meet_the_hdr_level_at_the_switch() {
        for classic in [true, false] {
            for (sun_z, sunlight, ambient) in [
                (0.88, Vec3::new(0.9, 0.9, 1.0), Vec3::splat(0.25)),
                (0.5, Vec3::new(1.0, 0.8, 0.6), Vec3::new(0.2, 0.24, 0.33)),
                (0.2, Vec3::new(0.53, 0.54, 1.0), Vec3::splat(0.05)),
                (0.95, Vec3::new(1.0, 1.0, 1.0), Vec3::splat(0.6)),
            ] {
                let s = lit(1.0 - 1e-6, classic, sun_z, sunlight, ambient);
                let (c, h) = (classic_level(&s, 0.0, 1.0), hdr_level(&s, 0.0));
                assert!((c / h - 1.0).abs() < 5e-3, "{classic} {sun_z} edge-on: {c} vs {h}");
                for k in 0..=32 {
                    let nl = k as f32 / 32.0;
                    let (c, h) = (classic_level(&s, nl, 1.0 - (0.5 * nl).powi(2)), hdr_level(&s, nl));
                    let stops = (c / h).log2().abs();
                    let limit = if nl >= sun_z { FIT_STOPS } else { GRAZING_STOPS };
                    assert!(stops <= limit, "{classic} {sun_z} {nl}: {c} vs {h} ({stops} stops)");
                }
                let mut last = classic_gains(&lit(1e-6, classic, sun_z, sunlight, ambient));
                assert!(last.iter().all(|g| (g - 1.0).abs() < 1e-4));
                for i in 1..=100 {
                    let g = classic_gains(&lit(i as f32 / 101.0, classic, sun_z, sunlight, ambient));
                    for k in 0..2 {
                        assert!((g[k].ln() - last[k].ln()).abs() < 0.1, "{classic} {sun_z} {i}");
                    }
                    last = g;
                }
            }
        }
    }
}
