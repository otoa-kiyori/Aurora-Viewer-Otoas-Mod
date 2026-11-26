//! The kind of a sky while it blends between the two kinds (local patch
//! set, scripts/patches/123-sky-mode-blend.toml).
//!
//! A sky is modern when it has a reflection probe ambiance (tone mapped,
//! exposed automatically, drawn at its HDR brightness scale, lit by the HDR
//! model), classic when it has no probe ambiance key (LLSettingsSky::
//! canAutoAdjust: none of these, the Legacy lighting), linear when its key
//! is there at 0 (the linear lighting, no tone mapping). The blended frame
//! of two skies of different kinds (SkyFrame::lerp: an environment change,
//! the selector's fade, or a day cycle between two keyframes) carries the
//! share of each kind, lerped from each side's own (1 or 0, or the share a
//! side being itself a blend has): a sky that is not blending has exactly 1
//! or 0. The shares hold while the frame keeps the probe ambiance and the
//! classic flag it was blended to (an editor changing them makes it a sky
//! of its own kind again).
//!
//! What follows the modern share, continuously (aurora-render
//! `sky_mode`): the brightness scale, the tone curve's mix, the legacy
//! gamma, the automatic exposure's range and its dark scene term. What
//! cannot be mixed, the lighting model (the shaders' branch between the HDR
//! lighting and the Legacy or linear one, the frame's light colours made
//! for it, the probes' irradiance capture, which follow `classic` and
//! `no_post`), takes the classic side's as soon as the modern share is
//! below 1, and the modern one only at 1: its step then falls where the
//! automatic exposure has its whole range to take it up (at the start of a
//! change from a modern sky, at the end of one to it), never where the
//! presentation is already the classic one's, whose exposure is fixed at 1.
//! Between two skies of the same kind nothing changes.
// aurora-local-patch: sky-mode-blend-module

use super::eep::SkyFrame;
use super::env::tone_mapped;
use glam::Vec3;

/// The shares of the kinds in a blended frame (`SkyFrame::kind`): modern,
/// linear, the rest classic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KindBlend {
    pub modern: f32,
    pub linear: f32,
    /// The frame's probe ambiance (its bits) and classic flag when it was
    /// blended: the shares hold while it keeps them.
    stamp: (u32, bool),
}

/// What a frame is lit and presented as.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkyKind {
    /// The modern share (`SkyParams::modern`).
    pub modern: f32,
    /// The HDR lighting model (else the Legacy or the linear one).
    pub lit_modern: bool,
    /// `SkyParams::classic`: the Legacy lighting (else the linear one when
    /// not `lit_modern`).
    pub classic: bool,
    /// `SkyParams::no_post`.
    pub no_post: bool,
}

/// A sky's own kind shares: (modern, linear).
fn own(sky: &SkyFrame) -> (f32, f32) {
    let modern = tone_mapped(sky.probe_ambiance);
    (f32::from(u8::from(modern)), f32::from(u8::from(!modern && !sky.can_auto_adjust)))
}

fn stamp(probe_ambiance: f32, can_auto_adjust: bool) -> (u32, bool) {
    (probe_ambiance.to_bits(), can_auto_adjust)
}

impl SkyFrame {
    /// The blend's shares while the frame keeps the fields it was blended
    /// to.
    fn kind_blend(&self) -> Option<KindBlend> {
        self.kind.filter(|k| k.stamp == stamp(self.probe_ambiance, self.can_auto_adjust))
    }

    /// The frame's kind shares (modern, linear): the blend's, else its own.
    pub fn kind_shares(&self) -> (f32, f32) {
        self.kind_blend().map_or_else(|| own(self), |k| (k.modern, k.linear))
    }

    /// What the frame is lit and presented as. A sky of its own kind as
    /// before: `classic` its classic flag, `no_post` without a probe
    /// ambiance (LLPipeline::tonemap's no_post).
    pub fn kind(&self) -> SkyKind {
        match self.kind_blend() {
            None => {
                let modern = tone_mapped(self.probe_ambiance);
                SkyKind {
                    modern: f32::from(u8::from(modern)),
                    lit_modern: modern,
                    classic: self.can_auto_adjust,
                    no_post: !modern,
                }
            }
            Some(k) => {
                let lit_modern = k.modern >= 1.0;
                SkyKind {
                    modern: k.modern.clamp(0.0, 1.0),
                    lit_modern,
                    // the classic side is the Legacy one unless a linear
                    // sky has a share in it
                    classic: !lit_modern && k.linear == 0.0,
                    no_post: !lit_modern,
                }
            }
        }
    }

    /// The sky's brightness scale (`SkyParams::hdr_scale`): a modern sky's
    /// `sqrt(gamma) x 2` (LLSettingsVOSky::applySpecial, sky_hdr_scale), 1
    /// for the others, and between the two kinds the blend of
    /// aurora-render `sky_mode::hdr_scale`.
    pub fn hdr_scale(&self) -> f32 {
        let modern = self.gamma.max(0.0).sqrt() * 2.0;
        match self.kind_blend() {
            None => {
                if self.probe_ambiance != 0.0 {
                    modern
                } else {
                    1.0
                }
            }
            Some(k) => aurora_render::sky_mode::hdr_scale(modern, k.modern),
        }
    }

    /// The log's note of a blend between the kinds (empty otherwise).
    pub fn kind_note(&self) -> String {
        let k = self.kind();
        if k.modern > 0.0 && k.modern < 1.0 {
            format!(", between kinds: modern share {:.3}", k.modern)
        } else {
            String::new()
        }
    }
}

/// The light and hemisphere colours of a lighting model: sun, zenith,
/// horizon, ground.
pub type Light = (Vec3, Vec3, Vec3, Vec3);

/// The light and hemisphere colours of the
/// classic lighting `a` and the HDR one `b` at the modern share `m`.
pub fn mix_light(a: Light, b: Light, m: f32) -> Light {
    (a.0.lerp(b.0, m), a.1.lerp(b.1, m), a.2.lerp(b.2, m), a.3.lerp(b.3, m))
}

/// The kind shares of `a` blended into `b` at `t` (SkyFrame::lerp, whose
/// probe ambiance and classic flag the stamp repeats): None between two
/// skies of the same kind, each of its own.
pub fn lerp_kind(a: &SkyFrame, b: &SkyFrame, t: f32) -> Option<KindBlend> {
    let (sa, sb) = (a.kind_shares(), b.kind_shares());
    if sa == sb && a.kind_blend().is_none() && b.kind_blend().is_none() {
        return None;
    }
    let lf = |x: f32, y: f32| x + (y - x) * t;
    Some(KindBlend {
        modern: lf(sa.0, sb.0),
        linear: lf(sa.1, sb.1),
        stamp: stamp(lf(a.probe_ambiance, b.probe_ambiance), a.can_auto_adjust && b.can_auto_adjust),
    })
}

#[cfg(test)]
mod tests {
    use super::super::eep::{DayCycle, Settings, WaterFrame};
    use super::super::eep_env::{EnvSelector, FrameInput};
    use super::super::env::Environment;
    use super::super::env_select::SettingsKind;
    use super::*;
    use aurora_render::sky_mode::{self, Post};
    use aurora_render::{SkyParams, Tonemapper};
    use std::sync::Arc;

    fn modern(gamma: f32, ambiance: f32) -> SkyFrame {
        SkyFrame {
            gamma,
            probe_ambiance: ambiance,
            can_auto_adjust: false,
            ..Default::default()
        }
    }

    fn classic(gamma: f32) -> SkyFrame {
        SkyFrame {
            gamma,
            ..Default::default()
        }
    }

    fn linear(gamma: f32) -> SkyFrame {
        SkyFrame {
            can_auto_adjust: false,
            ..classic(gamma)
        }
    }

    /// The renderer's values before this patch for a sky of one kind.
    fn upstream_sky(sky: &SkyFrame) -> (bool, bool, f32) {
        let hdr = if sky.probe_ambiance != 0.0 {
            sky.gamma.max(0.0).sqrt() * 2.0
        } else {
            1.0
        };
        (sky.can_auto_adjust, !tone_mapped(sky.probe_ambiance), hdr)
    }

    fn upstream_post(s: &SkyParams) -> Post {
        Post {
            tonemapper: if s.no_post { 2.0 } else { 0.0 },
            legacy_gamma: if s.no_post && s.gamma.is_finite() { s.gamma.max(0.0) } else { 1.0 },
            tonemap_mix: if s.classic { 0.0 } else { 0.7 },
        }
    }

    /// What the presentation of a frame is made of: the modern share, the
    /// brightness scale, the post values, the exposure range and its dark
    /// term, and the lighting flags.
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Shown {
        modern: f32,
        hdr_scale: f32,
        post: Post,
        range: (f32, f32),
        dark: f32,
        classic: bool,
        no_post: bool,
    }

    fn shown(sky: &SkyFrame) -> Shown {
        let env = Environment::from_eep(sky, &WaterFrame::default(), glam::Vec2::ZERO, 96.0);
        let s = &env.sky;
        Shown {
            modern: s.modern,
            hdr_scale: s.hdr_scale,
            post: sky_mode::post(s, true, Tonemapper::KhronosNeutral, 0.7),
            range: sky_mode::exposure_range(s, true),
            dark: sky_mode::exposure_dark(s),
            classic: s.classic,
            no_post: s.no_post,
        }
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

    const DURATION: f32 = 29.0;
    const DT: f64 = 1.0 / 30.0;

    /// The frames of a 29 s manual change from `a` to `b` through the
    /// selector: one of `a` alone, every frame of the change, and one past
    /// its end; with each frame's progress.
    fn change(a: SkyFrame, b: SkyFrame) -> Vec<(f32, SkyFrame)> {
        let (ida, idb) = (uuid::Uuid::from_u128(0xa1), uuid::Uuid::from_u128(0xb2));
        let mut sel = EnvSelector::default();
        sel.request_local(ida, SettingsKind::Sky);
        sel.on_settings(ida, Some(Arc::new(Settings::Sky(a))));
        let mut frames = vec![(0.0, sel.frame(&input(0.0)).0)];
        sel.set_manual_transition(DURATION);
        sel.request_local(idb, SettingsKind::Sky);
        sel.on_settings(idb, Some(Arc::new(Settings::Sky(b))));
        let start = 1.0;
        let mut now = start;
        while now < start + f64::from(DURATION) + 2.0 * DT {
            let t = (((now - start) as f32) / DURATION).min(1.0);
            frames.push((t, sel.frame(&input(now)).0));
            now += DT;
        }
        frames
    }

    /// Display colours to watch: greys and tints from the darkest night to
    /// above white, at each end of the exposure range.
    fn displayed(s: &Shown) -> Vec<Vec3> {
        let mut out = Vec::new();
        for i in 0..40 {
            let x = 0.001 * 1.25f32.powi(i);
            for c in [Vec3::splat(x), Vec3::new(x, x * 0.7, x * 0.4)] {
                for exposure in [s.range.0, 1.0, s.range.1] {
                    // (the sky's own light is scaled by the brightness scale)
                    out.push(sky_mode::display(&s.post, exposure, c * s.hdr_scale));
                }
            }
        }
        out
    }

    /// `side_classic`: the sky that is not modern is a classic one (else a
    /// linear one).
    fn check_change(a: SkyFrame, b: SkyFrame, side_classic: bool) {
        let frames = change(a, b);
        let (first, after) = (shown(&a), shown(&b));
        assert_eq!(shown(&frames[0].1), first, "the sky before the change");
        assert_eq!(shown(&frames.last().expect("frames").1), after, "the sky after it");
        let step = DT as f32 / DURATION;
        let (sa, sb) = (first.modern, after.modern);
        let mut last: Option<(f32, Shown, Vec<Vec3>)> = None;
        for (t, sky) in &frames[1..] {
            let s = shown(sky);
            // the share is the change's own progress, as every lerped value
            // (the gamma's here)
            if *t < 1.0 {
                let g = (sky.gamma - a.gamma) / (b.gamma - a.gamma);
                assert!((s.modern - (sa + (sb - sa) * g)).abs() < 1e-4, "{t}: {} {g}", s.modern);
            }
            // the first frame of the change is still the first sky
            if *t == 0.0 {
                assert_eq!(s, first, "the change's first frame");
            }
            // the lighting model: the modern one only at a modern share of 1
            assert_eq!(s.no_post, s.modern < 1.0, "{t}");
            assert_eq!(s.classic, s.modern < 1.0 && side_classic, "{t}");
            let d = displayed(&s);
            if let Some((lt, l, ld)) = &last {
                let k = (t - lt) / step;
                // (at the fastest pace of the change's progress: pi / 2 the
                // linear one's when it is eased)
                let bound = |per: f32| per * std::f32::consts::FRAC_PI_2 * step * k.max(1.0) + 1e-5;
                assert!((s.modern - l.modern).abs() <= bound(1.0), "{t}");
                assert!(
                    (s.hdr_scale.ln() - l.hdr_scale.ln()).abs() <= bound(2.0),
                    "{t}: {} {}",
                    l.hdr_scale,
                    s.hdr_scale
                );
                // (the mix of a frame without a tone curve does nothing)
                let mix = |s: &Shown| if s.post.tonemapper > 1.5 { 0.0 } else { s.post.tonemap_mix };
                assert!((mix(&s) - mix(l)).abs() <= bound(0.7), "{t}");
                assert!((s.post.legacy_gamma - l.post.legacy_gamma).abs() <= bound(1.0), "{t}");
                assert!(
                    (s.range.0 - l.range.0).abs() <= bound(2.0) && (s.range.1 - l.range.1).abs() <= bound(4.0),
                    "{t}"
                );
                // (the dark scene term counts only with a range above 1)
                let dark = |s: &Shown| if s.range.1 > 1.0 { s.dark } else { 0.0 };
                assert!((dark(&s) - dark(l)).abs() <= bound(1.0), "{t}: {} {}", dark(l), dark(&s));
                for (i, (x, y)) in d.iter().zip(ld).enumerate() {
                    assert!((*x - *y).abs().max_element() <= bound(8.0), "{t} colour {i}: {y} -> {x}");
                }
            }
            last = Some((*t, s, d));
        }
    }

    #[test]
    fn a_change_from_a_modern_sky_to_a_classic_one_has_no_step() {
        check_change(modern(1.0, 1.0), classic(1.5), true);
        check_change(modern(0.8, 0.4), linear(1.2), false);
    }

    #[test]
    fn a_change_from_a_classic_sky_to_a_modern_one_has_no_step() {
        check_change(classic(1.5), modern(1.0, 1.0), true);
        check_change(linear(0.9), modern(1.3, 0.6), false);
    }

    #[test]
    fn skies_of_the_same_kind_keep_every_value_of_before() {
        for (a, b) in [
            (modern(1.0, 1.0), modern(1.6, 0.3)),
            (classic(1.0), classic(1.8)),
            (linear(1.0), linear(0.7)),
        ] {
            for (_, sky) in change(a, b) {
                assert!(sky.kind.is_none());
                let s = shown(&sky);
                let (classic, no_post, hdr) = upstream_sky(&sky);
                assert_eq!((s.classic, s.no_post), (classic, no_post));
                assert_eq!(s.hdr_scale.to_bits(), hdr.to_bits());
                assert_eq!(s.modern, if no_post { 0.0 } else { 1.0 });
                let env = Environment::from_eep(&sky, &WaterFrame::default(), glam::Vec2::ZERO, 96.0);
                assert_eq!(s.post, upstream_post(&env.sky));
                assert_eq!(s.dark, 1.0);
            }
        }
    }

    #[test]
    fn the_blend_holds_on_the_lerped_frame_and_not_on_an_edited_one() {
        let (a, b) = (modern(1.0, 0.8), classic(1.2));
        for i in 0..=20 {
            let t = i as f32 / 20.0;
            let f = a.lerp(&b, t);
            let (m, l) = f.kind_shares();
            assert!((m - (1.0 - t)).abs() < 1e-6 && l == 0.0, "{t}");
            // the classic side is the Legacy one, at every share
            assert!(f.kind().classic || f.kind().lit_modern, "{t}");
        }
        // the first frame of a change from a classic sky: classic, not the
        // linear sky the lerped flags alone would make
        let start = classic(1.0).lerp(&modern(1.0, 1.0), 0.0);
        assert_eq!(start.kind(), classic(1.0).kind());
        // a blend of blends
        let half = a.lerp(&b, 0.5);
        let quarter = half.lerp(&linear(1.0), 0.5);
        let (m, l) = quarter.kind_shares();
        assert!((m - 0.25).abs() < 1e-6 && (l - 0.5).abs() < 1e-6);
        assert!(!quarter.kind().classic && quarter.kind().no_post);
        // an editor gives the blended frame a probe ambiance: its own kind
        let mut edited = half;
        edited.probe_ambiance = 0.25;
        assert_eq!(edited.kind_shares(), (1.0, 0.0));
        assert_eq!(edited.hdr_scale(), upstream_sky(&edited).2);
    }

    /// A day cycle between a modern and a classic keyframe: the plain
    /// keyframe blend carries the shares too, no texture fade.
    #[test]
    fn a_day_cycle_between_kinds_moves_with_its_position() {
        let mut day = DayCycle::fixed(modern(1.0, 1.0), WaterFrame::default());
        day.sky_tracks[0] = vec![(0.0, modern(1.0, 1.0)), (0.5, classic(1.0))];
        let mut last = shown(&day.sky_at(0.0, 0));
        assert_eq!(last.modern, 1.0);
        for i in 1..=500 {
            let sky = day.sky_at(i as f32 / 1000.0, 0);
            assert!(sky.texture_fade.is_none());
            let s = shown(&sky);
            assert!((s.modern - (1.0 - i as f32 / 500.0)).abs() < 1e-4, "{i}");
            assert!((s.hdr_scale - last.hdr_scale).abs() < 0.01 && (s.post.tonemap_mix - last.post.tonemap_mix).abs() < 0.01);
            last = s;
        }
        assert_eq!(last, shown(&classic(1.0)));
    }
}
