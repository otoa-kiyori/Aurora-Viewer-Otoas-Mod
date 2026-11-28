//! The sky's light handing over between the sun, the moon and none during
//! an environment change (local patch set,
//! scripts/patches/999-light-handover.toml).
//!
//! The Legacy Viewer lights the scene by the sun while it is above the
//! horizon, else by the moon while the moon is, else by nothing
//! (LLSettingsSky::getLightDirection, getIsSunUp, getIsMoonUp,
//! getSunMoonGlowFactor and calculateLightSettings's 0.001 moon brightness,
//! indra/llinventory/llsettingssky.cpp; skyV.glsl's 0.7 for the moon and
//! sun_up_factor, atmosphericsFuncs.glsl): the light's direction, the sky's
//! light (the moon's x 0.7), the objects' light, the haze's light, the glow
//! and the rainbow switch in one frame where the sun or the moon crosses
//! its horizon. A still sky is right that way, and so is a day cycle; an
//! environment change that sweeps the sun through the horizon (a night to a
//! day) shows the switch as a step of the whole frame.
//!
//! During an environment change, and only then, the frame carries a
//! [`Handover`] (SkyFrame::light, set by the selector's fade,
//! [`with_handover`]): the share of each light, the sun's and the moon's
//! (the rest the dark's), lerped at the change's progress from each side's
//! own (1 or 0 on a sky that is not changing, the shares shown on a change
//! begun over a running one), and the light's direction, which goes from
//! the source's light to the target's along the shorter great circle at the
//! same progress (not the blended sun's or moon's: those are often opposite
//! each other, and the light then has no way between them). Every value
//! the Legacy choice switches is the shares' mix of what each light gives
//! (world/env.rs): the sky's light (1, 0.7, 0.7), the glow factor (1, the
//! moon brightness x 0.25, 0), the objects' light (1, 1, 0), the rainbow's
//! and the halo's strength (1, 1, 0) and the diffuse light of the haze and
//! the classic lighting (the sun's, the moon's x its brightness x 0.6, the
//! dark's 0.001 x 0.6, along the light). Along a direction below the
//! horizon the sun's and the moon's light is none (their attenuation, exp(-a
//! / z), going to 0 as the direction reaches the horizon): a change between
//! a lit sky and one lit by nothing takes the light below the horizon with
//! no strength, so that the direction's crossing (the traced shadows' and
//! lights' switch, the classic ambient's |n.l|) happens where there is no
//! light to switch. The renderer is told the shares (SkyParams::handover):
//! the sky's and the clouds' glow is the sun's at its share (the Legacy
//! Viewer has none for the moon, the shaders' `glow_factor < 1` select),
//! the sky rays' strength and moon factor follow the shares (app.rs and
//! sky_mode.rs, where the rays' patches are there), and the water
//! caustics lose their gate on the true sun's height (the light already
//! carries the day factor).
//!
//! At the first frame of a change the shares and the direction are the
//! source's, at the last the target's, and a frame that is not changing
//! carries none: still skies, twilight ones included, and day cycles are
//! as before to the bit. A change between two skies lit by the same light
//! carries none either, unless the blended frame itself crosses a horizon
//! on the way (two suns low over opposite horizons): then the shares stay
//! that light and the direction is the blended body's, attenuated to none
//! below the horizon.
//!
//! `AURORA_LIGHT_HANDOVER=0` (a test variable) leaves every change as
//! before, for comparisons in one build.
// aurora-local-patch: light-handover-module

use super::eep::SkyFrame;
use glam::{Quat, Vec3};

/// The light of a frame while an environment change hands it over: the
/// sun's and the moon's shares (the rest the dark's) and its direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Handover {
    pub sun: f32,
    pub moon: f32,
    /// Towards the light (unit).
    pub dir: Vec3,
    /// The frame's sun and moon rotations when it was handed over: the
    /// handover holds while it keeps them (an editor or the photo window
    /// turning a body makes it a sky of its own light again).
    stamp: [u32; 8],
}

fn stamp(sky: &SkyFrame) -> [u32; 8] {
    let (s, m) = (sky.sun_rotation.to_array(), sky.moon_rotation.to_array());
    [s[0], s[1], s[2], s[3], m[0], m[1], m[2], m[3]].map(f32::to_bits)
}

/// The Legacy Viewer's light of a frame: the sun's and the moon's shares
/// (1 or 0) and the direction (LLSettingsSky::getLightDirection).
fn own(sky: &SkyFrame) -> (f32, f32, Vec3) {
    let (sun, moon) = (sky.sun_direction(), sky.moon_direction());
    if sun.z >= 0.0 {
        (1.0, 0.0, sun)
    } else if moon.z >= 0.0 {
        (0.0, 1.0, moon)
    } else {
        (0.0, 0.0, Vec3::NEG_Z)
    }
}

impl SkyFrame {
    /// The frame's handover while it keeps the rotations it was handed
    /// over with; None for a frame lit by its own light.
    pub fn light_handover(&self) -> Option<Handover> {
        self.light.filter(|h| h.stamp == stamp(self))
    }

    /// The frame's light: its handover's, else its own; and whether it is
    /// its own.
    fn light_shares(&self) -> ((f32, f32, Vec3), bool) {
        match self.light_handover() {
            Some(h) => ((h.sun, h.moon, h.dir), false),
            None => (own(self), true),
        }
    }
}

impl Handover {
    /// The dark's share.
    pub fn none(&self) -> f32 {
        (1.0 - self.sun - self.moon).max(0.0)
    }

    /// The share of a light from a body (the objects' light, the rainbow,
    /// the halo).
    pub fn lit(&self) -> f32 {
        self.sun + self.moon
    }

    /// The sky's light factor on its sunlight: 1 from the sun, 0.7 from the
    /// moon or the dark (skyV.glsl's "magic 0.7").
    pub fn sky_scale(&self) -> f32 {
        self.sun + 0.7 * (1.0 - self.sun)
    }

    /// The glow factor (LLSettingsSky::getSunMoonGlowFactor): 1 from the
    /// sun, the moon's brightness x 0.25, 0 from the dark.
    pub fn glow_factor(&self, moon_brightness: f32) -> f32 {
        self.sun + self.moon * moon_brightness * 0.25
    }

    /// The diffuse light of the haze and the classic lighting (world/env.rs
    /// `diffuse`: the sun's, else the moon's x 0.6): `base` the sky's
    /// sunlight x its transmittance, `atten` the attenuation along the
    /// light's path, `lighty` the Legacy Viewer's path length (1 / |z|, at
    /// least 8 float epsilons) which the dark's light takes from below. The
    /// sun's and the moon's take 1 / z: none from below the horizon, going
    /// to none as the light reaches it.
    pub fn diffuse(&self, base: Vec3, atten: Vec3, lighty: f32, moon_brightness: f32) -> Vec3 {
        let along = |l: f32| base * Vec3::from((-atten * l).to_array().map(f32::exp));
        let lit = 1.0 / self.dir.z.max(f32::EPSILON * 8.0);
        let mut d = along(lit) * (self.sun + self.moon * moon_brightness * 0.6);
        let none = self.none();
        if none > 0.0 {
            d += along(lighty) * (none * 0.001 * 0.6);
        }
        d
    }
}

/// The direction from `a` to `b` (unit) at `t`, along the shorter great
/// circle; exactly `a` at 0 and `b` at 1. Two opposite directions (the
/// dark's straight down and a moon at the zenith) turn about a horizontal
/// axis, through the horizon.
pub fn turn(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    if t <= 0.0 {
        return a;
    }
    if t >= 1.0 {
        return b;
    }
    let d = a.dot(b).clamp(-1.0, 1.0);
    if d > 0.9999 {
        return a.lerp(b, t).normalize_or(b);
    }
    if d < -0.9999 {
        let axis = a.cross(Vec3::X).normalize_or(a.cross(Vec3::Y).normalize_or(Vec3::Y));
        return (Quat::from_axis_angle(axis, std::f32::consts::PI * t) * a).normalize_or(b);
    }
    let theta = d.acos();
    let s = theta.sin();
    (a * ((1.0 - t) * theta).sin() / s + b * (t * theta).sin() / s).normalize_or(b)
}

/// Whether changes hand the light over (the test variable
/// `AURORA_LIGHT_HANDOVER=0` leaves them as before).
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AURORA_LIGHT_HANDOVER").map_or(true, |v| v.trim() != "0"))
}

/// The handover of `blended`, the selector's blend of `from` into `to` at
/// the change's progress `t`: None at the end (the target alone) and
/// between two skies lit by their own same light, unless the blended frame
/// itself is lit by another.
pub fn handover(blended: &SkyFrame, from: &SkyFrame, to: &SkyFrame, t: f32) -> Option<Handover> {
    if t.is_nan() || t >= 1.0 {
        return None;
    }
    let t = t.max(0.0);
    let ((sa, ma, da), own_a) = from.light_shares();
    let ((sb, mb, db), own_b) = to.light_shares();
    let (sun, moon, dir) = if own_a && own_b && (sa, ma) == (sb, mb) {
        let (s, m, d) = own(blended);
        if (s, m) == (sa, ma) {
            return None;
        }
        // the same light on both sides, another on the blended frame (a
        // body crossing its horizon on the way): that light stays, from
        // where the blended frame has it
        let d = if sa > 0.0 {
            blended.sun_direction()
        } else if ma > 0.0 {
            blended.moon_direction()
        } else {
            d
        };
        (sa, ma, d)
    } else {
        let lf = |x: f32, y: f32| x + (y - x) * t;
        (lf(sa, sb), lf(ma, mb), turn(da, db, t))
    };
    Some(Handover {
        sun,
        moon,
        dir,
        stamp: stamp(blended),
    })
}

/// The selector's blended frame with its handover (EnvSelector::frame).
pub fn with_handover(mut blended: SkyFrame, from: &SkyFrame, to: &SkyFrame, t: f32) -> SkyFrame {
    blended.light = if enabled() { handover(&blended, from, to, t) } else { None };
    blended
}

#[cfg(test)]
mod tests {
    use super::super::eep::{Settings, WaterFrame};
    use super::super::eep_env::{EnvSelector, FrameInput};
    use super::super::env::Environment;
    use super::super::env_select::SettingsKind;
    use super::*;
    use std::sync::Arc;

    /// A sky with its sun and moon at these elevations and azimuths
    /// (degrees, as the environment editor turns them).
    fn sky(sun: (f32, f32), moon: (f32, f32), modern: bool) -> SkyFrame {
        let rot = |(el, az): (f32, f32)| crate::ui::environment::rotation_from(az, el);
        SkyFrame {
            sun_rotation: rot(sun),
            moon_rotation: rot(moon),
            probe_ambiance: if modern { 1.0 } else { 0.0 },
            can_auto_adjust: !modern,
            ..Default::default()
        }
    }

    /// A classic night under a high moon and a modern day, from a viewer's
    /// log, and the same night with the moon set.
    fn night() -> SkyFrame {
        SkyFrame {
            sunlight: Vec3::new(0.349, 0.356, 0.660),
            ambient: Vec3::new(0.204, 0.242, 0.330),
            haze_horizon: 0.0,
            haze_density: 4.0,
            density_multiplier: 0.0003,
            distance_multiplier: 0.0,
            cloud_shadow: 0.27,
            ..sky((-90.0, 0.0), (90.0, 0.0), false)
        }
    }

    fn dark_night() -> SkyFrame {
        SkyFrame {
            moon_rotation: crate::ui::environment::rotation_from(180.0, -45.0),
            ..night()
        }
    }

    fn day() -> SkyFrame {
        SkyFrame {
            sunlight: Vec3::new(1.74118, 1.74118, 1.92941),
            ambient: Vec3::ZERO,
            haze_horizon: 0.25,
            haze_density: 0.78,
            density_multiplier: 0.0001126,
            distance_multiplier: 6.54,
            cloud_shadow: 0.54,
            ..sky((61.5, 320.8), (-45.0, 180.0), true)
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

    const DT: f64 = 1.0 / 60.0;

    /// The frames of a manual change of `seconds` from `a` to `b` through
    /// the selector, at 60 frames a second: `a` alone, every frame of the
    /// change, and two past its end.
    fn change(a: SkyFrame, b: SkyFrame, seconds: f32) -> Vec<SkyFrame> {
        let (ida, idb) = (uuid::Uuid::from_u128(0xa1), uuid::Uuid::from_u128(0xb2));
        let mut sel = EnvSelector::default();
        sel.request_local(ida, SettingsKind::Sky);
        sel.on_settings(ida, Some(Arc::new(Settings::Sky(a))));
        let mut frames = vec![sel.frame(&input(0.0)).0];
        sel.set_manual_transition(seconds);
        sel.request_local(idb, SettingsKind::Sky);
        sel.on_settings(idb, Some(Arc::new(Settings::Sky(b))));
        let mut now = 1.0;
        while now < 1.0 + f64::from(seconds) + 3.0 * DT {
            frames.push(sel.frame(&input(now)).0);
            now += DT;
        }
        frames
    }

    fn env(sky: &SkyFrame) -> Environment {
        Environment::from_eep(sky, &WaterFrame::default(), glam::Vec2::ZERO, 96.0)
    }

    /// What the Legacy choice switches, as the frame made it before this
    /// patch (world/env.rs before it): the light, the sky's light, the
    /// objects' light, the glow factor, the rainbow and the halo.
    fn upstream(sky: &SkyFrame) -> (Vec3, Vec3, Vec3, f32, f32, f32) {
        let (sun, moon) = (sky.sun_direction(), sky.moon_direction());
        let (sun_up, moon_up) = (sun.z >= 0.0, moon.z >= 0.0);
        let light = if sun_up {
            sun
        } else if moon_up {
            moon
        } else {
            Vec3::NEG_Z
        };
        let objects = if sun_up || moon_up {
            super::super::env::normalized_light(sky.sunlight)
        } else {
            Vec3::ZERO
        };
        let glow = if sun_up {
            1.0
        } else if moon_up {
            sky.moon_brightness * 0.25
        } else {
            0.0
        };
        let lit = |x: f32| if sun_up || moon_up { x } else { 0.0 };
        (
            light,
            if sun_up { sky.sunlight } else { sky.sunlight * 0.7 },
            objects,
            glow,
            lit(sky.moisture_level),
            lit(sky.ice_level),
        )
    }

    fn switched(e: &Environment) -> (Vec3, Vec3, Vec3, f32, f32, f32) {
        (
            e.sun_dir,
            e.sky.sunlight,
            e.sky.object_sunlight,
            e.sky.sun_moon_glow_factor,
            e.sky.moisture_level,
            e.sky.ice_level,
        )
    }

    fn bits(v: (Vec3, Vec3, Vec3, f32, f32, f32)) -> Vec<u32> {
        let mut out: Vec<u32> = [v.0, v.1, v.2].iter().flat_map(|x| x.to_array()).map(f32::to_bits).collect();
        out.extend([v.3, v.4, v.5].map(f32::to_bits));
        out
    }

    /// A still sky, the sun or the moon a hair or a few degrees from the
    /// horizon included, carries no handover and is the Legacy choice to
    /// the bit, through the selector too.
    #[test]
    fn a_still_sky_is_lit_as_before_to_the_bit() {
        let els = [-90.0, -45.0, -6.0, -3.0, -1.0, -0.2, 0.0, 0.2, 1.0, 3.0, 6.0, 45.0, 90.0];
        for &s in &els {
            for &m in &els {
                for modern in [false, true] {
                    let mut k = sky((s, 30.0), (m, 210.0), modern);
                    k.moisture_level = 0.4;
                    k.ice_level = 0.3;
                    assert!(k.light_handover().is_none());
                    let e = env(&k);
                    assert_eq!(e.sky.handover, None, "{s} {m}");
                    assert_eq!(bits(switched(&e)), bits(upstream(&k)), "{s} {m} {modern}");
                    // shown by the selector, still
                    let frames = change(k, k, 5.0);
                    for f in &frames {
                        assert!(f.light_handover().is_none(), "{s} {m}");
                    }
                }
            }
        }
    }

    /// A day cycle's keyframes do not hand over: its blend is the plain one.
    #[test]
    fn a_day_cycle_blend_carries_no_handover() {
        let f = night().lerp(&day(), 0.5);
        assert!(f.light.is_none());
    }

    /// The first frame of a change is the source's light, the last the
    /// target's, and between them the shares move with the progress and
    /// the direction turns at its pace, both ways, for the moon-up and the
    /// moon-down night, at 5 s and 29 s.
    #[test]
    fn a_change_hands_the_light_over_from_its_source_to_its_target() {
        for (a, b) in [(night(), day()), (day(), night()), (dark_night(), day()), (day(), dark_night())] {
            for seconds in [5.0f32, 29.0] {
                let frames = change(a, b, seconds);
                let (first, last) = (&frames[0], frames.last().expect("frames"));
                assert!(first.light_handover().is_none() && last.light_handover().is_none());
                assert_eq!(bits(switched(&env(first))), bits(upstream(&a)));
                assert_eq!(bits(switched(&env(last))), bits(upstream(&b)));
                // the change's fastest pace: pi / 2 of its linear one (the
                // eased progress), or 1 without it
                let pace = std::f32::consts::FRAC_PI_2 * (DT as f32) / seconds;
                let mut prev = switched(&env(first));
                let mut handed = 0;
                for f in &frames[1..] {
                    let e = env(f);
                    let now = switched(&e);
                    if let Some(h) = f.light_handover() {
                        handed += 1;
                        assert!((0.0..=1.0).contains(&h.sun) && (0.0..=1.0).contains(&h.moon));
                        assert!(h.sun + h.moon <= 1.0 + 1e-6);
                        assert_eq!(e.sky.handover, Some([h.sun, h.moon]));
                    }
                    // the direction turns by at most the angle between the
                    // two lights at the pace (under pi)
                    let turned = prev.0.dot(now.0).clamp(-1.0, 1.0).acos();
                    assert!(turned <= std::f32::consts::PI * pace + 1e-3, "{seconds} s: turned {turned}");
                    // the sky's light, the glow factor, the objects' light:
                    // by their difference between the lights at the pace
                    let span = (b.sunlight - a.sunlight).abs().max_element() + 0.3 * a.sunlight.max(b.sunlight).max_element();
                    assert!((now.1 - prev.1).abs().max_element() <= span * pace + 1e-4, "{seconds} s sky light");
                    assert!((now.3 - prev.3).abs() <= pace + 1e-5, "{seconds} s glow");
                    assert!((now.2 - prev.2).abs().max_element() <= 2.0 * pace + 1e-3, "{seconds} s objects");
                    prev = now;
                }
                assert!(handed > 10, "{seconds} s: {handed} frames handed over");
            }
        }
    }

    /// Every light the frame shades with moves by a fraction of itself a
    /// frame: no step of the classic or the modern lighting's colours, the
    /// haze or the sky's light where the sun crosses the horizon (the
    /// Legacy choice stepped them there), for both nights, both ways, at
    /// 5 s and 29 s; the same frames without their handover (the Legacy
    /// choice) step.
    #[test]
    fn no_light_steps_through_the_change() {
        let lum = |c: Vec3| c.max(Vec3::ZERO).dot(Vec3::new(0.2126, 0.7152, 0.0722));
        let watch = |e: &Environment| {
            [
                lum(e.sun_color) + lum(e.zenith),
                lum(e.zenith),
                lum(e.fog_color),
                lum(e.sky.sunlight),
                lum(e.sun_color) * e.sun_dir.z.max(0.0) + lum(e.horizon),
            ]
        };
        let worst = |frames: &[SkyFrame]| {
            let mut worst = [0.0f32; 5];
            for w in frames.windows(2) {
                let (x, y) = (watch(&env(&w[0])), watch(&env(&w[1])));
                for i in 0..5 {
                    // in stops, above a floor of the night's darkest
                    let step = ((y[i] + 1e-4) / (x[i] + 1e-4)).log2().abs();
                    worst[i] = worst[i].max(step);
                }
            }
            worst
        };
        for (a, b) in [(night(), day()), (day(), night()), (dark_night(), day()), (day(), dark_night())] {
            for seconds in [5.0f32, 29.0] {
                let frames = change(a, b, seconds);
                let w = worst(&frames);
                let bound = if seconds < 10.0 { 0.15 } else { 0.05 };
                assert!(w.iter().all(|w| *w < bound), "{seconds} s: {w:?}");
                let legacy: Vec<SkyFrame> = frames.iter().map(|f| SkyFrame { light: None, ..*f }).collect();
                let l = worst(&legacy);
                assert!(l.iter().any(|w| *w > 0.3), "{seconds} s without: {l:?}");
            }
        }
    }

    /// The blended frame's direction when both sides are lit by the same
    /// light it keeps, and the Legacy choice switches on the way: that light
    /// stays the light, faded below the horizon.
    #[test]
    fn a_body_dipping_on_the_way_stays_the_light() {
        // the sun 5 degrees up, turned 140 degrees about an axis a little off
        // the vertical: 3 degrees up at the end, 2.5 below on the way
        let a = sky((5.0, 0.0), (-40.0, 180.0), false);
        let axis = Vec3::new(0.045, 0.168, 0.985).normalize();
        let b = SkyFrame {
            sun_rotation: Quat::from_axis_angle(axis, 140f32.to_radians()) * a.sun_rotation,
            ..a
        };
        assert!(b.sun_direction().z > 0.0);
        let mut dipped = 0;
        for i in 0..=100 {
            let t = i as f32 / 100.0;
            let f = a.lerp(&b, t);
            let h = handover(&f, &a, &b, t);
            if f.sun_direction().z < 0.0 {
                dipped += 1;
                let h = h.expect("a dip hands over");
                assert_eq!((h.sun, h.moon), (1.0, 0.0));
                assert_eq!(h.dir, f.sun_direction());
            } else {
                assert!(h.is_none(), "{t}");
            }
        }
        assert!(dipped > 0, "the sun does not dip on this path");
    }

    /// A change begun over a running one starts from the shown shares and
    /// direction; an edited body drops them.
    #[test]
    fn a_change_over_a_change_starts_from_the_shown_light() {
        let (a, b) = (night(), day());
        let mid = with_handover(a.lerp(&b, 0.4), &a, &b, 0.4);
        let h = mid.light_handover().expect("handed over");
        assert!((h.sun - 0.4).abs() < 1e-6 && (h.moon - 0.6).abs() < 1e-6);
        // back to the night from there
        let c = night();
        let start = with_handover(mid.lerp(&c, 0.0), &mid, &c, 0.0);
        let s = start.light_handover().expect("handed over");
        assert_eq!((s.sun, s.moon), (h.sun, h.moon));
        assert!(s.dir.distance(h.dir) < 1e-6);
        let half = with_handover(mid.lerp(&c, 0.5), &mid, &c, 0.5);
        let q = half.light_handover().expect("handed over");
        assert!((q.sun - 0.2).abs() < 1e-6 && (q.moon - 0.8).abs() < 1e-6);
        // an editor turns the sun: its own light again
        let mut edited = mid;
        edited.sun_rotation = crate::ui::environment::rotation_from(0.0, 30.0);
        assert!(edited.light_handover().is_none());
        assert_eq!(bits(switched(&env(&edited))), bits(upstream(&edited)));
    }

    /// The direction's way: exact ends, unit, the shorter arc, and opposite
    /// directions (the dark's down and a moon at the zenith) through the
    /// horizon.
    #[test]
    fn the_light_turns_along_the_shorter_way() {
        let (a, b) = (Vec3::Z, Vec3::new(0.3, -0.4, 0.86).normalize());
        assert_eq!(turn(a, b, 0.0), a);
        assert_eq!(turn(a, b, 1.0), b);
        let full = a.angle_between(b);
        for i in 1..20 {
            let t = i as f32 / 20.0;
            let d = turn(a, b, t);
            assert!((d.length() - 1.0).abs() < 1e-5);
            assert!((a.angle_between(d) - full * t).abs() < 1e-4, "{t}");
            assert!(d.z > 0.85);
        }
        let mut prev = Vec3::NEG_Z;
        for i in 1..=40 {
            let d = turn(Vec3::NEG_Z, Vec3::Z, i as f32 / 40.0);
            assert!((d.length() - 1.0).abs() < 1e-5);
            assert!(prev.angle_between(d) < std::f32::consts::PI / 40.0 + 1e-4);
            prev = d;
        }
        assert!(turn(Vec3::NEG_Z, Vec3::Z, 0.5).z.abs() < 1e-5);
        // the sun exactly opposite the moon, the sky's two bodies: the light
        // goes from one to the other over the zenith
        let sun = Vec3::new(0.5, 0.0, 0.866).normalize();
        let moon = Vec3::new(-0.5, 0.0, 0.866).normalize();
        assert!(turn(moon, sun, 0.5).z > 0.99);
    }

    /// The dark's share, the shares' mixes, and the diffuse light: each
    /// light's own at its share, none from a body's light from below the
    /// horizon.
    #[test]
    fn the_mixes_are_each_lights_own() {
        let h = |sun: f32, moon: f32, dir: Vec3| Handover {
            sun,
            moon,
            dir,
            stamp: [0; 8],
        };
        let up = Vec3::new(0.0, 0.6, 0.8);
        assert_eq!(h(1.0, 0.0, up).sky_scale(), 1.0);
        assert!((h(0.0, 1.0, up).sky_scale() - 0.7).abs() < 1e-6);
        assert!((h(0.0, 0.0, up).sky_scale() - 0.7).abs() < 1e-6);
        assert_eq!(h(1.0, 0.0, up).glow_factor(0.5), 1.0);
        assert!((h(0.0, 1.0, up).glow_factor(0.5) - 0.125).abs() < 1e-6);
        assert_eq!(h(0.0, 0.0, up).glow_factor(0.5), 0.0);
        assert!((h(0.3, 0.3, up).none() - 0.4).abs() < 1e-6);
        let (base, atten) = (Vec3::splat(2.0), Vec3::splat(0.2));
        let sun = h(1.0, 0.0, up).diffuse(base, atten, 1.0 / 0.8, 0.5);
        assert!((sun - base * (-0.2f32 / 0.8).exp()).abs().max_element() < 1e-6);
        let moon = h(0.0, 1.0, up).diffuse(base, atten, 1.0 / 0.8, 0.5);
        assert!((moon - sun * 0.3).abs().max_element() < 1e-6);
        // from below the horizon: the bodies give none, the dark its own
        let below = Vec3::new(0.0, 0.6, -0.8);
        assert_eq!(h(0.5, 0.5, below).diffuse(base, atten, 1.0 / 0.8, 0.5), Vec3::ZERO);
        let dark = h(0.0, 0.0, Vec3::NEG_Z).diffuse(base, atten, 1.0, 0.5);
        assert!((dark - base * (-0.2f32).exp() * 0.0006).abs().max_element() < 1e-7);
    }
}
