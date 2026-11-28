//! A change between two skies in the offline demo, through the environment
//! selector (`AURORA_DEMO_SKY_CHANGE=<sky A>;<sky B>[;<seconds>[;<frame>]]`).
//!
//! Sky A is picked as the local sky at frame 30, at once; sky B at the given
//! frame (600 by default), over the given seconds (5 by default). Both go
//! through `EnvSelector::request_local`, the call a sky picked in World ›
//! Environment makes, so the change blends exactly as a manual change does
//! in world; only its duration (the manual transition time, a setting) comes
//! from the variable, for that launch, and no setting is written. Swapping
//! A and B gives the other direction.
//!
//! A sky is a list of `key=value` separated by commas, every key optional:
//! - `probe=<x>`: the reflection probe ambiance. Without it the sky is
//!   classic (no key, LLSettingsSky::canAutoAdjust); `probe=0` writes the
//!   key at 0 (lit linearly, not tone mapped); above 0 the sky is modern;
//! - `gamma=<x>` (1);
//! - `sun=<elevation>[:<azimuth>]` in degrees (45:135);
//! - `moon=<elevation>[:<azimuth>]` (-30, opposite the sun: below the
//!   horizon unless given);
//! - `sunlight=<r:g:b or grey>` (2.2:2.34:2.7, as `AURORA_DEMO_SKY`);
//! - `ambient=<r:g:b or grey>` (the default sky's when not given);
//! - `haze=<horizon>:<density>`, `density=<x>` and `distance=<x>`: the
//!   legacy haze's horizon and density, density multiplier and distance
//!   multiplier, the four numbers the `sky frame:` log line shows as
//!   "haze a / b, density x, distance x";
//! - `cloudshadow=<x>` and `moonlight=<x>` (the moon's brightness).
//!
//! A key not given is not written: the sky takes the default value.
// aurora-local-patch: demo-sky-change-module

use crate::world::eep::Settings;
use aurora_llsd::{Llsd, llsd_map};
use std::sync::OnceLock;
use uuid::Uuid;

/// The local settings assets of the two skies (known locally, nothing is
/// fetched).
const SKY_A: Uuid = Uuid::from_u128(0xE0E0_5C00_0000_0000_0000_0000_0000_0001);
const SKY_B: Uuid = Uuid::from_u128(0xE0E0_5C00_0000_0000_0000_0000_0000_0002);
/// Sky A is picked at this frame, without a transition.
const FIRST_FRAME: u64 = 30;
const DEFAULT_SECONDS: f32 = 5.0;
const DEFAULT_FRAME: u64 = 600;

/// One sky of the variable.
#[derive(Debug, Clone, PartialEq)]
pub struct SkySpec {
    pub probe: Option<f32>,
    pub gamma: f32,
    /// Elevation and azimuth, degrees.
    pub sun: (f32, f32),
    pub moon: (f32, f32),
    pub sunlight: [f32; 3],
    pub ambient: Option<[f32; 3]>,
    /// Haze horizon and haze density.
    pub haze: Option<(f32, f32)>,
    pub density: Option<f32>,
    pub distance: Option<f32>,
    pub cloud_shadow: Option<f32>,
    pub moon_brightness: Option<f32>,
}

/// The whole variable.
#[derive(Debug, Clone, PartialEq)]
pub struct Change {
    pub a: SkySpec,
    pub b: SkySpec,
    pub seconds: f32,
    pub frame: u64,
}

fn colour(s: &str) -> Option<[f32; 3]> {
    let parts: Vec<f32> = s.split(':').map(|c| c.trim().parse().ok()).collect::<Option<_>>()?;
    let c = match parts[..] {
        [a] => [a, a, a],
        [r, g, b] => [r, g, b],
        _ => return None,
    };
    c.iter().all(|v| v.is_finite() && *v >= 0.0).then_some(c)
}

/// A finite number, 0 or more.
fn number(s: &str) -> Option<f32> {
    s.trim().parse::<f32>().ok().filter(|v| v.is_finite() && *v >= 0.0)
}

fn angles(s: &str) -> Option<(f32, Option<f32>)> {
    let mut it = s.split(':').map(|v| v.trim().parse::<f32>().ok().filter(|x| x.is_finite()));
    let elevation = it.next()??;
    let azimuth = match it.next() {
        Some(a) => Some(a?),
        None => None,
    };
    (it.next().is_none() && (-90.0..=90.0).contains(&elevation)).then_some((elevation, azimuth))
}

/// One sky (`key=value,…`); None for an unknown key or an unreadable value.
pub fn parse_sky(s: &str) -> Option<SkySpec> {
    let mut sky = SkySpec {
        probe: None,
        gamma: 1.0,
        sun: (45.0, 135.0),
        moon: (-30.0, 315.0),
        sunlight: [2.2, 2.34, 2.7],
        ambient: None,
        haze: None,
        density: None,
        distance: None,
        cloud_shadow: None,
        moon_brightness: None,
    };
    let mut moon = None;
    for field in s.split(',').map(str::trim).filter(|f| !f.is_empty()) {
        let (key, value) = field.split_once('=')?;
        let value = value.trim();
        match key.trim() {
            "probe" => sky.probe = Some(value.parse::<f32>().ok().filter(|p| p.is_finite() && *p >= 0.0)?),
            "gamma" => sky.gamma = value.parse::<f32>().ok().filter(|g| g.is_finite() && *g >= 0.0)?,
            "sun" => {
                let (el, az) = angles(value)?;
                sky.sun = (el, az.unwrap_or(sky.sun.1));
            }
            "moon" => moon = Some(angles(value)?),
            "sunlight" => sky.sunlight = colour(value)?,
            "ambient" => sky.ambient = Some(colour(value)?),
            "haze" => {
                let (horizon, density) = value.split_once(':')?;
                sky.haze = Some((number(horizon)?, number(density)?));
            }
            "density" => sky.density = Some(number(value)?),
            "distance" => sky.distance = Some(number(value)?),
            "cloudshadow" => sky.cloud_shadow = Some(number(value)?),
            "moonlight" => sky.moon_brightness = Some(number(value)?),
            _ => return None,
        }
    }
    let opposite = (sky.sun.1 + 180.0).rem_euclid(360.0);
    sky.moon = match moon {
        Some((el, az)) => (el, az.unwrap_or(opposite)),
        None => (-30.0, opposite),
    };
    Some(sky)
}

/// The variable; None when a part is missing or unreadable, or when sky B
/// would come before sky A is in.
pub fn parse(v: &str) -> Option<Change> {
    let mut parts = v.split(';').map(str::trim);
    let a = parse_sky(parts.next()?)?;
    let b = parse_sky(parts.next()?)?;
    let seconds = match parts.next().filter(|s| !s.is_empty()) {
        Some(s) => s.parse::<f32>().ok().filter(|s| (0.0..=60.0).contains(s))?,
        None => DEFAULT_SECONDS,
    };
    let frame = match parts.next().filter(|s| !s.is_empty()) {
        Some(f) => f.parse::<u64>().ok()?,
        None => DEFAULT_FRAME,
    };
    (parts.next().is_none() && frame > FIRST_FRAME).then_some(Change { a, b, seconds, frame })
}

fn change() -> Option<&'static Change> {
    static C: OnceLock<Option<Change>> = OnceLock::new();
    C.get_or_init(|| {
        let v = std::env::var("AURORA_DEMO_SKY_CHANGE").ok()?;
        let c = parse(&v);
        match &c {
            Some(c) => log::info!("demo sky change: {c:?}"),
            None => log::warn!("demo sky change: AURORA_DEMO_SKY_CHANGE unreadable: {v}"),
        }
        c
    })
    .as_ref()
}

/// The settings map of a sky, as a settings asset carries it.
pub fn sky_llsd(s: &SkySpec) -> Llsd {
    let quat = |(el, az): (f32, f32)| {
        let q = crate::ui::environment::rotation_from(az, el);
        Llsd::Array(q.to_array().into_iter().map(|v| Llsd::Real(f64::from(v))).collect())
    };
    let arr = |c: [f32; 3]| Llsd::Array(c.into_iter().map(|v| Llsd::Real(f64::from(v))).collect());
    let mut sky = llsd_map! {
        "type" => "sky",
        "sun_rotation" => quat(s.sun),
        "moon_rotation" => quat(s.moon),
        "sunlight_color" => arr(s.sunlight),
        "gamma" => f64::from(s.gamma),
    };
    let mut haze = aurora_llsd::Map::new();
    let mut put = |key: &str, v: Option<f32>| {
        if let Some(v) = v {
            haze.insert(key.into(), Llsd::Real(f64::from(v)));
        }
    };
    put("haze_horizon", s.haze.map(|h| h.0));
    put("haze_density", s.haze.map(|h| h.1));
    put("density_multiplier", s.density);
    put("distance_multiplier", s.distance);
    if let Some(a) = s.ambient {
        haze.insert("ambient".into(), arr(a));
    }
    if !haze.is_empty() {
        sky.insert("legacy_haze", Llsd::Map(haze));
    }
    if let Some(c) = s.cloud_shadow {
        sky.insert("cloud_shadow", f64::from(c));
    }
    if let Some(m) = s.moon_brightness {
        sky.insert("moon_brightness", f64::from(m));
    }
    if let Some(p) = s.probe {
        sky.insert("reflection_probe_ambiance", f64::from(p));
    }
    sky
}

/// The two skies' settings assets, for the local settings store.
pub fn settings_assets() -> Vec<(Uuid, Settings, Llsd)> {
    let Some(c) = change() else {
        return Vec::new();
    };
    [(SKY_A, &c.a), (SKY_B, &c.b)]
        .into_iter()
        .filter_map(|(id, s)| {
            let map = sky_llsd(s);
            Settings::from_llsd(&map).map(|settings| (id, settings, map))
        })
        .collect()
}

/// The sky to pick at this frame.
pub fn pick(frame: u64) -> Option<Uuid> {
    let c = change()?;
    let id = if frame == FIRST_FRAME {
        SKY_A
    } else if frame == c.frame {
        SKY_B
    } else {
        return None;
    };
    log::info!(
        "demo sky change: frame {frame}: sky {} ({} s)",
        if id == SKY_A { "A" } else { "B" },
        if id == SKY_A { 0.0 } else { c.seconds }
    );
    Some(id)
}

/// The manual transition time for this frame: none before sky B (sky A
/// comes at once), the variable's after; None without the variable.
pub fn transition(demo: bool, frame: u64) -> Option<f32> {
    let c = change().filter(|_| demo)?;
    Some(if frame < c.frame { 0.0 } else { c.seconds })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::eep::SkyFrame;

    const NIGHT: &str = "sun=-90,gamma=1,sunlight=0.349:0.356:0.660,ambient=0.204:0.242:0.330";
    const DAY: &str = "probe=1,gamma=1,sun=61.5,sunlight=1.741:1.741:1.929";

    fn frame(s: &SkySpec) -> SkyFrame {
        match Settings::from_llsd(&sky_llsd(s)) {
            Some(Settings::Sky(f)) => f,
            _ => panic!("not a sky"),
        }
    }

    #[test]
    fn sky_change_reads_both_skies_with_their_defaults() {
        let c = parse(&format!("{NIGHT};{DAY};29")).expect("her pair");
        assert_eq!(c.seconds, 29.0);
        assert_eq!(c.frame, DEFAULT_FRAME);
        assert_eq!(c.a.probe, None);
        assert_eq!(c.a.sun, (-90.0, 135.0));
        assert_eq!(c.a.moon, (-30.0, 315.0));
        assert_eq!(c.a.ambient, Some([0.204, 0.242, 0.330]));
        assert_eq!(c.b.probe, Some(1.0));
        assert_eq!(c.b.sunlight, [1.741, 1.741, 1.929]);
        assert_eq!(c.b.ambient, None);
        let c = parse(&format!("{DAY};{NIGHT}")).expect("swapped");
        assert_eq!((c.seconds, c.frame), (DEFAULT_SECONDS, DEFAULT_FRAME));
        let c = parse("sun=10:90,moon=20;probe=0,moon=5:200;5;900").expect("angles");
        assert_eq!((c.a.sun, c.a.moon), ((10.0, 90.0), (20.0, 270.0)));
        assert_eq!(c.b.moon, (5.0, 200.0));
        assert_eq!(c.frame, 900);
        let c = parse("haze=0:4,density=0.0003,distance=0,cloudshadow=0.27,moonlight=0.6;gamma=1").expect("haze");
        assert_eq!(c.a.haze, Some((0.0, 4.0)));
        assert_eq!((c.a.density, c.a.distance), (Some(0.0003), Some(0.0)));
        assert_eq!((c.a.cloud_shadow, c.a.moon_brightness), (Some(0.27), Some(0.6)));
        assert_eq!((c.b.haze, c.b.density, c.b.cloud_shadow), (None, None, None));
    }

    #[test]
    fn sky_change_refuses_what_it_cannot_read() {
        for v in [
            "",
            "gamma=1",
            "gamma=1;sun=95",
            "gamma=1;colour=1",
            "gamma=1;gamma=x",
            "gamma=1;sunlight=1:2",
            "gamma=1;probe=-1",
            "gamma=1;haze=0.2",
            "gamma=1;density=-1",
            "gamma=1;gamma=1;61",
            "gamma=1;gamma=1;5;30",
            "gamma=1;gamma=1;5;600;1",
        ] {
            assert_eq!(parse(v), None, "{v}");
        }
    }

    #[test]
    fn sky_change_skies_are_of_the_kind_asked() {
        let c = parse(&format!("{NIGHT};{DAY}")).expect("her pair");
        let (night, day) = (frame(&c.a), frame(&c.b));
        // classic: no probe ambiance key; modern: the key above 0
        assert!(night.can_auto_adjust);
        assert!(!day.can_auto_adjust);
        assert!((day.probe_ambiance - 1.0).abs() < 1e-6);
        let linear = frame(&parse_sky("probe=0").expect("linear"));
        assert!(!linear.can_auto_adjust);
        assert_eq!(linear.probe_ambiance, 0.0);
        // the sun and the moon where they were asked
        let elevation = |d: glam::Vec3| d.z.clamp(-1.0, 1.0).asin().to_degrees();
        assert!((elevation(night.sun_direction()) + 90.0).abs() < 0.1);
        assert!((elevation(day.sun_direction()) - 61.5).abs() < 0.1);
        assert!((elevation(night.moon_direction()) + 30.0).abs() < 0.1);
        assert!((night.gamma - 1.0).abs() < 1e-6);
        // the haze, the multipliers, the cloud shadow and the moon as asked
        let s = frame(&parse_sky("haze=0:4,density=0.0003,distance=0,cloudshadow=0.27,moonlight=0.6").expect("haze"));
        assert_eq!((s.haze_horizon, s.haze_density), (0.0, 4.0));
        assert!((s.density_multiplier - 0.0003).abs() < 1e-9);
        assert_eq!(s.distance_multiplier, 0.0);
        assert!((s.cloud_shadow - 0.27).abs() < 1e-6);
        assert!((s.moon_brightness - 0.6).abs() < 1e-6);
    }
}
