//! AURORA_DEMO_CAUSTICS: the caustics' options in the water demo (local
// aurora-local-patch: caustics-water-texture-demo-module
//! patch set), best with `AURORA_DEMO_WATER=caustics` (an underwater view of
//! the sunny lagoon, the waves held still like the `optics-` views).
//!
//! Words, comma separated:
//! - `texture`: Preferences' "Use water texture" on (implies `map`);
//! - `map`: the environment's default water normal map gets a stand-in, as
//!   the offline demo cannot download it (demo_legacy_bump.rs does the same
//!   for the bump presets); the water surface then draws it too;
//! - `no-map`: no stand-in even with `texture`: the analytic pattern the
//!   viewer falls back on while a normal map loads;
//! - `scale=<x>`: the caustics' Scale.

use glam::Vec3;
use std::sync::OnceLock;
use uuid::Uuid;

/// The camera of `AURORA_DEMO_WATER=caustics`: just under the surface, over
/// the lagoon's sand from about 1 m to 4 m deep, the sun behind.
pub const VIEW: (Vec3, Vec3) = (Vec3::new(44.0, 116.0, 19.3), Vec3::new(53.0, 125.0, 17.0));

#[derive(Debug, Default, PartialEq)]
pub struct Options {
    pub texture: bool,
    pub map: bool,
    pub scale: Option<f32>,
}

fn parse(value: &str) -> Options {
    let mut o = Options::default();
    let mut no_map = false;
    for word in value.split(',').map(str::trim) {
        match word {
            "texture" => o.texture = true,
            "map" => o.map = true,
            "no-map" => no_map = true,
            _ => {
                if let Some(scale) = word.strip_prefix("scale=").and_then(|s| s.parse::<f32>().ok()) {
                    o.scale = Some(scale);
                }
            }
        }
    }
    o.map = (o.map || o.texture) && !no_map;
    o
}

fn options() -> Option<&'static Options> {
    static OPTIONS: OnceLock<Option<Options>> = OnceLock::new();
    OPTIONS
        .get_or_init(|| std::env::var("AURORA_DEMO_CAUSTICS").ok().map(|v| parse(&v)))
        .as_ref()
}

/// The options into this run's settings (the water demo's setup).
pub fn configure(settings: &mut crate::settings::Settings) {
    let Some(o) = options() else { return };
    settings.water_effects.caustics = true;
    settings.water_effects.caustics_texture = o.texture;
    if let Some(scale) = o.scale {
        settings.water_effects.caustics_scale = aurora_render::water_effects::caustic_scale(scale);
    }
    log::info!(
        "caustics demo: texture {}, stand-in normal map {}, scale {}",
        o.texture,
        o.map,
        settings.water_effects.caustics_scale
    );
}

pub fn is_texture(id: &Uuid) -> bool {
    *id == crate::world::eep::DEFAULT_WATER_NORMAL && options().is_some_and(|o| o.map)
}

/// The stand-in water normal map, when asked for.
pub fn texture(id: &Uuid) -> Option<(Vec<u8>, u32, u32)> {
    is_texture(id).then(stand_in)
}

/// A tileable normal map of irregular ripples: random-phase waves with whole
/// numbers of periods across the image (so it repeats seamlessly, as the
/// surface tiles it), their slopes falling with frequency. Unlike the
/// analytic micro-waves (six regular modes), no direction dominates.
fn stand_in() -> (Vec<u8>, u32, u32) {
    const N: usize = 256;
    const WAVES: usize = 28;
    // a fixed integer sequence: the same image every run
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f32 / (1u64 << 53) as f32
    };
    let mut waves = Vec::with_capacity(WAVES);
    while waves.len() < WAVES {
        let k = [(next() * 19.0) as i32 - 9, (next() * 19.0) as i32 - 9];
        let length = ((k[0] * k[0] + k[1] * k[1]) as f32).sqrt();
        if !(2.0..=9.5).contains(&length) {
            continue;
        }
        // a slope of 1 / |k| along k: more energy in the long ripples
        waves.push((k[0] as f32, k[1] as f32, length, next() * std::f32::consts::TAU));
    }
    let slope = |u: f32, v: f32| {
        let mut g = [0.0f32; 2];
        for (kx, ky, length, phase) in &waves {
            let s = (std::f32::consts::TAU * (kx * u + ky * v) + phase).sin() / (length * length);
            g[0] -= kx * s;
            g[1] -= ky * s;
        }
        g
    };
    let mut gradients = Vec::with_capacity(N * N);
    for y in 0..N {
        for x in 0..N {
            gradients.push(slope(x as f32 / N as f32, y as f32 / N as f32));
        }
    }
    // tilt the normals about as much as Second Life's default water map
    let rms = (gradients.iter().map(|g| g[0] * g[0] + g[1] * g[1]).sum::<f32>() / gradients.len() as f32).sqrt();
    let gain = 0.45 / rms.max(1e-6);
    let mut rgba = Vec::with_capacity(N * N * 4);
    for g in gradients {
        let n = Vec3::new(g[0] * gain, g[1] * gain, 1.0).normalize();
        let byte = |c: f32| ((c * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
        rgba.extend_from_slice(&[byte(n.x), byte(n.y), byte(n.z), 255]);
    }
    (rgba, N as u32, N as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_words_set_the_switch_the_stand_in_and_the_scale() {
        assert_eq!(parse(""), Options::default());
        assert_eq!(
            parse("texture"),
            Options {
                texture: true,
                map: true,
                scale: None
            }
        );
        assert_eq!(
            parse("map, scale=2.5"),
            Options {
                texture: false,
                map: true,
                scale: Some(2.5)
            }
        );
        assert_eq!(
            parse("texture,no-map,scale=x"),
            Options {
                texture: true,
                map: false,
                scale: None
            }
        );
    }

    #[test]
    fn the_stand_in_tiles_and_its_normals_face_up() {
        let (rgba, w, h) = stand_in();
        assert_eq!((w, h, rgba.len()), (256, 256, 256 * 256 * 4));
        let at = |x: usize, y: usize| {
            let i = (y * 256 + x) * 4;
            [rgba[i] as i32, rgba[i + 1] as i32, rgba[i + 2] as i32]
        };
        let mut tilted = 0;
        for i in 0..256 {
            // across the seam the change is a neighbour's, not a jump
            for (a, b) in [(at(255, i), at(0, i)), (at(i, 255), at(i, 0))] {
                assert!(a.iter().zip(b).all(|(a, b)| (a - b).abs() <= 24), "{a:?} {b:?}");
            }
            for x in 0..256 {
                let n = at(x, i);
                assert!(n[2] > 160, "faces up: {n:?}");
                tilted += usize::from((n[0] - 128).abs() > 40 || (n[1] - 128).abs() > 40);
            }
        }
        assert!(tilted > 256 * 256 / 10, "ripples, not a flat map: {tilted}");
    }
}
