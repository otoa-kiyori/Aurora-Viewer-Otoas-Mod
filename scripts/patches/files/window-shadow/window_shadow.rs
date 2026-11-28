//! The strength of the drop shadow under the windows and pop-ups
//! (Preferences › Appearance › Skin › Window shadow). Local patch set;
//! the Legacy Viewer has no such option.
//!
//! One setting scales one shadow, `egui::Visuals::window_shadow` and
//! `popup_shadow`, which `Theme::apply` builds and every window
//! (`widgets::Floater`, egui windows), menu, tooltip, overlay panel
//! (`widgets::overlay`) and the loading screen's map tile read. The
//! name plates, the dragged item's ghost and the alerts' glow have their
//! own shading and do not follow it.
//!
//! The strength goes from 0 (no shadow) to 2; 1, the default, is the
//! theme's own shadow exactly (4 points down, 14 of blur, 35 % of the bar
//! colour). Each half is a straight line between its two ends: below 1
//! the shadow shrinks and fades to nothing, above 1 it grows to `MAX`.
//! High contrast keeps no shadow whatever this says (`Theme::apply`).
//!
//! The choice is process-wide (`set`), read where the theme is applied,
//! like `crate::high_contrast`.

// aurora-local-patch: window-shadow-module

use std::sync::atomic::{AtomicU32, Ordering};

/// The settings' default: the theme's shadow as it was.
pub const DEFAULT: f32 = 1.0;
/// The strongest the slider goes.
pub const MAX_STRENGTH: f32 = 2.0;

/// One end of the scale: offset down, blur, spread (points) and the opacity
/// of the bar colour.
#[derive(Clone, Copy)]
struct Look {
    offset: f32,
    blur: f32,
    spread: f32,
    alpha: f32,
}

/// The theme's shadow (strength 1).
const BASE: Look = Look {
    offset: 4.0,
    blur: 14.0,
    spread: 0.0,
    alpha: 0.35,
};

/// A clearly visible, larger soft shadow (strength 2).
const MAX: Look = Look {
    offset: 10.0,
    blur: 40.0,
    spread: 2.0,
    alpha: 0.7,
};

static STRENGTH: AtomicU32 = AtomicU32::new(DEFAULT.to_bits());

/// The strength the theme uses from now on (applied at the next
/// `Theme::apply`).
pub fn set(strength: f32) {
    STRENGTH.store(clamp(strength).to_bits(), Ordering::Relaxed);
}

pub fn current() -> f32 {
    f32::from_bits(STRENGTH.load(Ordering::Relaxed))
}

/// The settings' strength, or in demo mode `AURORA_DEMO_WINDOW_SHADOW`
/// (0 to 2, a capture's, never saved).
pub fn wanted(settings: &crate::settings::Settings) -> f32 {
    if std::env::var_os("AURORA_DEMO").is_some()
        && let Some(s) = std::env::var("AURORA_DEMO_WINDOW_SHADOW")
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
            .filter(|s| s.is_finite())
    {
        return clamp(s);
    }
    settings.window_shadow
}

/// In range, and the default for a value that is not a number.
pub fn clamp(strength: f32) -> f32 {
    if strength.is_finite() {
        strength.clamp(0.0, MAX_STRENGTH)
    } else {
        DEFAULT
    }
}

/// The shadow of `strength`, in `color` (the theme passes its bar colour).
pub fn shadow(strength: f32, color: egui::Color32) -> egui::Shadow {
    let s = clamp(strength);
    if s <= 0.0 {
        return egui::Shadow::NONE;
    }
    let look = if s <= DEFAULT {
        let t = s / DEFAULT;
        Look {
            offset: BASE.offset * t,
            blur: BASE.blur * t,
            spread: BASE.spread * t,
            alpha: BASE.alpha * t,
        }
    } else {
        let t = (s - DEFAULT) / (MAX_STRENGTH - DEFAULT);
        let lerp = |a: f32, b: f32| a + (b - a) * t;
        Look {
            offset: lerp(BASE.offset, MAX.offset),
            blur: lerp(BASE.blur, MAX.blur),
            spread: lerp(BASE.spread, MAX.spread),
            alpha: lerp(BASE.alpha, MAX.alpha),
        }
    };
    // every value is in 0..=40 by construction
    egui::Shadow {
        offset: [0, look.offset.round() as i8],
        blur: look.blur.round() as u8,
        spread: look.spread.round() as u8,
        color: color.gamma_multiply(look.alpha),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BAR: egui::Color32 = egui::Color32::from_rgb(0x1b, 0x1e, 0x26);

    /// The default is the theme's shadow before this setting, bit for bit.
    #[test]
    fn the_default_is_the_former_shadow() {
        let former = egui::Shadow {
            offset: [0, 4],
            blur: 14,
            spread: 0,
            color: BAR.gamma_multiply(0.35),
        };
        assert_eq!(shadow(DEFAULT, BAR), former);
        assert_eq!(crate::settings::Settings::default().window_shadow, DEFAULT);
    }

    #[test]
    fn zero_is_no_shadow_and_the_top_is_the_largest() {
        assert_eq!(shadow(0.0, BAR), egui::Shadow::NONE);
        assert_eq!(shadow(-1.0, BAR), egui::Shadow::NONE);
        let top = shadow(MAX_STRENGTH, BAR);
        assert_eq!((top.offset, top.blur, top.spread), ([0, 10], 40, 2));
        assert_eq!(top.color, BAR.gamma_multiply(0.7));
        assert_eq!(shadow(9.0, BAR), top, "clamped");
        assert_eq!(shadow(f32::NAN, BAR), shadow(DEFAULT, BAR));
    }

    /// Stronger is never smaller nor lighter.
    #[test]
    fn the_shadow_grows_with_the_strength() {
        let mut last = shadow(0.0, BAR);
        for i in 1..=200 {
            let next = shadow(i as f32 / 100.0, BAR);
            assert!(next.offset[1] >= last.offset[1] && next.blur >= last.blur && next.spread >= last.spread);
            assert!(next.color.a() >= last.color.a(), "at {i} %");
            last = next;
        }
    }
}
