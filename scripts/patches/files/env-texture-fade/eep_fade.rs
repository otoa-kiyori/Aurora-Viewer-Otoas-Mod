//! Cross-fades of the environment's textures (local patch set,
//! scripts/patches/998-env-texture-fade.toml). While the environment
//! changes (the selector's fade, `EnvSelector::frame` in world/eep_env.rs:
//! another environment, a parcel's, an altitude zone's, a personal one, a
//! fade begun over another one), each texture element of the sky and the
//! water (the cloud noise, the sun, the moon, the rainbow, the 22° halo and
//! the water normal map) shows its two layers at once, each drawn exactly as
//! its own side would draw it: `first · (1 − blend) + next · blend`, the
//! blend being the change's own linear progress. The frame used to switch to
//! the next texture half way and to slide, zoom, warp or speed up the one it
//! showed with the lerped values.
//!
//! A layer is a texture with everything that places, sizes, moves or warps
//! its pattern:
//! - the clouds ([`CloudLayer`]): the texture, `cloud_pos_density1` and 2 xy
//!   (the offsets), `cloud_scale` (the zoom of the dome coordinates),
//!   `cloud_variance` (the distortion taps) and `cloud_scroll_rate`, with its
//!   own scroll position, which [`CloudScroll`] advances at its own rate;
//! - the water ([`WaterLayer`]): the normal map, `wave1` and `wave2` (both
//!   the direction and the speed of its layers) and `normal_scale`. The
//!   water has no accumulated position: the shaders place each map layer at
//!   `wave × time` (env.wgsl `water_pixel`, patch 91's slope pass, patch
//!   120's shape), so a layer drawn with its own waves is also at its own
//!   place, and stays there when its slot changes.
//!
//! A difference in any of them starts the two-layer fade. What only changes
//! an amount or a colour, not where the pattern is, stays blended as a
//! number, on both layers alike: the cloud colour, the densities
//! (`cloud_pos_density1` and 2 z), `cloud_shadow` (the cover); the water's
//! fog, fresnel and blur multiplier (how far the normals bend the
//! reflection and the refraction). Each of them moves its value from one
//! side to the other linearly, and none of them slides the pattern.
//!
//! Only that change fades: [`SkyFrame::fade_to`] and [`WaterFrame::fade_to`]
//! (and their `_with` forms), which the selector alone calls. The blend
//! between the keyframes of a day cycle (`DayCycle::sky_at` / `water_at`)
//! stays the plain `lerp`, exactly as before: its ids switch half way, its
//! offsets, scale, variance, scroll rate, waves and normal scale are lerped,
//! and it carries no fade. A day cycle moves between two keyframes for
//! hours; two layers all that time would double the environment's texture
//! reads for nothing a user sees change.
//!
//! A second texture still loading ([`TextureClock`]): its layer is kept at
//! share 0 (the first one shows alone, where it is) and, once it is in, its
//! share rises from 0 at twice the change's pace until it meets the change's
//! progress, then follows it; if the change's time is over first, the change
//! goes on for the textures alone (the frame's values are the target's) at
//! that pace up to 1, the old layer staying until then. Never a switch:
//! only a texture that is not in 30 s after the change's time gives up and
//! switches, as before.
//!
//! The Legacy Viewer blends some of them: LLSettingsSky::blend and
//! LLSettingsWater::blend (indra/llinventory/llsettingssky.cpp,
//! llsettingswater.cpp) keep the next textures' ids and a blend factor that
//! cloudsF.glsl, sunDiscF.glsl and waterF.glsl mix with
//! (LLDrawPoolWLSky::renderSkyCloudsDeferred and renderHeavenlyBodies,
//! indra/newview/lldrawpoolwlsky.cpp; LLDrawPoolWater::renderPostDeferred,
//! lldrawpoolwater.cpp). There both textures share the lerped offsets, scale
//! and waves (they slide and zoom), the scroll rate is lerped into one shared
//! scroll, the moon keeps its first texture to the end of the fade (its
//! blend is commented out in renderHeavenlyBodies) and the rainbow and the
//! halo switch at the end. Here every element fades and each layer keeps
//! its own place, size and motion: better than the Legacy Viewer.
//!
//! A frame keeps its fade with its own texture fields as they were when it
//! was blended (`Stamped::still`): a frame changed after its blend (an
//! editor, RLVa, an experience's push) no longer matches them and shows its
//! own textures. The frames' own fields, which the editors, RLVa's replies
//! and the settings files read, keep their values (the switch half way).
//! Nothing here is sent or saved: it only draws.
// aurora-local-patch: env-texture-fade-module

use super::eep::{SkyFrame, WaterFrame};
use glam::{Vec2, Vec3};
use uuid::Uuid;

/// One cloud noise texture with what places, sizes, moves and warps its
/// pattern: `cloud_pos_density1` and `cloud_pos_density2` xy, `cloud_scale`,
/// `cloud_variance` and `cloud_scroll_rate` (its scroll position is kept by
/// [`CloudScroll`]). The densities, the colour and the cover stay the sky's,
/// blended as numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloudLayer {
    pub id: Uuid,
    pub offset1: Vec2,
    pub offset2: Vec2,
    pub scale: f32,
    pub variance: f32,
    pub rate: Vec2,
}

/// One water normal map with what moves and scales it: the two waves (the
/// direction and the speed of its layers) and the normal scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaterLayer {
    pub id: Uuid,
    pub wave1: Vec2,
    pub wave2: Vec2,
    pub scale: Vec3,
}

/// A texture element during a cross-fade: `first` drawn at `1 - blend`,
/// `next` at `blend`. Still (one layer) when both are the same; a `next`
/// at blend 0 is a layer whose texture is still loading, kept (and asked
/// for) without a share yet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fade<L> {
    pub first: L,
    pub next: L,
    pub blend: f32,
}

/// An element's share of its incoming layer for one frame of a change
/// ([`TextureClock::advance`]): `t`, and `hold` while that layer's texture
/// is not in yet (its layer is kept at share `t`, 0 until it came).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Share {
    pub t: f32,
    pub hold: bool,
}

impl Share {
    /// The change's own progress, nothing waited for.
    pub fn at(t: f32) -> Share {
        Share { t, hold: false }
    }
}

fn unit(v: f32) -> f32 {
    if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 }
}

impl<L: Copy + PartialEq> Fade<L> {
    pub fn still(layer: L) -> Fade<L> {
        Fade {
            first: layer,
            next: layer,
            blend: 0.0,
        }
    }

    /// Two textures show.
    pub fn fading(&self) -> bool {
        self.blend > 0.0 && self.first != self.next
    }

    /// Two layers are kept: both shown, or the second one waiting for its
    /// texture at share 0. The first then keeps its own place.
    pub fn two(&self) -> bool {
        self.first != self.next
    }

    /// The layers that show, a waiting second one left out.
    fn settled(&self) -> Fade<L> {
        if self.blend <= 0.0 { Fade::still(self.first) } else { *self }
    }

    /// The fade from this one to `to` at `t` (0..1, as SkyFrame::lerp):
    /// the four weighted layers, equal ones merged, the two heaviest kept,
    /// their weights renormalised. They keep the order they come in (this
    /// fade's first, its next, then `to`'s), so a layer stays in its slot
    /// as the weights move and its texture is never handed to the other.
    pub fn towards(&self, to: &Fade<L>, t: f32) -> Fade<L> {
        let (t, a, b) = (unit(t), unit(self.blend), unit(to.blend));
        let weighted = [
            (self.first, (1.0 - t) * (1.0 - a)),
            (self.next, (1.0 - t) * a),
            (to.first, t * (1.0 - b)),
            (to.next, t * b),
        ];
        let mut layers = [weighted[0]; 4];
        let mut n = 0;
        for (layer, w) in weighted {
            match layers[..n].iter_mut().find(|(l, _)| *l == layer) {
                Some(kept) => kept.1 += w,
                None => {
                    layers[n] = (layer, w);
                    n += 1;
                }
            }
        }
        let layers = &layers[..n];
        // the heaviest (the earlier of equal weights), but `skip`
        let heaviest = |skip: Option<usize>| {
            (0..layers.len())
                .filter(|i| Some(*i) != skip)
                .reduce(|best, i| if layers[i].1 > layers[best].1 { i } else { best })
        };
        let Some(i) = heaviest(None) else {
            return Fade::still(self.first);
        };
        match heaviest(Some(i)) {
            Some(j) if layers[j].1 > 0.0 => {
                let ((first, w0), (next, w1)) = (layers[i.min(j)], layers[i.max(j)]);
                Fade {
                    first,
                    next,
                    blend: w1 / (w0 + w1),
                }
            }
            _ => Fade::still(layers[i].0),
        }
    }

    /// Whether `to`'s layer can wait for its texture in this fade's second
    /// slot: one layer shows on each side, and they differ. (A change begun
    /// over a running one has two layers already: its incoming one takes a
    /// slot by [`Self::towards`], and a texture still loading there has no
    /// share yet, `Environment::set_fade_slots`.)
    pub fn can_wait(&self, to: &Fade<L>) -> bool {
        let (a, b) = (self.settled(), to.settled());
        a.first == a.next && b.first == b.next && a.first != b.first
    }

    /// [`Self::towards`] at the share of `share`; while `to`'s texture is
    /// still loading (`hold`, and [`Self::can_wait`]) its layer is kept in
    /// the second slot at that share (0 until it came) instead of being left
    /// out, so that it is asked for while the first layer shows alone.
    pub fn join(&self, to: &Fade<L>, share: Share) -> Fade<L> {
        if share.hold && self.can_wait(to) {
            let (a, b) = (self.settled().first, to.settled().first);
            let s = unit(share.t);
            return if s >= 1.0 {
                Fade::still(b)
            } else {
                Fade {
                    first: a,
                    next: b,
                    blend: s,
                }
            };
        }
        self.towards(to, share.t)
    }
}

/// The sky's texture elements as they show.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkyFade {
    pub cloud: Fade<CloudLayer>,
    pub sun: Fade<Uuid>,
    pub moon: Fade<Uuid>,
    pub rainbow: Fade<Uuid>,
    pub halo: Fade<Uuid>,
}

impl CloudLayer {
    /// A sky's own cloud layer.
    pub fn of(sky: &SkyFrame) -> CloudLayer {
        CloudLayer {
            id: sky.cloud_id,
            offset1: sky.cloud_pos_density1.truncate(),
            offset2: sky.cloud_pos_density2.truncate(),
            scale: sky.cloud_scale,
            variance: sky.cloud_variance,
            rate: sky.cloud_scroll_rate,
        }
    }

    /// How far apart two layers are, to tell which layer of the last frame
    /// a layer whose values moved (a day cycle beneath) is: another texture
    /// is farther than any value.
    fn distance(&self, o: &CloudLayer) -> f32 {
        let d = (self.offset1 - o.offset1).abs().element_sum()
            + (self.offset2 - o.offset2).abs().element_sum()
            + (self.scale - o.scale).abs()
            + (self.variance - o.variance).abs()
            + (self.rate - o.rate).abs().element_sum();
        let d = if d.is_finite() { d } else { 1e6 };
        if self.id == o.id { d } else { d + 1e7 }
    }
}

impl SkyFade {
    /// A sky's own textures, still.
    pub fn of(sky: &SkyFrame) -> SkyFade {
        SkyFade {
            cloud: Fade::still(CloudLayer::of(sky)),
            sun: Fade::still(sky.sun_id),
            moon: Fade::still(sky.moon_id),
            rainbow: Fade::still(sky.rainbow_id),
            halo: Fade::still(sky.halo_id),
        }
    }

    pub fn fading(&self) -> bool {
        self.cloud.fading() || self.sun.fading() || self.moon.fading() || self.rainbow.fading() || self.halo.fading()
    }

    pub fn towards(&self, to: &SkyFade, t: f32) -> SkyFade {
        self.join(to, [Share::at(t); 5])
    }

    /// [`Fade::join`] for each element, with its own share (cloud, sun,
    /// moon, rainbow, halo).
    pub fn join(&self, to: &SkyFade, shares: [Share; 5]) -> SkyFade {
        SkyFade {
            cloud: self.cloud.join(&to.cloud, shares[0]),
            sun: self.sun.join(&to.sun, shares[1]),
            moon: self.moon.join(&to.moon, shares[2]),
            rainbow: self.rainbow.join(&to.rainbow, shares[3]),
            halo: self.halo.join(&to.halo, shares[4]),
        }
    }
}

/// The water normal map as it shows.
pub type WaterFade = Fade<WaterLayer>;

impl WaterLayer {
    /// A water's own normal map, waves and scale.
    pub fn of(water: &WaterFrame) -> WaterLayer {
        WaterLayer {
            id: water.normal_map,
            wave1: water.wave1,
            wave2: water.wave2,
            scale: water.normal_scale,
        }
    }
}

/// A frame's cross-fade and its own texture fields when it was made (its
/// textures still): the fade holds while the frame keeps them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stamped<F> {
    pub fade: F,
    pub still: F,
}

impl SkyFrame {
    /// What the sky's texture elements show: the cross-fade it was blended
    /// with while it keeps the fields it was blended to, else its own
    /// textures, still.
    pub fn shown_textures(&self) -> SkyFade {
        let still = SkyFade::of(self);
        match self.texture_fade {
            Some(s) if s.still == still => s.fade,
            _ => still,
        }
    }

    /// An environment change's blend from this sky to `b` at `t` (the
    /// selector's fade): `lerp`, whose own fields stay as they are, with the
    /// cross-fade of the texture elements at `t`. A day cycle's keyframes
    /// blend with `lerp` alone and never fade.
    pub fn fade_to(&self, b: &SkyFrame, t: f32) -> SkyFrame {
        self.fade_to_with(b, t, [Share::at(t); 5])
    }

    /// [`Self::fade_to`] with each element's own share ([`TextureClock`]):
    /// the values at `t` (from 1 on, `b`'s own), the textures at their
    /// shares.
    pub fn fade_to_with(&self, b: &SkyFrame, t: f32, shares: [Share; 5]) -> SkyFrame {
        let base = if t >= 1.0 { *b } else { self.lerp(b, t) };
        base.with_texture_fade(self, b, shares)
    }

    /// The textures of a blend of `a` and `b`: the fade from `a`'s to `b`'s
    /// at the shares, kept with this blended frame's own fields; none while
    /// it shows the frame's own textures.
    fn with_texture_fade(mut self, a: &SkyFrame, b: &SkyFrame, shares: [Share; 5]) -> SkyFrame {
        self.texture_fade = None;
        // (the usual frame: the same textures on both sides, no fade)
        if a.texture_fade.is_none() && b.texture_fade.is_none() && SkyFade::of(a) == SkyFade::of(b) {
            return self;
        }
        let fade = a.shown_textures().join(&b.shown_textures(), shares);
        let still = SkyFade::of(&self);
        if fade != still {
            self.texture_fade = Some(Stamped { fade, still });
        }
        self
    }
}

impl WaterFrame {
    /// What the water's normal map shows (see [`SkyFrame::shown_textures`]).
    pub fn shown_textures(&self) -> WaterFade {
        let still = Fade::still(WaterLayer::of(self));
        match self.texture_fade {
            Some(s) if s.still == still => s.fade,
            _ => still,
        }
    }

    /// An environment change's blend (see [`SkyFrame::fade_to`]): `lerp`
    /// with the cross-fade of the normal map.
    pub fn fade_to(&self, b: &WaterFrame, t: f32) -> WaterFrame {
        self.fade_to_with(b, t, Share::at(t))
    }

    /// [`Self::fade_to`] with the normal map's own share.
    pub fn fade_to_with(&self, b: &WaterFrame, t: f32, share: Share) -> WaterFrame {
        let base = if t >= 1.0 { *b } else { self.lerp(b, t) };
        base.with_texture_fade(self, b, share)
    }

    /// The normal map of a blend of `a` and `b` (see [`SkyFrame::fade_to`]).
    fn with_texture_fade(mut self, a: &WaterFrame, b: &WaterFrame, share: Share) -> WaterFrame {
        self.texture_fade = None;
        if a.texture_fade.is_none() && b.texture_fade.is_none() && WaterLayer::of(a) == WaterLayer::of(b) {
            return self;
        }
        let fade = a.shown_textures().join(&b.shown_textures(), share);
        let still = Fade::still(WaterLayer::of(&self));
        if fade != still {
            self.texture_fade = Some(Stamped { fade, still });
        }
        self
    }
}

/// How long after an environment change's time a texture it waits for may
/// still come in before the change gives up and switches it, as before (a
/// missing asset must not hold the old one for ever).
pub const TEXTURE_WAIT_MAX: f64 = 30.0;

/// The texture shares of one environment change (the selector's fade), for
/// the six elements (cloud, sun, moon, rainbow, halo, water). An element
/// whose incoming texture is in follows the change's progress `t` exactly.
/// One still loading keeps share 0; once it is in, its share rises at twice
/// the change's pace (2 × elapsed / duration) until it meets `t`, then
/// follows it, and past the change's time on to 1: a late texture fades in
/// over at most half the change's duration after it came, never at once.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextureClock {
    shares: [f32; 6],
    last: f64,
    /// When the change's own time ran out.
    over: Option<f64>,
}

impl TextureClock {
    pub fn new(now: f64) -> TextureClock {
        TextureClock {
            shares: [0.0; 6],
            last: now,
            over: None,
        }
    }

    /// This frame's shares: `t` the change's progress (1 once its time is
    /// over), `duration` its length (s), `waiting` the elements whose
    /// incoming texture is not in yet ([`waiting`]).
    pub fn advance(&mut self, now: f64, t: f32, duration: f32, waiting: [bool; 6]) -> [Share; 6] {
        let t = unit(t);
        let dt = (now - self.last).max(0.0);
        self.last = now;
        if t >= 1.0 && self.over.is_none() {
            self.over = Some(now);
        }
        let gave_up = self.over.is_some_and(|o| now - o > TEXTURE_WAIT_MAX);
        let step = if duration > 0.0 {
            (2.0 * dt / f64::from(duration)) as f32
        } else {
            1.0
        };
        let mut out = [Share::at(0.0); 6];
        for (i, s) in self.shares.iter_mut().enumerate() {
            let hold = waiting[i] && !gave_up;
            if gave_up {
                *s = 1.0;
            } else if !hold {
                *s = (*s + step).min(t);
            }
            out[i] = Share { t: *s, hold };
        }
        out
    }

    /// Every texture is in and at its full share: the change is over.
    pub fn done(&self) -> bool {
        self.shares.iter().all(|s| *s >= 1.0)
    }
}

/// The elements (cloud, sun, moon, rainbow, halo, water) of a change from
/// `from` to `to` whose incoming texture is not in yet (`ready`: loaded, or
/// nil, the element's stand-in) and can wait for it ([`Fade::can_wait`]).
/// A rainbow or a halo the target does not show is never waited for (its
/// texture is not asked for).
pub fn waiting(from: (&SkyFrame, &WaterFrame), to: (&SkyFrame, &WaterFrame), ready: impl Fn(Uuid) -> bool) -> [bool; 6] {
    let (a, b) = (from.0.shown_textures(), to.0.shown_textures());
    let ids = |f: &Fade<Uuid>, g: &Fade<Uuid>, shown: bool| shown && f.can_wait(g) && !ready(g.settled().first);
    let (wa, wb) = (from.1.shown_textures(), to.1.shown_textures());
    [
        a.cloud.can_wait(&b.cloud) && !ready(b.cloud.settled().first.id),
        ids(&a.sun, &b.sun, true),
        ids(&a.moon, &b.moon, true),
        ids(&a.rainbow, &b.rainbow, to.0.moisture_level > 0.0),
        ids(&a.halo, &b.halo, to.0.ice_level > 0.0),
        wa.can_wait(&wb) && !ready(wb.settled().first.id),
    ]
}

/// The scroll positions of the cloud layers shown (LLEnvironment's
/// mCloudScrollDelta, rate / 100 per second, one for each layer): a layer
/// advances at its own scroll rate. Frame to frame a layer keeps its
/// position: the same layer as the last frame's (a cross-fade's sides do
/// not change), else the nearest of those left (a still sky under a day
/// cycle, whose lerped values move a little each frame: exactly the single
/// accumulated scroll of before), else, a layer coming in, the position of
/// the one it fades from (both then start one over the other). So the
/// layer that remains when a fade ends goes on from where it was, and a
/// layer that takes a slot over in a change begun over another takes the
/// place of the one it replaces.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CloudScroll {
    held: [Option<(CloudLayer, Vec2)>; 2],
}

impl CloudScroll {
    /// Advances the layers of `shown` by `dt` seconds (0: held still, the
    /// photo's light); their positions, the first's then the second's (the
    /// first's again while one layer shows).
    pub fn advance(&mut self, shown: &Fade<CloudLayer>, dt: f32) -> [Vec2; 2] {
        let n = if shown.two() { 2 } else { 1 };
        let layers = &[shown.first, shown.next][..n];
        let held = self.held;
        let mut used = [false; 2];
        // the same layer as the last frame's
        let mut pos: [Option<Vec2>; 2] = [None; 2];
        for (i, layer) in layers.iter().enumerate() {
            if let Some(k) = (0..2).find(|&k| !used[k] && held[k].is_some_and(|(l, _)| l == *layer)) {
                used[k] = true;
                pos[i] = held[k].map(|(_, p)| p);
            }
        }
        // a layer whose values moved: the nearest one left
        for (i, layer) in layers.iter().enumerate() {
            if pos[i].is_some() {
                continue;
            }
            let d = |k: usize| held[k].map_or(f32::INFINITY, |(l, _)| l.distance(layer));
            let left = (0..2).filter(|&k| !used[k] && held[k].is_some());
            if let Some(k) = left.min_by(|&x, &y| d(x).total_cmp(&d(y))) {
                used[k] = true;
                pos[i] = held[k].map(|(_, p)| p);
            }
        }
        // a layer coming in: on the one it fades from
        let start = [pos[0].or(pos[1]), pos[1].or(pos[0])];
        let mut out = [Vec2::ZERO; 2];
        self.held = [None; 2];
        for (i, layer) in layers.iter().enumerate() {
            let mut p = start[i].unwrap_or(Vec2::ZERO) + layer.rate * dt / 100.0;
            if !p.is_finite() {
                p = Vec2::ZERO;
            }
            p = Vec2::new(p.x.rem_euclid(1024.0), p.y.rem_euclid(1024.0));
            self.held[i] = Some((*layer, p));
            out[i] = p;
        }
        if n == 1 {
            out[1] = out[0];
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3Swizzles;

    const A: Uuid = Uuid::from_u128(0xa);
    const B: Uuid = Uuid::from_u128(0xb);
    const C: Uuid = Uuid::from_u128(0xc);

    fn near(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn equal_textures_do_not_fade() {
        for t in [0.0, 0.3, 0.5, 1.0] {
            let f = Fade::still(A).towards(&Fade::still(A), t);
            assert_eq!(f, Fade::still(A), "{t}");
            assert!(!f.fading());
        }
        // one texture on both sides of a fade is still too
        assert!(
            !Fade {
                first: A,
                next: A,
                blend: 0.4
            }
            .fading()
        );
    }

    #[test]
    fn a_simple_fade_is_the_ratio_and_ends_on_each_side() {
        for t in [0.1f32, 0.5, 0.7, 0.99] {
            let f = Fade::still(A).towards(&Fade::still(B), t);
            assert_eq!((f.first, f.next), (A, B), "stable order at {t}");
            assert!(near(f.blend, t), "{t}: {}", f.blend);
        }
        assert_eq!(Fade::still(A).towards(&Fade::still(B), 0.0), Fade::still(A));
        assert_eq!(Fade::still(A).towards(&Fade::still(B), 1.0), Fade::still(B));
        // out of range or not a number: the nearest end, the start
        assert_eq!(Fade::still(A).towards(&Fade::still(B), -2.0), Fade::still(A));
        assert_eq!(Fade::still(A).towards(&Fade::still(B), 3.0), Fade::still(B));
        assert_eq!(Fade::still(A).towards(&Fade::still(B), f32::NAN), Fade::still(A));
    }

    #[test]
    fn a_nested_fade_keeps_the_two_heaviest_renormalised() {
        // A 0.3, B 0.2, C 0.5: B goes, A keeps the first slot
        let f = Fade {
            first: A,
            next: B,
            blend: 0.4,
        }
        .towards(&Fade::still(C), 0.5);
        assert_eq!((f.first, f.next), (A, C));
        assert!(near(f.blend, 0.5 / 0.8), "{}", f.blend);
        // two fades into each other: equal layers merge
        let f = Fade {
            first: A,
            next: B,
            blend: 0.5,
        }
        .towards(
            &Fade {
                first: B,
                next: C,
                blend: 0.5,
            },
            0.5,
        );
        // A 0.25, B 0.25 + 0.25, C 0.25: B, then A (the earlier of the ties)
        assert_eq!((f.first, f.next), (A, B));
        assert!(near(f.blend, 0.5 / 0.75), "{}", f.blend);
        let f = Fade {
            first: A,
            next: B,
            blend: 0.5,
        }
        .towards(&Fade::still(B), 0.5);
        assert_eq!((f.first, f.next), (A, B));
        assert!(near(f.blend, 0.75), "{}", f.blend);
    }

    #[test]
    fn a_fade_begun_again_half_way_starts_where_it_was() {
        let shown = Fade {
            first: A,
            next: B,
            blend: 0.3,
        };
        assert_eq!(shown.towards(&Fade::still(C), 0.0), shown);
        let soon = shown.towards(&Fade::still(C), 0.05);
        assert_eq!((soon.first, soon.next), (A, B));
        assert!(near(soon.blend, 0.3), "renormalised: {}", soon.blend);
        // C takes B's slot once it outweighs it; A never moves
        let later = shown.towards(&Fade::still(C), 0.4);
        assert_eq!((later.first, later.next), (A, C));
        assert!(near(later.blend, 0.4 / (0.42 + 0.4)), "{}", later.blend);
        assert_eq!(shown.towards(&Fade::still(C), 1.0), Fade::still(C));
        // back to the first: B fades out of its slot
        let back = shown.towards(&Fade::still(A), 0.5);
        assert_eq!((back.first, back.next), (A, B));
        assert!(near(back.blend, 0.15), "{}", back.blend);
    }

    fn sky(cloud: Uuid, x: f32) -> SkyFrame {
        SkyFrame {
            cloud_id: cloud,
            cloud_pos_density1: Vec3::new(x, 0.2, 1.0),
            cloud_pos_density2: Vec3::new(x * 2.0, 0.4, 0.5),
            moon_id: B,
            ..SkyFrame::default()
        }
    }

    #[test]
    fn the_sky_fades_each_cloud_texture_where_its_own_offsets_put_it() {
        let (a, b) = (sky(A, 0.1), sky(C, 0.6));
        let mid = a.fade_to(&b, 0.3);
        // the frame's own fields as before: the lerped offsets, the switch half way
        assert_eq!(mid.cloud_id, A);
        assert_eq!(a.fade_to(&b, 0.7).cloud_id, C);
        assert!(near(mid.cloud_pos_density1.x, 0.25));
        let shown = mid.shown_textures();
        assert_eq!((shown.cloud.first.id, shown.cloud.next.id), (A, C));
        assert_eq!(
            (shown.cloud.first.offset1, shown.cloud.next.offset1),
            (Vec2::new(0.1, 0.2), Vec2::new(0.6, 0.2))
        );
        assert_eq!(
            (shown.cloud.first.offset2, shown.cloud.next.offset2),
            (Vec2::new(0.2, 0.4), Vec2::new(1.2, 0.4))
        );
        assert!(near(shown.cloud.blend, 0.3));
        // the elements that do not change do not fade
        assert!(!shown.moon.fading() && !shown.sun.fading() && !shown.halo.fading());
        // the same texture moved: it fades too, from one place to the other
        let moved = sky(A, 0.1).fade_to(&sky(A, 0.6), 0.5).shown_textures();
        assert!(moved.cloud.fading());
        assert_eq!(moved.cloud.first.offset1.x, 0.1);
        // nothing changes: no fade carried at all
        assert!(a.fade_to(&sky(A, 0.1), 0.5).texture_fade.is_none());
        let f = SkyFrame::default();
        assert!(f.fade_to(&f, 0.4).texture_fade.is_none());
        // the plain blend (a day cycle's) never fades
        assert!(a.lerp(&b, 0.3).texture_fade.is_none());
    }

    /// The same texture at the same offsets, one value apart that moves or
    /// reshapes its pattern (a Midday and a Midnight preset typically
    /// differ in just these): two layers, each with its own value, the
    /// share linear in the progress; identical skies carry nothing.
    #[test]
    fn a_change_of_scroll_scale_or_variance_alone_cross_fades_the_clouds() {
        let base = sky(A, 0.1);
        let changes = [
            SkyFrame {
                cloud_scroll_rate: Vec2::new(-1.5, 0.7),
                ..base
            },
            SkyFrame { cloud_scale: 1.3, ..base },
            SkyFrame {
                cloud_variance: 0.8,
                ..base
            },
        ];
        for (n, other) in changes.iter().enumerate() {
            for t in [0.1f32, 0.25, 0.5, 0.75, 0.9] {
                let c = base.fade_to(other, t).shown_textures().cloud;
                assert!(c.fading(), "{n} at {t}");
                assert_eq!((c.first, c.next), (CloudLayer::of(&base), CloudLayer::of(other)), "{n} at {t}");
                assert!(near(c.blend, t), "{n} at {t}: {}", c.blend);
            }
            // the ends: one layer, the side's own
            assert_eq!(base.fade_to(other, 0.0).shown_textures().cloud, Fade::still(CloudLayer::of(&base)));
            assert_eq!(base.fade_to(other, 1.0).shown_textures().cloud, Fade::still(CloudLayer::of(other)));
            // the plain blend (a day cycle's) still lerps them, no fade
            let day = base.lerp(other, 0.5);
            assert!(day.texture_fade.is_none());
            assert_eq!(day.shown_textures().cloud, Fade::still(CloudLayer::of(&day)));
        }
        // identical skies: nothing carried, one layer
        let same = base.fade_to(&base, 0.4);
        assert!(same.texture_fade.is_none());
        assert!(!same.shown_textures().cloud.two());
    }

    /// Each layer scrolls at its own rate; the one that remains goes on
    /// from where it was when the fade ends: no jump at the start (the
    /// incoming layer starts on the outgoing one) nor at the end.
    #[test]
    fn each_cloud_layer_scrolls_at_its_own_rate_and_the_last_one_goes_on() {
        let slow = CloudLayer::of(&SkyFrame {
            cloud_scroll_rate: Vec2::new(10.0, 0.0),
            ..sky(A, 0.1)
        });
        let fast = CloudLayer {
            rate: Vec2::new(30.0, 20.0),
            ..slow
        };
        let mut s = CloudScroll::default();
        let dt = 0.5;
        // the still sky before the change
        let mut p = [Vec2::ZERO; 2];
        for _ in 0..4 {
            p = s.advance(&Fade::still(slow), dt);
        }
        assert!((p[0] - Vec2::new(0.2, 0.0)).length() < 1e-5, "{p:?}");
        let before = p[0];
        // the change: both layers, the incoming one on the outgoing one
        let mut fade = Fade {
            first: slow,
            next: fast,
            blend: 0.0,
        };
        let mut last = before;
        let mut next = Vec2::ZERO;
        for k in 1..=10 {
            fade.blend = k as f32 / 11.0;
            let q = s.advance(&fade, dt);
            assert!((q[0] - last - slow.rate * dt / 100.0).length() < 1e-5, "first at its own rate, {k}");
            if k == 1 {
                assert!(
                    (q[1] - (before + fast.rate * dt / 100.0)).length() < 1e-5,
                    "starts on the first: {q:?}"
                );
            } else {
                assert!(
                    (q[1] - next - fast.rate * dt / 100.0).length() < 1e-5,
                    "second at its own rate, {k}"
                );
            }
            (last, next) = (q[0], q[1]);
        }
        // the end: the remaining layer goes on from the second's place
        let end = s.advance(&Fade::still(fast), dt);
        assert!((end[0] - (next + fast.rate * dt / 100.0)).length() < 1e-5, "{end:?} after {next:?}");
        // and a still sky under a day cycle (its values move a little each
        // frame) keeps one accumulated scroll, as before
        let moved = CloudLayer {
            rate: Vec2::new(29.0, 20.0),
            ..fast
        };
        let after = s.advance(&Fade::still(moved), dt);
        assert!((after[0] - (end[0] + moved.rate * dt / 100.0)).length() < 1e-5);
        // the photo's light holds it still
        assert_eq!(s.advance(&Fade::still(moved), 0.0), [after[0]; 2]);
    }

    /// A change begun over a running one: the layers that stay keep their
    /// positions whatever their slot, and a layer that takes a slot over
    /// takes the place of the one it replaces.
    #[test]
    fn an_interrupted_change_keeps_every_shown_layer_where_it_was() {
        let l = |id: Uuid, rate: f32| CloudLayer {
            id,
            offset1: Vec2::ZERO,
            offset2: Vec2::ZERO,
            scale: 0.42,
            variance: 0.0,
            rate: Vec2::new(rate, 0.0),
        };
        let (a, b, c) = (l(A, 10.0), l(A, 40.0), l(A, -5.0));
        let mut s = CloudScroll::default();
        s.advance(&Fade::still(a), 1.0);
        let ab = s.advance(
            &Fade {
                first: a,
                next: b,
                blend: 0.3,
            },
            1.0,
        );
        // C takes B's slot: A stays, C goes on from B's place
        let ac = s.advance(
            &Fade {
                first: a,
                next: c,
                blend: 0.5,
            },
            1.0,
        );
        assert!((ac[0] - (ab[0] + a.rate / 100.0)).length() < 1e-5);
        assert!((ac[1] - (ab[1] + c.rate / 100.0)).length() < 1e-5);
        // A leaves and C moves to the first slot: it keeps its place
        let cb = s.advance(
            &Fade {
                first: c,
                next: b,
                blend: 0.5,
            },
            1.0,
        );
        assert!((cb[0] - (ac[1] + c.rate / 100.0)).length() < 1e-5, "{cb:?} {ac:?}");
        // the end: C alone, from its place
        let end = s.advance(&Fade::still(c), 1.0);
        assert!((end[0] - (cb[0] + c.rate / 100.0)).length() < 1e-5);
        // the slot rule itself never hands a shown texture to the other slot
        let shown = Fade {
            first: a,
            next: b,
            blend: 0.4,
        };
        for t in [0.0f32, 0.1, 0.3, 0.6, 0.9] {
            let f = shown.towards(&Fade::still(c), t);
            assert_eq!(f.first, a, "{t}");
        }
    }

    #[test]
    fn a_fade_into_a_fade_keeps_the_layers_it_shows() {
        // what is shown part way through a change, faded to another sky
        let shown = sky(A, 0.1).fade_to(&sky(B, 0.3), 0.4);
        let f = shown.fade_to(&sky(C, 0.9), 0.5).shown_textures().cloud;
        // A 0.3, B 0.2, C 0.5
        assert_eq!((f.first.id, f.next.id), (A, C));
        assert_eq!((f.first.offset1.x, f.next.offset1.x), (0.1, 0.9));
        assert!(near(f.blend, 0.625), "{}", f.blend);
    }

    #[test]
    fn a_frame_changed_after_its_blend_shows_its_own_textures() {
        let mid = sky(A, 0.1).fade_to(&sky(C, 0.6), 0.3);
        assert!(mid.shown_textures().cloud.fading());
        // another texture (RLVa's cloudtexture, an experience's push)
        let other = SkyFrame { cloud_id: B, ..mid };
        assert_eq!(other.shown_textures(), SkyFade::of(&other));
        // moved (cloud density in RLVa, an editor)
        let moved = SkyFrame {
            cloud_pos_density1: Vec3::new(0.9, 0.2, 1.0),
            ..mid
        };
        assert!(!moved.shown_textures().cloud.fading());
        // what is not a layer's leaves the fade as it is: the densities too
        let lit = SkyFrame {
            gamma: 2.0,
            cloud_pos_density1: mid.cloud_pos_density1.xy().extend(0.3),
            ..mid
        };
        assert_eq!(lit.shown_textures(), mid.shown_textures());
    }

    #[test]
    fn the_water_fades_each_normal_map_with_its_own_waves_and_scale() {
        let a = WaterFrame {
            normal_map: A,
            wave1: Vec2::new(1.0, 0.0),
            normal_scale: Vec3::splat(2.0),
            ..WaterFrame::default()
        };
        let b = WaterFrame {
            normal_map: B,
            wave1: Vec2::new(-0.5, 0.5),
            normal_scale: Vec3::splat(4.0),
            ..WaterFrame::default()
        };
        let mid = a.fade_to(&b, 0.25);
        assert_eq!(mid.normal_map, A);
        let f = mid.shown_textures();
        assert_eq!((f.first, f.next), (WaterLayer::of(&a), WaterLayer::of(&b)));
        assert!(near(f.blend, 0.25));
        // the waves alone changed: they fade too (no slide)
        let turned = WaterFrame {
            wave1: Vec2::new(0.0, 1.0),
            ..a
        };
        assert!(a.fade_to(&turned, 0.5).shown_textures().fading());
        // validated keeps the fade of a frame within the ranges
        assert_eq!(mid.validated().shown_textures(), f);
        // changed after its blend: its own map
        let other = WaterFrame { normal_map: C, ..mid };
        assert!(!other.shown_textures().fading());
        assert!(a.fade_to(&a, 0.5).texture_fade.is_none());
        // the plain blend (a day cycle's) never fades
        assert!(a.lerp(&b, 0.25).texture_fade.is_none());
    }

    /// The same normal map, only a wave's speed, a wave's direction or the
    /// normal scale apart: two layers, each with its own waves and scale
    /// (the shaders place a layer at `wave × time`, so its own waves are its
    /// own place), the share linear, and the last one the target's own; the
    /// amounts (fog, fresnel, blur multiplier) blend and fade nothing.
    #[test]
    fn a_change_of_waves_or_normal_scale_alone_cross_fades_the_water() {
        let base = WaterFrame {
            normal_map: A,
            wave1: Vec2::new(1.0, 0.5),
            wave2: Vec2::new(-0.3, 0.2),
            normal_scale: Vec3::new(2.0, 2.0, 1.0),
            ..WaterFrame::default()
        };
        let changes = [
            WaterFrame {
                wave1: base.wave1 * 3.0,
                ..base
            },
            WaterFrame {
                wave2: Vec2::new(0.2, 0.3),
                ..base
            },
            WaterFrame {
                normal_scale: Vec3::splat(6.0),
                ..base
            },
        ];
        for (n, other) in changes.iter().enumerate() {
            for t in [0.1f32, 0.5, 0.9] {
                let f = base.fade_to(other, t).shown_textures();
                assert_eq!((f.first, f.next), (WaterLayer::of(&base), WaterLayer::of(other)), "{n} at {t}");
                assert!(near(f.blend, t), "{n} at {t}");
            }
            assert_eq!(base.fade_to(other, 1.0).shown_textures(), Fade::still(WaterLayer::of(other)));
        }
        let amounts = WaterFrame {
            fog_density: 9.0,
            fresnel_scale: 0.9,
            blur_multiplier: 0.3,
            ..base
        };
        assert!(base.fade_to(&amounts, 0.5).texture_fade.is_none());
        assert!(base.fade_to(&base, 0.5).texture_fade.is_none());
    }

    /// The late texture's rule: an element whose texture is in follows the
    /// change exactly; one still loading keeps share 0, then rises from 0
    /// at twice the change's pace until it meets the change, and past the
    /// change's time on to 1, never at once; one that never comes gives up
    /// 30 s after the change's time.
    #[test]
    fn a_late_texture_joins_without_a_pop() {
        let d = 10.0f32;
        let mut clock = TextureClock::new(0.0);
        let wait = |cloud: bool| [cloud, false, false, false, false, false];
        let mut shares = Vec::new();
        for k in 0..=40 {
            let now = f64::from(k) * 0.5;
            let t = (now as f32 / d).min(1.0);
            // the cloud texture comes in at 6 s (t 0.6)
            let s = clock.advance(now, t, d, wait(now < 6.0));
            assert!((s[1].t - t).abs() < 1e-6, "the others follow the change at {now}");
            assert_eq!(s[0].hold, now < 6.0);
            shares.push((now, t, s[0].t));
        }
        for w in shares.windows(2) {
            let ((_, _, a), (now, t, b)) = (w[0], w[1]);
            assert!(b >= a && b <= t.max(a) + 1e-6, "rises, never past the change: {now} {a} {b}");
            assert!(b - a <= 2.0 * 0.5 / d + 1e-6, "at most twice the change's pace: {now}");
        }
        let at = |n: f64| shares.iter().find(|s| s.0 == n).map_or(-1.0, |s| s.2);
        assert_eq!(at(5.5), 0.0, "not in: no share");
        assert!((at(6.0) - 0.1).abs() < 1e-5, "from 0 at its arrival: {}", at(6.0));
        assert!((at(7.0) - 0.3).abs() < 1e-5, "{}", at(7.0));
        // past the change's time it goes on, and ends at 1
        assert!(at(10.0) < 1.0 && at(10.0) > 0.7);
        assert_eq!(at(12.0), 1.0);
        assert!(clock.done());
        // in from the start: exactly the change's progress
        let mut on_time = TextureClock::new(0.0);
        for k in 0..=10 {
            let t = k as f32 / 10.0;
            let s = on_time.advance(f64::from(t * d), t, d, wait(false));
            assert!((s[0].t - t).abs() < 1e-6, "{t}");
        }
        assert!(on_time.done());
        // never in: held, then switched 30 s after the change's time
        let mut never = TextureClock::new(0.0);
        never.advance(10.0, 1.0, d, wait(true));
        assert!(!never.done());
        assert!(never.advance(39.0, 1.0, d, wait(true))[0].hold);
        assert!(never.advance(39.0, 1.0, d, wait(true))[0].t < 1.0);
        let s = never.advance(41.0, 1.0, d, wait(true));
        assert!(s[0].t >= 1.0 && !s[0].hold && never.done());
    }

    /// While a texture loads, its layer waits in the second slot at share 0
    /// (asked for, nothing drawn of it), the first one showing alone at its
    /// own place; then it comes in from 0; and it still fades in when the
    /// change's time is over (the frame's values then the target's).
    #[test]
    fn a_layer_waits_in_its_slot_for_its_texture() {
        let (a, b) = (sky(A, 0.1), sky(C, 0.6));
        let held = |t: f32, s: f32| [Share { t: s, hold: true }, Share::at(t), Share::at(t), Share::at(t), Share::at(t)];
        let waiting = a.fade_to_with(&b, 0.4, held(0.4, 0.0));
        let c = waiting.shown_textures().cloud;
        assert_eq!((c.first, c.next, c.blend), (CloudLayer::of(&a), CloudLayer::of(&b), 0.0));
        assert!(c.two() && !c.fading());
        assert!(near(waiting.cloud_pos_density1.x, 0.3), "the values go on");
        let joining = a.fade_to_with(&b, 0.6, held(0.6, 0.1));
        assert!(near(joining.shown_textures().cloud.blend, 0.1));
        // over its time: the target's values, the textures still fading
        let late = a.fade_to_with(&b, 1.0, held(1.0, 0.5));
        assert_eq!(late.cloud_pos_density1, b.cloud_pos_density1);
        assert_eq!(late.shown_textures().cloud.next, CloudLayer::of(&b));
        assert!(near(late.shown_textures().cloud.blend, 0.5));
        assert_eq!(a.fade_to_with(&b, 1.0, held(1.0, 1.0)).shown_textures(), SkyFade::of(&b));
        // which elements wait: only one whose incoming texture is not in
        let wa = WaterFrame::default();
        let ready = |id: Uuid| id != C;
        assert_eq!(waiting_of(&a, &b, &wa, ready), [true, false, false, false, false, false]);
        assert_eq!(waiting_of(&a, &b, &wa, |_| true), [false; 6]);
        // the same texture moved or rescaled: it is in, nothing waits
        let scaled = SkyFrame { cloud_scale: 1.2, ..a };
        assert_eq!(waiting_of(&a, &scaled, &wa, |id| id == A), [false; 6]);
    }

    fn waiting_of(a: &SkyFrame, b: &SkyFrame, w: &WaterFrame, ready: impl Fn(Uuid) -> bool) -> [bool; 6] {
        // (with C only on the cloud: the other elements are equal)
        let b = SkyFrame { moon_id: a.moon_id, ..*b };
        waiting((a, w), (&b, w), |id| id.is_nil() || ready(id))
    }

    /// Between the keyframes of a day cycle nothing fades: the frames are
    /// the plain blend's, as before the cross-fade existed (the ids switch
    /// half way, the offsets, waves and normal scale are lerped), with no
    /// fade carried, one texture per element and every share 0.
    #[test]
    fn a_day_cycle_blends_its_keyframes_without_a_fade() {
        use super::super::eep::DayCycle;
        use super::super::env::Environment;
        let first = SkyFrame {
            sun_id: A,
            rainbow_id: A,
            halo_id: A,
            moisture_level: 0.5,
            ice_level: 0.5,
            cloud_scale: 0.3,
            cloud_variance: 0.2,
            cloud_scroll_rate: Vec2::new(0.5, 0.1),
            ..sky(A, 0.1)
        };
        let second = SkyFrame {
            sun_id: C,
            moon_id: C,
            rainbow_id: C,
            halo_id: C,
            cloud_scale: 0.9,
            cloud_variance: 0.6,
            cloud_scroll_rate: Vec2::new(-2.0, 1.0),
            ..sky(C, 0.6)
        };
        let wa = WaterFrame {
            normal_map: A,
            wave1: Vec2::new(1.0, 0.0),
            wave2: Vec2::new(0.0, 1.0),
            normal_scale: Vec3::splat(2.0),
            ..WaterFrame::default()
        };
        let wb = WaterFrame {
            normal_map: B,
            wave1: Vec2::new(-0.5, 0.5),
            wave2: Vec2::new(0.3, -0.2),
            normal_scale: Vec3::splat(4.0),
            ..WaterFrame::default()
        };
        let day = DayCycle {
            sky_tracks: [vec![(0.0, first), (0.5, second)], Vec::new(), Vec::new(), Vec::new()],
            water: vec![(0.0, wa), (0.5, wb)],
            ..DayCycle::default()
        };
        // both segments, each on both sides of its half-way switch
        for pos in [0.0f32, 0.1, 0.249, 0.25, 0.251, 0.4, 0.5, 0.6, 0.749, 0.751, 0.99] {
            let (s, w) = (day.sky_at(pos, 0), day.water_at(pos));
            let ((a, b), (c, d), t) = if pos < 0.5 {
                ((first, second), (wa, wb), pos / 0.5)
            } else {
                ((second, first), (wb, wa), (pos - 0.5) / 0.5)
            };
            // the plain blend's frame, every field to the bit
            assert_eq!(format!("{s:?}"), format!("{:?}", a.lerp(&b, t)), "{pos}");
            assert_eq!(w, c.lerp(&d, t), "{pos}");
            // which is the frame of before: no fade, the ids half way,
            // the offsets, scale, variance, rate, waves and scale lerped
            assert!(s.texture_fade.is_none() && w.texture_fade.is_none(), "{pos}");
            let pick = |x: Uuid, y: Uuid| if t < 0.5 { x } else { y };
            assert_eq!(
                [s.cloud_id, s.sun_id, s.moon_id, s.rainbow_id, s.halo_id, w.normal_map],
                [
                    pick(a.cloud_id, b.cloud_id),
                    pick(a.sun_id, b.sun_id),
                    pick(a.moon_id, b.moon_id),
                    pick(a.rainbow_id, b.rainbow_id),
                    pick(a.halo_id, b.halo_id),
                    pick(c.normal_map, d.normal_map),
                ],
                "{pos}"
            );
            assert_eq!(s.cloud_pos_density1, a.cloud_pos_density1.lerp(b.cloud_pos_density1, t));
            assert_eq!(s.cloud_pos_density2, a.cloud_pos_density2.lerp(b.cloud_pos_density2, t));
            assert_eq!(s.cloud_scroll_rate, a.cloud_scroll_rate.lerp(b.cloud_scroll_rate, t));
            assert_eq!(
                (w.wave1, w.wave2, w.normal_scale),
                (
                    c.wave1.lerp(d.wave1, t),
                    c.wave2.lerp(d.wave2, t),
                    c.normal_scale.lerp(d.normal_scale, t)
                )
            );
            // one texture each, and nothing for the renderer to mix
            assert_eq!(s.shown_textures(), SkyFade::of(&s), "{pos}");
            assert_eq!(w.shown_textures(), Fade::still(WaterLayer::of(&w)), "{pos}");
            let env = Environment::from_eep(&s, &w, Vec2::new(0.25, 0.5), 96.0);
            assert!(env.next_textures.iter().all(Uuid::is_nil), "{pos}");
            assert_eq!((env.sky.fade, env.water.fade), Default::default(), "{pos}");
            // the cloud layer drawn is the frame's own, lerped
            assert_eq!((env.sky.cloud_scale, env.sky.cloud_variance), (s.cloud_scale, s.cloud_variance));
            // (an environment change between the same two frames would fade)
            if t > 0.0 {
                assert!(a.fade_to(&b, t).shown_textures().fading(), "{pos}");
                assert!(c.fade_to(&d, t).shown_textures().fading(), "{pos}");
            }
        }
    }
}
