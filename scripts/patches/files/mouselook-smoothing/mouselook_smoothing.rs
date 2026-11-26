// aurora-local-patch: mouselook-smoothing-module
//! Mouselook smoothing (local patch set): a choice of filters between the
//! direction the mouse asks for (the aim: yaw and pitch after the
//! sensitivity, pitch limits applied) and the direction the view looks.
//!
//! The view IS the agent's heading and pitch (`AgentState::yaw` / `pitch`):
//! the body rotation, the head rotation, the look-at and the camera of the
//! AgentUpdate all follow the smoothed view, so the region and other people
//! see what the user sees. Only the aim is held here. Off, or a strength of
//! 0, bypasses everything: the mouse writes the agent's angles as before
//! and [`Smoother::step`] changes nothing.
//!
//! Every filter runs on the real frame time and is frame-rate independent;
//! a frame longer than [`SNAP_DT`] (a pause, a hitch) puts the view on the
//! aim. Yaw is handled across the ±180° wrap; pitch never leaves the agent's
//! limits (`agent::clamp_pitch`). When the aim is held still the view ends
//! exactly on it.

use crate::agent::{AgentState, clamp_pitch};
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::f32::consts::{PI, TAU};

/// The technique chosen in Preferences › Camera › Mouselook.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Technique {
    /// The view turns exactly with the mouse (as without this patch).
    #[default]
    Off,
    /// The mean of the aim over a short window.
    FrameAverage,
    /// A first-order low-pass toward the aim. Its relative is the Legacy
    /// Viewer's fixed-strength MouseSmooth (LLViewerWindow::updateMouseDelta,
    /// indra/newview/llviewerwindow.cpp, originally LGPL 2.1), which smooths
    /// the mouse's delta at a fixed rate and rounds it to whole pixels.
    Exponential,
    /// The One Euro filter: smooth slow motion, direct fast motion.
    OneEuro,
    /// A critically damped spring.
    DampedSpring,
    /// The crosshair moves at once, the view follows it.
    LeadingCursor,
}

impl Technique {
    pub const ALL: [Technique; 6] = [
        Technique::Off,
        Technique::FrameAverage,
        Technique::Exponential,
        Technique::OneEuro,
        Technique::DampedSpring,
        Technique::LeadingCursor,
    ];
}

/// The strength slider's range.
pub const STRENGTH_RANGE: std::ops::RangeInclusive<f32> = 0.0..=1.0;

/// A strength read from a file: finite and within [`STRENGTH_RANGE`].
pub fn sanitize_strength(strength: f32) -> f32 {
    if strength.is_finite() {
        strength.clamp(*STRENGTH_RANGE.start(), *STRENGTH_RANGE.end())
    } else {
        0.0
    }
}

/// A frame at least this long puts the view on the aim (no catch-up after a
/// pause, no explosion of a filter).
pub const SNAP_DT: f32 = 0.5;
/// Anything else that turns the agent by more than this in one frame (a
/// teleport's heading, a seat, a script's or a tool's turn) is taken as a
/// new direction: the filter starts again there instead of swinging.
const EXTERNAL_JUMP: f32 = 1.0;
/// Closer than this to the aim (radians, about 0.0006°) the view is put on it.
const SETTLE: f32 = 1e-5;

/// Frame average: the window is (N - 1) frames at 60 fps, N from 1 to 8.
const AVERAGE_FRAMES: f32 = 7.0;
/// Exponential: the time constant at full strength (s).
const EXPONENTIAL_TAU: f32 = 0.5;
/// One Euro: the minimum cutoff at a strength near 0 and at 1 (Hz).
const ONE_EURO_FAST: f32 = 25.0;
const ONE_EURO_SLOW: f32 = 0.5;
/// One Euro: the speed coefficient (Hz per rad/s of turning).
const ONE_EURO_BETA: f32 = 6.0;
/// One Euro: the cutoff of the speed's own filter (Hz), the paper's value.
const ONE_EURO_SPEED_CUTOFF: f32 = 1.0;
/// Damped spring and leading cursor: the catch-up time at full strength (s).
const SPRING_TIME: f32 = 1.0;
/// Leading cursor: the aim stays within this share of the half screen.
pub const LEAD_LIMIT: f32 = 0.85;

/// The window of the frame average (s).
pub fn average_window(strength: f32) -> f32 {
    AVERAGE_FRAMES * sanitize_strength(strength) / 60.0
}

/// The exponential filter's time constant (s): squared, so the first half
/// of the slider covers the light settings (0.25 → 31 ms, 0.5 → 125 ms).
pub fn exponential_tau(strength: f32) -> f32 {
    let s = sanitize_strength(strength);
    EXPONENTIAL_TAU * s * s
}

/// The One Euro filter's minimum cutoff (Hz), geometric from 25 Hz to 0.5 Hz.
pub fn one_euro_min_cutoff(strength: f32) -> f32 {
    ONE_EURO_FAST * (ONE_EURO_SLOW / ONE_EURO_FAST).powf(sanitize_strength(strength))
}

/// The spring's catch-up time (s), squared like the exponential's.
pub fn spring_time(strength: f32) -> f32 {
    let s = sanitize_strength(strength);
    SPRING_TIME * s * s
}

/// An angle in [-π, π).
pub fn wrap(angle: f32) -> f32 {
    (angle + PI).rem_euclid(TAU) - PI
}

/// A direction as the agent holds it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Angles {
    pub yaw: f32,
    pub pitch: f32,
}

impl Angles {
    pub fn of(agent: &AgentState) -> Angles {
        Angles {
            yaw: agent.yaw,
            pitch: agent.pitch,
        }
    }

    /// The unit vector (yaw about +Z, pitch up), as `Camera::mouselook_aim`.
    pub fn forward(self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
        )
    }

    /// From `from` to here, the yaw the short way round.
    fn delta(self, from: Angles) -> Vec2 {
        Vec2::new(wrap(self.yaw - from.yaw), self.pitch - from.pitch)
    }

    /// Moved by `d`, the yaw wrapped, the pitch within the limits.
    fn offset(self, d: Vec2) -> Angles {
        Angles {
            yaw: wrap(self.yaw + d.x),
            pitch: clamp_pitch(self.pitch + d.y),
        }
    }
}

/// The view's lens, for the leading cursor's screen margin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lens {
    /// Vertical field of view (radians).
    pub fov_y: f32,
    /// Width over height.
    pub aspect: f32,
}

/// Where `aim` is on the screen of a view looking at `view`, in normalised
/// device coordinates (-1 to 1, +y up); None when behind the eye or for an
/// unusable lens.
pub fn ndc(view: Angles, aim: Angles, lens: Lens) -> Option<Vec2> {
    let half = (lens.fov_y * 0.5).tan();
    if !(half.is_finite() && half > 0.0 && lens.aspect.is_finite() && lens.aspect > 0.0) {
        return None;
    }
    let forward = view.forward();
    let right = Vec3::new(view.yaw.sin(), -view.yaw.cos(), 0.0);
    let up = right.cross(forward);
    let a = aim.forward();
    let z = a.dot(forward);
    if z <= 1e-3 {
        return None;
    }
    Some(Vec2::new(a.dot(right) / z / (half * lens.aspect), a.dot(up) / z / half))
}

/// Whether `aim` is within the leading cursor's margin of the screen.
pub fn on_screen(view: Angles, aim: Angles, lens: Lens) -> bool {
    let tan = (lens.fov_y * 0.5).tan();
    if !(tan.is_finite() && tan > 0.0 && lens.aspect.is_finite() && lens.aspect > 0.0) {
        // no usable lens: nothing to keep in view
        return true;
    }
    ndc(view, aim, lens).is_some_and(|p| p.x.abs() <= LEAD_LIMIT && p.y.abs() <= LEAD_LIMIT)
}

/// The view moved toward `aim` just enough for the aim to be within the
/// margin (the view is dragged along by the crosshair at the edge).
fn keep_on_screen(view: Angles, aim: Angles, lens: Lens) -> Angles {
    if on_screen(view, aim, lens) {
        return view;
    }
    let d = aim.delta(view);
    let (mut lo, mut hi) = (0.0_f32, 1.0_f32);
    for _ in 0..24 {
        let mid = 0.5 * (lo + hi);
        if on_screen(view.offset(d * mid), aim, lens) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    if hi >= 1.0 { aim } else { view.offset(d * hi) }
}

/// 1 - e^(-2π fc dt): the exact discrete step of a first-order low-pass of
/// cutoff `fc` for an input held during `dt`.
fn alpha(cutoff: f32, dt: f32) -> f32 {
    1.0 - (-TAU * cutoff * dt).exp()
}

// The filters below take the aim as moving in a straight line during the
// frame, from where it was at the last frame to where it is now (the mouse
// moved during the frame, not at its end), and solve their equation exactly
// over it. A given motion then gives the same view at any frame rate. `x`
// is the view's offset from the aim now, `s` the aim's speed over the frame.

/// A first-order low-pass of time constant `tau`, one frame of `dt`.
fn low_pass(x: f32, s: f32, tau: f32, dt: f32) -> f32 {
    -s * tau + (x + s * dt + s * tau) * (-dt / tau).exp()
}

/// A critically damped spring of natural frequency `omega` (the closed
/// form of the usual "smooth damp"), one frame of `dt`, view velocity `v`;
/// returns the new offset and velocity. It never crosses the aim: a frame
/// that would ends on it, at rest.
fn spring(x: f32, v: f32, s: f32, omega: f32, dt: f32) -> (f32, f32) {
    if x == 0.0 && s == 0.0 {
        return (0.0, 0.0);
    }
    // in the frame of the moving aim the spring rests at -2s/ω
    let rest = -2.0 * s / omega;
    let (z, zv) = (x + s * dt - rest, v - s);
    let e = (-omega * dt).exp();
    let j = (zv + omega * z) * dt;
    let (nx, nv) = ((z + j) * e + rest, (zv - omega * j) * e + s);
    if x != 0.0 && nx != 0.0 && nx.signum() != x.signum() {
        (0.0, 0.0)
    } else {
        (nx, nv)
    }
}

#[derive(Debug, Clone)]
enum Filter {
    /// The aim of the last frames, newest first: each frame's time and the
    /// aim at its start and its end.
    Average(VecDeque<(f32, Angles, Angles)>),
    Exponential,
    OneEuro {
        speed: Vec2,
    },
    /// Damped spring and leading cursor.
    Spring {
        velocity: Vec2,
    },
}

#[derive(Debug, Clone)]
struct State {
    technique: Technique,
    seated: bool,
    aim: Angles,
    /// The aim at the last frame.
    previous: Angles,
    view: Angles,
    filter: Filter,
}

impl State {
    fn new(technique: Technique, at: Angles, seated: bool) -> State {
        let mut state = State {
            technique,
            seated,
            aim: at,
            previous: at,
            view: at,
            filter: Filter::Exponential,
        };
        state.snap();
        state
    }

    /// The view on the aim, the filter at rest there.
    fn snap(&mut self) {
        self.view = self.aim;
        self.previous = self.aim;
        self.filter = match self.technique {
            Technique::FrameAverage => Filter::Average(VecDeque::from([(f32::MAX, self.aim, self.aim)])),
            Technique::OneEuro => Filter::OneEuro { speed: Vec2::ZERO },
            Technique::DampedSpring | Technique::LeadingCursor => Filter::Spring { velocity: Vec2::ZERO },
            Technique::Off | Technique::Exponential => Filter::Exponential,
        };
    }

    /// The view toward the aim, one frame of `dt` seconds.
    fn advance(&mut self, strength: f32, dt: f32, lens: Lens) {
        let (aim, previous, view) = (self.aim, self.previous, self.view);
        if dt <= 0.0 {
            return;
        }
        self.previous = aim;
        let x = view.delta(aim);
        let speed_now = aim.delta(previous) / dt;
        let low = |tau: f32| {
            if tau <= 0.0 {
                aim
            } else {
                aim.offset(Vec2::new(low_pass(x.x, speed_now.x, tau, dt), low_pass(x.y, speed_now.y, tau, dt)))
            }
        };
        let next = match &mut self.filter {
            Filter::Average(history) => average(history, average_window(strength), previous, aim, dt),
            Filter::Exponential => low(exponential_tau(strength)),
            Filter::OneEuro { speed } => {
                // the paper's derivative, low-passed, raises the cutoff
                *speed += (speed_now - *speed) * alpha(ONE_EURO_SPEED_CUTOFF, dt);
                let cutoff = one_euro_min_cutoff(strength) + ONE_EURO_BETA * speed.length();
                low(1.0 / (TAU * cutoff))
            }
            Filter::Spring { velocity } => {
                let time = spring_time(strength);
                if time <= 0.0 {
                    *velocity = Vec2::ZERO;
                    aim
                } else {
                    let omega = 2.0 / time;
                    let (yx, yv) = spring(x.x, velocity.x, speed_now.x, omega, dt);
                    let (px, pv) = spring(x.y, velocity.y, speed_now.y, omega, dt);
                    *velocity = Vec2::new(yv, pv);
                    aim.offset(Vec2::new(yx, px))
                }
            }
        };
        let settled = aim.delta(next).abs().max_element() < SETTLE
            && match &self.filter {
                Filter::Spring { velocity } => velocity.abs().max_element() < SETTLE,
                Filter::OneEuro { speed, .. } => speed.abs().max_element() < SETTLE,
                _ => true,
            };
        if settled {
            self.snap();
            return;
        }
        self.view = next;
        if self.technique == Technique::LeadingCursor {
            self.view = keep_on_screen(self.view, self.aim, lens);
        }
    }
}

/// The mean of the aim over the last `window` seconds (the aim moving in a
/// straight line during each frame). Averaged as offsets from the aim, so a
/// still aim gives the aim exactly.
fn average(history: &mut VecDeque<(f32, Angles, Angles)>, window: f32, previous: Angles, aim: Angles, dt: f32) -> Angles {
    history.push_front((dt, previous, aim));
    let (mut covered, mut sum, mut keep) = (0.0_f32, Vec2::ZERO, 0);
    for &(held, from, to) in history.iter() {
        if covered >= window {
            break;
        }
        let take = held.min(window - covered);
        // the mean over the newest `take` seconds of that frame
        let (end, start) = (to.delta(aim), from.delta(aim));
        let share = if held > 0.0 { take / held } else { 0.0 };
        sum += (end - (end - start) * (0.5 * share)) * take;
        covered += take;
        keep += 1;
    }
    history.truncate(keep.max(1));
    if covered <= 0.0 || sum == Vec2::ZERO {
        aim
    } else {
        aim.offset(sum / covered)
    }
}

/// The smoothing's state, kept by the camera while mouselook is smoothed.
#[derive(Debug, Clone, Default)]
pub struct Smoother {
    state: Option<State>,
}

impl Smoother {
    /// The mouse turns the aim, not the agent: a filter is running.
    pub fn engaged(&self) -> bool {
        self.state.is_some()
    }

    /// The aim and the view, while a filter runs.
    pub fn angles(&self) -> Option<(Angles, Angles)> {
        self.state.as_ref().map(|s| (s.aim, s.view))
    }

    /// The leading cursor's aim and view (the crosshair off the centre).
    pub fn leading(&self) -> Option<(Angles, Angles)> {
        self.state
            .as_ref()
            .filter(|s| s.technique == Technique::LeadingCursor)
            .map(|s| (s.aim, s.view))
    }

    /// The mouse's turn, already scaled by the sensitivity (radians).
    pub fn turn(&mut self, yaw: f32, pitch: f32) {
        if let Some(s) = &mut self.state {
            s.aim = s.aim.offset(Vec2::new(yaw, pitch));
        }
    }

    /// The frame turned under the view (a turning seat): the aim, the view
    /// and the history turn with it, with no smoothing of that turn.
    pub fn shift_frame(&mut self, yaw: f32) {
        let Some(s) = &mut self.state else {
            return;
        };
        let d = Vec2::new(wrap(yaw), 0.0);
        if d.x == 0.0 || !d.x.is_finite() {
            return;
        }
        s.aim = s.aim.offset(d);
        s.previous = s.previous.offset(d);
        s.view = s.view.offset(d);
        if let Filter::Average(history) = &mut s.filter {
            for (_, from, to) in history.iter_mut() {
                *from = from.offset(d);
                *to = to.offset(d);
            }
        }
    }

    /// Forget everything: the next frame starts from the agent's direction.
    pub fn reset(&mut self) {
        self.state = None;
    }

    /// One frame. `active`: mouselook is on and the mouse turns the agent
    /// (no VR, flycam, free camera or Director track). `current`: the
    /// agent's direction now, which is the view of the last frame plus
    /// whatever else turned it since (a controller's stick, through the
    /// agent's update, becomes a turn of the aim and is smoothed too).
    /// Returns the direction to give the agent, or None to leave it as it
    /// is (Off, strength 0, not active, or the first frame).
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        active: bool,
        technique: Technique,
        strength: f32,
        current: Angles,
        seated: bool,
        dt: f32,
        lens: Lens,
    ) -> Option<Angles> {
        if !active {
            self.state = None;
            return None;
        }
        let strength = sanitize_strength(strength);
        if technique == Technique::Off || strength <= 0.0 {
            // switched off while running: the mouse's last turns are kept
            return self.state.take().map(|s| s.aim);
        }
        if !(current.yaw.is_finite() && current.pitch.is_finite()) {
            self.state = None;
            return None;
        }
        let fresh = self.state.as_ref().is_none_or(|s| s.technique != technique || s.seated != seated);
        if fresh {
            self.state = Some(State::new(technique, current, seated));
            return None;
        }
        let s = self.state.as_mut()?;
        let outside = current.delta(s.view);
        if outside.abs().max_element() > EXTERNAL_JUMP {
            *s = State::new(technique, current, seated);
            return None;
        }
        if outside != Vec2::ZERO {
            s.aim = s.aim.offset(outside);
        }
        let dt = if dt.is_finite() { dt.max(0.0) } else { SNAP_DT };
        if dt >= SNAP_DT {
            s.snap();
        } else {
            s.advance(strength, dt, lens);
        }
        Some(s.view)
    }
}

impl super::Camera {
    /// One frame of the mouselook smoothing, after the agent's own update
    /// (its controller turn) and before the camera reads the agent's angles.
    pub fn smooth_mouselook(&mut self, agent: &mut AgentState, s: &super::CameraSettings, dt: f32, aspect: f32) {
        let active = self.mouselook() && self.vr_mode.is_none() && self.flycam.is_none() && self.free.is_none() && self.track.is_none();
        let lens = Lens { fov_y: self.fov_y, aspect };
        let seated = agent.is_sitting();
        let technique = s.mouselook_smoothing;
        let strength = s.mouselook_smoothing_strength;
        if let Some(view) = self
            .ml_smooth
            .step(active, technique, strength, Angles::of(agent), seated, dt, lens)
        {
            agent.yaw = view.yaw;
            agent.pitch = view.pitch;
        }
    }

    /// Forget the smoothing (a teleport, a mode the mouse does not turn).
    pub fn reset_mouselook_smoothing(&mut self) {
        self.ml_smooth.reset();
    }

    /// The direction of the leading cursor's aim in the world, None in the
    /// other modes (the aim is then the view). Seated in a vehicle the turn
    /// from the view is applied about the view's own axes.
    fn leading_aim_direction(&self) -> Option<Vec3> {
        let (aim, view) = self.ml_smooth.leading()?;
        if !self.mouselook() {
            return None;
        }
        if self.avatar.seat.is_some() && !self.avatar.camera_decoupled {
            let forward = self.forward();
            let up = self.view_up();
            let left = up.cross(forward).normalize_or(Vec3::Y);
            let d = aim.delta(view);
            let turn = glam::Quat::from_axis_angle(up.normalize_or(Vec3::Z), d.x) * glam::Quat::from_axis_angle(left, -d.y);
            return Some((turn * forward).normalize_or(forward));
        }
        Some(aim.forward())
    }

    /// The forward direction a mouselook pick, shot or "under the crosshair"
    /// test uses: the view's, or the leading cursor's aim.
    pub fn mouselook_aim_forward(&self) -> Vec3 {
        self.leading_aim_direction().unwrap_or_else(|| self.forward())
    }

    /// Where the crosshair is, in normalised device coordinates (+y up),
    /// for a screen of `aspect`; None: the centre (every mode but the
    /// leading cursor).
    pub fn mouselook_aim_ndc(&self, aspect: f32) -> Option<Vec2> {
        let direction = self.leading_aim_direction()?;
        let clip = self.proj(aspect) * self.view() * (self.position + direction * 10.0).extend(1.0);
        if clip.w <= 1e-4 {
            return None;
        }
        let p = Vec2::new(clip.x / clip.w, clip.y / clip.w);
        p.is_finite().then(|| p.clamp(Vec2::splat(-1.0), Vec2::splat(1.0)))
    }

    /// The pixel a mouselook click picks on a `width` × `height` view: the
    /// centre, or the leading cursor's crosshair. Every mouselook pick of
    /// "the centre of the screen" goes through here.
    pub fn mouselook_aim_pixel(&self, width: f32, height: f32) -> (f32, f32) {
        let centre = (width * 0.5, height * 0.5);
        if !(width > 0.0 && height > 0.0) {
            return centre;
        }
        self.mouselook_aim_ndc(width / height)
            .map_or(centre, |p| (width * 0.5 * (1.0 + p.x), height * 0.5 * (1.0 - p.y)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LENS: Lens = Lens {
        fov_y: 1.0,
        aspect: 16.0 / 9.0,
    };
    const ON: [Technique; 5] = [
        Technique::FrameAverage,
        Technique::Exponential,
        Technique::OneEuro,
        Technique::DampedSpring,
        Technique::LeadingCursor,
    ];

    fn start(technique: Technique, strength: f32, at: Angles) -> Smoother {
        let mut s = Smoother::default();
        assert_eq!(s.step(true, technique, strength, at, false, 1.0 / 60.0, LENS), None);
        assert!(s.engaged());
        s
    }

    /// A frame as the viewer runs it: the agent holds the last view, the
    /// mouse turns the aim, the step gives the new view.
    fn frame(s: &mut Smoother, technique: Technique, strength: f32, agent: &mut Angles, turn: Vec2, dt: f32) {
        s.turn(turn.x, turn.y);
        if let Some(view) = s.step(true, technique, strength, *agent, false, dt, LENS) {
            *agent = view;
        }
    }

    /// A turn at `rate` (rad/s) for `seconds`, then held still for `hold`
    /// seconds, at `fps`; the view's yaw at the end of each part.
    fn sweep(technique: Technique, strength: f32, rate: Vec2, seconds: f32, hold: f32, fps: f32) -> (Angles, Angles, Smoother) {
        let at = Angles { yaw: 0.0, pitch: 0.0 };
        let mut s = start(technique, strength, at);
        let mut agent = at;
        let dt = 1.0 / fps;
        let n = (seconds * fps).round() as usize;
        for _ in 0..n {
            frame(&mut s, technique, strength, &mut agent, rate * dt, dt);
        }
        let moving = agent;
        let m = (hold * fps).round() as usize;
        for _ in 0..m {
            frame(&mut s, technique, strength, &mut agent, Vec2::ZERO, dt);
        }
        (moving, agent, s)
    }

    #[test]
    fn off_or_strength_zero_bypasses_the_filter() {
        let at = Angles { yaw: 2.9, pitch: -0.3 };
        for technique in Technique::ALL {
            let mut s = Smoother::default();
            for strength in [0.0, -1.0, f32::NAN] {
                assert_eq!(s.step(true, technique, strength, at, false, 1.0 / 60.0, LENS), None);
                assert!(!s.engaged(), "{technique:?}");
            }
        }
        let mut s = Smoother::default();
        for _ in 0..10 {
            assert_eq!(s.step(true, Technique::Off, 1.0, at, false, 1.0 / 60.0, LENS), None);
            assert!(!s.engaged());
        }
        // a turn while bypassed goes nowhere: the caller turns the agent
        s.turn(0.5, 0.5);
        assert!(s.angles().is_none());
    }

    #[test]
    fn switching_off_while_running_keeps_the_aim() {
        let at = Angles { yaw: 0.0, pitch: 0.0 };
        let mut s = start(Technique::Exponential, 1.0, at);
        s.turn(0.4, 0.1);
        let aim = s.angles().expect("running").0;
        assert_eq!(s.step(true, Technique::Off, 1.0, at, false, 1.0 / 60.0, LENS), Some(aim));
        assert!(!s.engaged());
        // leaving mouselook forgets it
        let mut s = start(Technique::Exponential, 1.0, at);
        s.turn(0.4, 0.1);
        assert_eq!(s.step(false, Technique::Exponential, 1.0, at, false, 1.0 / 60.0, LENS), None);
        assert!(!s.engaged());
    }

    #[test]
    fn every_technique_converges_to_a_held_aim_and_ends_exactly_on_it() {
        for technique in ON {
            for strength in [0.05, 0.5, 1.0] {
                let (moving, end, s) = sweep(technique, strength, Vec2::new(2.0, 0.6), 0.4, 12.0, 60.0);
                let (aim, view) = s.angles().expect("running");
                assert_eq!(view, aim, "{technique:?} {strength}");
                assert_eq!(end, aim, "{technique:?} {strength}");
                if strength >= 0.5 && technique != Technique::FrameAverage {
                    assert!(aim.delta(moving).length() > 1e-3, "{technique:?} {strength}: some lag while moving");
                }
            }
        }
    }

    #[test]
    fn the_spring_and_the_leading_cursor_never_overshoot() {
        for technique in [Technique::DampedSpring, Technique::LeadingCursor] {
            for strength in [0.2, 0.6, 1.0] {
                let at = Angles { yaw: 0.0, pitch: 0.0 };
                let mut s = start(technique, strength, at);
                let mut agent = at;
                let dt = 1.0 / 90.0;
                // a fast flick, then held: the view approaches from one side
                for _ in 0..9 {
                    frame(&mut s, technique, strength, &mut agent, Vec2::new(0.06, -0.02), dt);
                }
                let aim = s.angles().expect("running").0;
                let mut last = aim.delta(agent);
                for _ in 0..1200 {
                    frame(&mut s, technique, strength, &mut agent, Vec2::ZERO, dt);
                    let left = aim.delta(agent);
                    assert!(left.x >= 0.0 && left.y <= 0.0, "{technique:?} {strength}: crossed the aim {left}");
                    assert!(left.x <= last.x + 1e-7 && left.y >= last.y - 1e-7, "{technique:?}: went back");
                    last = left;
                }
                assert_eq!(agent, aim);
            }
        }
    }

    #[test]
    fn the_same_motion_ends_at_the_same_place_at_30_and_120_fps() {
        for technique in ON {
            for strength in [0.3, 0.7, 1.0] {
                let rate = Vec2::new(1.5, 0.4);
                let (slow, _, _) = sweep(technique, strength, rate, 0.5, 0.0, 30.0);
                let (fast, _, _) = sweep(technique, strength, rate, 0.5, 0.0, 120.0);
                let gap = fast.delta(slow).length();
                // the frame average's window is quantised by the frames
                let tolerance = 2e-3;
                assert!(gap < tolerance, "{technique:?} {strength}: {gap} rad apart");
                // and both settle on the aim (0.75 rad, 0.2 rad) the same
                let (_, slow_end, _) = sweep(technique, strength, rate, 0.5, 6.0, 30.0);
                let (_, fast_end, _) = sweep(technique, strength, rate, 0.5, 6.0, 120.0);
                assert!(fast_end.delta(slow_end).length() < 1e-4, "{technique:?} {strength}");
            }
        }
    }

    #[test]
    fn yaw_crosses_the_half_turn_the_short_way() {
        for technique in ON {
            let at = Angles {
                yaw: PI - 0.05,
                pitch: 0.0,
            };
            let mut s = start(technique, 0.6, at);
            let mut agent = at;
            for _ in 0..10 {
                frame(&mut s, technique, 0.6, &mut agent, Vec2::new(0.02, 0.0), 1.0 / 60.0);
                assert!(agent.yaw.abs() > 2.9, "{technique:?}: went the long way ({})", agent.yaw);
                assert!((-PI..PI).contains(&agent.yaw));
            }
            for _ in 0..600 {
                frame(&mut s, technique, 0.6, &mut agent, Vec2::ZERO, 1.0 / 60.0);
            }
            assert!((agent.yaw - wrap(PI + 0.15)).abs() < 1e-5, "{technique:?}: {}", agent.yaw);
            assert!(agent.yaw < 0.0);
        }
    }

    #[test]
    fn a_huge_frame_snaps_to_the_aim() {
        for technique in ON {
            let at = Angles { yaw: 0.0, pitch: 0.0 };
            let mut s = start(technique, 1.0, at);
            let mut agent = at;
            frame(&mut s, technique, 1.0, &mut agent, Vec2::new(0.5, 0.2), 1.0 / 60.0);
            for dt in [10.0, f32::INFINITY, f32::NAN, 1e30] {
                s.turn(0.1, 0.0);
                let view = s.step(true, technique, 1.0, agent, false, dt, LENS).expect("running");
                assert_eq!(view, s.angles().expect("running").0, "{technique:?} {dt}");
                assert!(view.yaw.is_finite() && view.pitch.is_finite());
                agent = view;
            }
        }
    }

    #[test]
    fn pitch_stays_within_its_limits() {
        let (up, down) = (clamp_pitch(10.0), clamp_pitch(-10.0));
        for technique in ON {
            let at = Angles {
                yaw: 0.0,
                pitch: up - 0.05,
            };
            let mut s = start(technique, 1.0, at);
            let mut agent = at;
            for i in 0..200 {
                let turn = if i < 40 {
                    0.2
                } else if i < 120 {
                    -0.3
                } else {
                    0.0
                };
                frame(&mut s, technique, 1.0, &mut agent, Vec2::new(0.0, turn), 1.0 / 144.0);
                assert!(agent.pitch <= up && agent.pitch >= down, "{technique:?}: {}", agent.pitch);
                let aim = s.angles().expect("running").0;
                assert!(aim.pitch <= up && aim.pitch >= down);
            }
        }
    }

    #[test]
    fn one_euro_follows_a_fast_sweep_and_damps_jitter() {
        // a fast turn, 3 rad/s (about 170° a second) for half a second
        let (moving, _, s) = sweep(Technique::OneEuro, 0.5, Vec2::new(3.0, 0.0), 0.5, 0.0, 120.0);
        let lag = s.angles().expect("running").0.delta(moving).length();
        assert!(lag < 0.035, "lag {:.2}°", lag.to_degrees());
        // the exponential of the same smoothing of slow motion lags more
        let tau = 1.0 / (TAU * one_euro_min_cutoff(0.5));
        let (exp_moving, _, e) = sweep(
            Technique::Exponential,
            (tau / EXPONENTIAL_TAU).sqrt(),
            Vec2::new(3.0, 0.0),
            0.5,
            0.0,
            120.0,
        );
        let exp_lag = e.angles().expect("running").0.delta(exp_moving).length();
        assert!(exp_lag > 2.0 * lag, "one euro {lag}, exponential {exp_lag}");
        // a hand's tremor: ±0.05° every frame around a still aim
        let at = Angles { yaw: 0.0, pitch: 0.0 };
        let mut s = start(Technique::OneEuro, 0.5, at);
        let mut agent = at;
        let step = 0.1_f32.to_radians();
        let (mut low, mut high) = (f32::MAX, f32::MIN);
        for i in 0..240 {
            let turn = if i % 2 == 0 { step } else { -step };
            frame(&mut s, Technique::OneEuro, 0.5, &mut agent, Vec2::new(turn, 0.0), 1.0 / 120.0);
            if i >= 120 {
                (low, high) = (low.min(agent.yaw), high.max(agent.yaw));
            }
        }
        // the aim swings by 0.1°: the view by less than a fifth of it
        assert!(high - low < step / 5.0, "{:.4}°", (high - low).to_degrees());
    }

    #[test]
    fn the_leading_cursor_keeps_its_aim_on_the_screen() {
        let at = Angles { yaw: 0.0, pitch: 0.0 };
        let mut s = start(Technique::LeadingCursor, 1.0, at);
        let mut agent = at;
        for i in 0..240 {
            let turn = if i < 60 { Vec2::new(0.05, 0.02) } else { Vec2::new(-0.04, -0.03) };
            frame(&mut s, Technique::LeadingCursor, 1.0, &mut agent, turn, 1.0 / 60.0);
            let (aim, view) = s.leading().expect("leading");
            assert_eq!(view, agent);
            let p = ndc(view, aim, LENS).expect("in front");
            assert!(p.x.abs() <= LEAD_LIMIT + 1e-3 && p.y.abs() <= LEAD_LIMIT + 1e-3, "{i}: {p}");
        }
        // the other techniques keep no leading aim (crosshair in the centre)
        for technique in [
            Technique::FrameAverage,
            Technique::Exponential,
            Technique::OneEuro,
            Technique::DampedSpring,
        ] {
            assert!(start(technique, 1.0, at).leading().is_none());
        }
    }

    #[test]
    fn another_turn_of_the_agent_turns_the_aim_and_a_jump_starts_again() {
        let at = Angles { yaw: 0.0, pitch: 0.0 };
        let mut s = start(Technique::DampedSpring, 1.0, at);
        // a stick's turn of the agent between two frames: smoothed as an aim
        let turned = Angles { yaw: 0.1, pitch: 0.0 };
        let view = s
            .step(true, Technique::DampedSpring, 1.0, turned, false, 1.0 / 60.0, LENS)
            .expect("running");
        assert!((s.angles().expect("running").0.yaw - 0.1).abs() < 1e-6);
        assert!(view.yaw < 0.1);
        // a teleport's new heading: no swing
        let away = Angles { yaw: -2.5, pitch: 0.3 };
        assert_eq!(s.step(true, Technique::DampedSpring, 1.0, away, false, 1.0 / 60.0, LENS), None);
        assert_eq!(s.angles(), Some((away, away)));
        // sitting down starts again from the direction then
        assert_eq!(s.step(true, Technique::DampedSpring, 1.0, away, true, 1.0 / 60.0, LENS), None);
        // a turning seat turns aim and view together
        s.turn(0.2, 0.0);
        s.shift_frame(0.3);
        let (aim, view) = s.angles().expect("running");
        assert!((aim.delta(away).x - 0.5).abs() < 1e-6 && (view.delta(away).x - 0.3).abs() < 1e-6);
    }

    #[test]
    fn the_settings_are_bounded_and_round_trip() {
        let loaded: crate::camera::CameraSettings = serde_json::from_str("{}").expect("old file");
        assert_eq!(loaded.mouselook_smoothing, Technique::Off);
        assert_eq!(loaded.mouselook_smoothing_strength, 0.0);
        for technique in Technique::ALL {
            let saved = crate::camera::CameraSettings {
                mouselook_smoothing: technique,
                mouselook_smoothing_strength: 0.37,
                ..Default::default()
            };
            let text = serde_json::to_string(&saved).expect("save");
            let back: crate::camera::CameraSettings = serde_json::from_str(&text).expect("load");
            assert_eq!(back.mouselook_smoothing, technique);
            assert_eq!(back.mouselook_smoothing_strength, 0.37);
        }
        for (v, bounded) in [(-0.5, 0.0), (2.0, 1.0), (f32::NAN, 0.0), (f32::INFINITY, 0.0), (0.25, 0.25)] {
            let s = crate::camera::CameraSettings {
                mouselook_smoothing_strength: v,
                ..Default::default()
            }
            .sanitized();
            assert_eq!(s.mouselook_smoothing_strength, bounded, "{v}");
        }
        assert!(serde_json::from_str::<crate::camera::CameraSettings>(r#"{"mouselook_smoothing":"one_euro"}"#).is_ok());
    }

    #[test]
    fn off_leaves_the_mouse_turn_of_the_agent_bit_for_bit() {
        let settings = crate::camera::CameraSettings::default();
        let mut world = crate::world::World::new(std::sync::Arc::new(crate::scene::avatar::AvatarLibrary::load()));
        let mut camera = super::super::Camera::default();
        camera.change_to_mouselook(&mut world.agent, &settings, false);
        let mut expected = world.agent.clone();
        for (dx, dy) in [(3.0, -1.5), (-0.7, 0.25), (120.0, 40.0)] {
            camera.look(dx, dy, 1.3, &mut world.agent);
            let k = super::super::DRAG_RATE * 1.3;
            expected.yaw -= dx * k;
            expected.pitch = clamp_pitch(expected.pitch - dy * k);
            camera.smooth_mouselook(&mut world.agent, &settings, 1.0 / 60.0, 16.0 / 9.0);
            assert_eq!(world.agent.yaw.to_bits(), expected.yaw.to_bits());
            assert_eq!(world.agent.pitch.to_bits(), expected.pitch.to_bits());
        }
        // on, the mouse turns the aim and the agent follows it
        let on = crate::camera::CameraSettings {
            mouselook_smoothing: Technique::Exponential,
            mouselook_smoothing_strength: 1.0,
            ..Default::default()
        };
        camera.smooth_mouselook(&mut world.agent, &on, 1.0 / 60.0, 16.0 / 9.0);
        let before = world.agent.yaw;
        camera.look(50.0, 0.0, 1.0, &mut world.agent);
        assert_eq!(world.agent.yaw, before, "the mouse turned the aim only");
        camera.smooth_mouselook(&mut world.agent, &on, 1.0 / 60.0, 16.0 / 9.0);
        assert!(world.agent.yaw < before && world.agent.yaw > before - 50.0 * super::super::DRAG_RATE);
    }
}
