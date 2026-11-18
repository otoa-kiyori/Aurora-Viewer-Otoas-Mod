// aurora-local-patch: ao-over-scripted-module
//! Scripted animation state overrides (llSetAnimationOverride) lose to the
//! AO. The simulator plays the override's own animation in place of the stock
//! state animation, so the viewer is never told the state it stands for: it
//! is told here from the seat, the flight, the keys and the ground contact,
//! and `Runtime::update_with_easing` then replaces it with the AO's animation
//! for that state, as it does the stock one.
//!
//! The Legacy Viewer's AO loses to them: AOEngine::override
//! (indra/newview/aoengine.cpp) only knows the stock state animations
//! (AOSet::getStateByRemapID, indra/newview/aoset.cpp), so an override plays
//! and the AO stops its own animation for that state. This is a deliberate,
//! local difference: the AO wins.
//!
//! An override is a raw signal that is not a stock animation (nor the sit's
//! cover), that this viewer did not ask for (AgentAnimation), whose source
//! (AnimationSourceList) is our agent, one of our attachments or our seat,
//! and while no stock signal of its state plays. An attachment's animation is
//! only taken when it starts at a change of state (within `WINDOW`), so a
//! worn object's own animation (a hold, a dance) started at any other time is
//! left alone. A seat's animation stands for the sit only.

use super::{SIT_GENERIC, State};
use crate::own_motion::{Intent, Mode};
use crate::world::PlayingAnimation;
use crate::world::status::{ANIM_AWAY, ANIM_DO_NOT_DISTURB};
use aurora_net::NetCommand;
use glam::Vec3;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use uuid::Uuid;

/// How long after its first signal an animation can still be taken for an
/// override, and how far from a change of state an attachment's must start.
const WINDOW: Duration = Duration::from_millis(1500);
/// A known override absent from the list (our own stop's echo, a region
/// crossing) keeps its state this long.
const GRACE: Duration = Duration::from_secs(3);
/// An animation we asked for counts as ours this long after its last signal.
const REQUEST_GRACE: Duration = Duration::from_secs(5);
/// Speeds (m/s) the simulator still moves us at with no key: coasting.
const COAST: f32 = 0.5;
const RUN_SPEED: f32 = 4.0;

/// Who started an animation on our avatar, from the AnimationSourceList.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Source {
    Agent,
    Attachment,
    Seat,
    Other,
}

/// What the viewer knows of our avatar's state this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Context {
    pub on_object: bool,
    pub ground_sit: bool,
    pub flying: bool,
    pub on_ground: bool,
    /// The keys sent to the simulator (`Intent::from_flags`).
    pub intent: Option<Intent>,
    /// Horizontal speed.
    pub speed: f32,
    /// The crouch key.
    pub down: bool,
    /// `AgentState::body_turn`: +1 left, -1 right.
    pub body_turn: i8,
}

/// A scripted override standing for `state` this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Override {
    pub state: State,
    pub signal: PlayingAnimation,
    pub source: Source,
}

/// The state the simulator plays for our avatar, as far as the viewer can
/// tell. None when it cannot: a ground sit, or in the air without flying
/// (jumps, falls, landings), where an override stays as it is.
pub fn infer(c: &Context) -> Option<State> {
    if c.on_object {
        return Some(State::Sitting);
    }
    if c.ground_sit {
        return None;
    }
    if c.flying {
        return Some(match c.intent {
            Some(i) if i.direction != Vec3::ZERO => State::Flying,
            Some(i) if i.vertical > 0.0 => State::FlyingUp,
            Some(i) if i.vertical < 0.0 => State::FlyingDown,
            _ if c.speed > RUN_SPEED => State::Flying,
            _ => State::Hovering,
        });
    }
    if !c.on_ground {
        return None;
    }
    Some(match c.intent.map(|i| i.mode) {
        Some(Mode::Run) => State::Running,
        Some(Mode::Crouch) => State::CrouchWalk,
        Some(_) => State::Walking,
        None if c.speed > RUN_SPEED => State::Running,
        None if c.speed > COAST => State::Walking,
        None if c.down => State::Crouching,
        None if c.body_turn > 0 => State::TurningLeft,
        None if c.body_turn < 0 => State::TurningRight,
        None => State::Standing,
    })
}

/// The simulator's and the viewer's own animations, never an override.
fn stock(id: &Uuid) -> bool {
    State::from_motion(*id).is_some() || [SIT_GENERIC, ANIM_AWAY, ANIM_DO_NOT_DISTURB].contains(id)
}

struct Known {
    state: State,
    sequence: i32,
    source: Source,
    seen: Instant,
}

#[derive(Default)]
pub struct Tracker {
    known: HashMap<Uuid, Known>,
    /// Signals not taken (yet): sequence and first signal.
    first: HashMap<Uuid, (i32, Instant)>,
    /// Animations this viewer asked the simulator to start, and their last
    /// signal (or request).
    requested: HashMap<Uuid, Instant>,
    /// The state last inferred, and when it changed.
    inferred: Option<(Option<State>, Instant)>,
}

impl Tracker {
    /// Every command this viewer sends (`App::send_from`): our own starts are
    /// never overrides, and a reset of the overrides forgets them.
    pub fn note_command(&mut self, command: &NetCommand, now: Instant) {
        let mut start = |id: &Uuid| {
            self.requested.insert(*id, now);
        };
        match command {
            NetCommand::AgentAnimation { anim, start: true } => start(anim),
            NetCommand::AgentAnimations { anims, start: true } => anims.iter().for_each(start),
            NetCommand::AgentAnimationChanges { changes } => {
                changes.iter().filter(|(_, s)| *s).for_each(|(id, _)| start(id));
            }
            NetCommand::AnimationStateReset => self.known.clear(),
            _ => {}
        }
    }

    /// This frame's overrides in `raw`. `held` are the signals the runtime
    /// stopped or restarts itself: a known override among them keeps its
    /// state while its echo is gone.
    pub fn update(
        &mut self,
        raw: &[PlayingAnimation],
        context: &Context,
        mut source_of: impl FnMut(&Uuid) -> Option<Source>,
        held: &[Uuid],
        now: Instant,
    ) -> Vec<Override> {
        let state = infer(context);
        if self.inferred.is_none_or(|(old, _)| old != state) {
            self.inferred = Some((state, now));
        }
        let changed = self.inferred.map_or(now, |(_, at)| at);
        let present = |id: &Uuid| raw.iter().any(|a| a.id == *id);
        self.requested.retain(|id, last| {
            if present(id) {
                *last = now;
            }
            now.saturating_duration_since(*last) < REQUEST_GRACE
        });
        self.known
            .retain(|id, k| present(id) || held.contains(id) || now.saturating_duration_since(k.seen) < GRACE);
        self.first.retain(|id, _| present(id));
        let stock_of = |s: State| raw.iter().any(|a| State::from_motion(a.id) == Some(s));
        let mut out = Vec::new();
        for a in raw {
            if stock(&a.id) {
                continue;
            }
            if let Some(k) = self.known.get_mut(&a.id) {
                if k.sequence != a.sequence {
                    // Started again by the simulator: a state entered again.
                    match state {
                        Some(s) => {
                            k.state = s;
                            k.sequence = a.sequence;
                        }
                        None => {
                            self.known.remove(&a.id);
                            self.first.insert(a.id, (a.sequence, now));
                            continue;
                        }
                    }
                }
                k.seen = now;
                if !stock_of(k.state) {
                    out.push(Override {
                        state: k.state,
                        signal: *a,
                        source: k.source,
                    });
                }
                continue;
            }
            let first = self.first.entry(a.id).or_insert((a.sequence, now));
            if first.0 != a.sequence {
                *first = (a.sequence, now);
            }
            let since = first.1;
            if self.requested.contains_key(&a.id) || now.saturating_duration_since(since) > WINDOW {
                continue;
            }
            let Some(s) = state.filter(|s| !stock_of(*s)) else { continue };
            let Some(source) = source_of(&a.id) else { continue };
            let taken = match source {
                Source::Agent => true,
                Source::Attachment => changed + WINDOW >= since,
                Source::Seat => s == State::Sitting,
                Source::Other => false,
            };
            if taken {
                self.first.remove(&a.id);
                self.known.insert(
                    a.id,
                    Known {
                        state: s,
                        sequence: a.sequence,
                        source,
                        seen: now,
                    },
                );
                out.push(Override {
                    state: s,
                    signal: *a,
                    source,
                });
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::super::Set;
    use super::super::runtime::Runtime;
    use super::super::tests::{fixture, frame, step};
    use super::*;
    use aurora_net::control;

    fn ground() -> Context {
        Context {
            on_object: false,
            ground_sit: false,
            flying: false,
            on_ground: true,
            intent: None,
            speed: 0.0,
            down: false,
            body_turn: 0,
        }
    }
    fn keys(flags: u32, running: bool) -> Option<Intent> {
        Intent::from_flags(flags, 0.0, running)
    }
    fn anim(id: u128, sequence: i32, now: Instant) -> PlayingAnimation {
        PlayingAnimation::from_signal(Uuid::from_u128(id), sequence, now, None)
    }

    #[test]
    fn the_state_is_told_from_the_seat_flight_keys_and_ground() {
        let at = |c: Context| infer(&c);
        assert_eq!(at(ground()), Some(State::Standing));
        assert_eq!(
            at(Context {
                on_object: true,
                ..ground()
            }),
            Some(State::Sitting)
        );
        assert_eq!(
            at(Context {
                ground_sit: true,
                ..ground()
            }),
            None,
            "a ground sit is not told"
        );
        assert_eq!(
            at(Context {
                on_ground: false,
                ..ground()
            }),
            None,
            "jumps and falls are not told"
        );
        let walk = keys(control::AT_POS, false);
        assert_eq!(at(Context { intent: walk, ..ground() }), Some(State::Walking));
        let run = keys(control::AT_POS, true);
        assert_eq!(at(Context { intent: run, ..ground() }), Some(State::Running));
        let crouch = keys(control::AT_POS | control::UP_NEG, false);
        assert_eq!(
            at(Context {
                intent: crouch,
                ..ground()
            }),
            Some(State::CrouchWalk)
        );
        let nudge = keys(control::NUDGE_AT_POS, false);
        assert_eq!(at(Context { intent: nudge, ..ground() }), Some(State::Walking));
        assert_eq!(at(Context { speed: 2.0, ..ground() }), Some(State::Walking), "coasting");
        assert_eq!(at(Context { speed: 5.0, ..ground() }), Some(State::Running));
        assert_eq!(at(Context { down: true, ..ground() }), Some(State::Crouching));
        assert_eq!(at(Context { body_turn: 1, ..ground() }), Some(State::TurningLeft));
        assert_eq!(at(Context { body_turn: -1, ..ground() }), Some(State::TurningRight));
        let air = Context {
            flying: true,
            on_ground: false,
            ..ground()
        };
        assert_eq!(at(air), Some(State::Hovering));
        let fly = keys(control::FLY | control::AT_POS, false);
        assert_eq!(at(Context { intent: fly, ..air }), Some(State::Flying));
        let up = keys(control::FLY | control::UP_POS, false);
        assert_eq!(at(Context { intent: up, ..air }), Some(State::FlyingUp));
        let down = keys(control::FLY | control::UP_NEG, false);
        assert_eq!(at(Context { intent: down, ..air }), Some(State::FlyingDown));
    }

    #[test]
    fn an_unasked_animation_of_our_agent_attachment_or_seat_is_an_override() {
        let now = Instant::now();
        let x = anim(0x51, 1, now);
        let take = |source: Option<Source>, context: Context| {
            let mut t = Tracker::default();
            t.update(&[x], &context, |_| source, &[], now)
        };
        let standing = take(Some(Source::Agent), ground());
        assert_eq!(
            standing,
            vec![Override {
                state: State::Standing,
                signal: x,
                source: Source::Agent
            }]
        );
        assert_eq!(take(Some(Source::Attachment), ground()).len(), 1, "started at a change of state");
        assert!(take(Some(Source::Other), ground()).is_empty(), "another object's animation");
        assert!(take(None, ground()).is_empty(), "no known source");
        assert!(take(Some(Source::Seat), ground()).is_empty(), "a seat's stands for the sit only");
        let seated = Context {
            on_object: true,
            ..ground()
        };
        assert_eq!(take(Some(Source::Seat), seated)[0].state, State::Sitting);
        assert!(
            take(
                Some(Source::Agent),
                Context {
                    on_ground: false,
                    ..ground()
                }
            )
            .is_empty(),
            "a state not told stays scripted"
        );
        let mut t = Tracker::default();
        let stock = PlayingAnimation::from_signal(State::Standing.motion(), 1, now, None);
        assert!(
            t.update(&[x, stock], &ground(), |_| Some(Source::Agent), &[], now).is_empty(),
            "with the stock signal of its state, not an override"
        );
        for id in [SIT_GENERIC, ANIM_AWAY, State::Walking.motion()] {
            let a = PlayingAnimation::from_signal(id, 1, now, None);
            assert!(
                Tracker::default()
                    .update(&[a], &ground(), |_| Some(Source::Agent), &[], now)
                    .is_empty()
            );
        }
    }

    #[test]
    fn our_own_starts_are_never_overrides() {
        let now = Instant::now();
        let x = anim(0x52, 1, now);
        for command in [
            NetCommand::AgentAnimation { anim: x.id, start: true },
            NetCommand::AgentAnimations {
                anims: vec![x.id],
                start: true,
            },
            NetCommand::AgentAnimationChanges {
                changes: vec![(Uuid::from_u128(9), false), (x.id, true)],
            },
        ] {
            let mut t = Tracker::default();
            t.note_command(&command, now);
            assert!(t.update(&[x], &ground(), |_| Some(Source::Agent), &[], now).is_empty());
            let later = now + REQUEST_GRACE + Duration::from_secs(1);
            assert!(
                t.update(&[x], &ground(), |_| Some(Source::Agent), &[], later).is_empty(),
                "still ours while it plays"
            );
        }
        let mut t = Tracker::default();
        t.note_command(&NetCommand::AgentAnimation { anim: x.id, start: false }, now);
        assert_eq!(t.update(&[x], &ground(), |_| Some(Source::Agent), &[], now).len(), 1);
    }

    #[test]
    fn a_worn_objects_animation_started_away_from_a_change_of_state_is_left_alone() {
        let now = Instant::now();
        let mut t = Tracker::default();
        assert!(t.update(&[], &ground(), |_| None, &[], now).is_empty());
        let later = now + Duration::from_secs(10);
        let hold = anim(0x53, 1, later);
        let attachment = |_: &Uuid| Some(Source::Attachment);
        assert!(t.update(&[hold], &ground(), attachment, &[], later).is_empty(), "a hold, a dance");
        let walk = Context {
            intent: keys(control::AT_POS, false),
            ..ground()
        };
        let step = later + Duration::from_secs(2);
        assert!(
            t.update(&[hold], &walk, attachment, &[], step).is_empty(),
            "not taken later either, at the next change of state"
        );
        let walking = anim(0x54, 1, step);
        let r = t.update(&[hold, walking], &walk, attachment, &[], step + Duration::from_millis(200));
        assert_eq!(r.len(), 1, "the one that came with the walk");
        assert_eq!((r[0].signal.id, r[0].state), (walking.id, State::Walking));
    }

    #[test]
    fn a_known_override_keeps_its_state_while_held_and_is_told_again_when_restarted() {
        let now = Instant::now();
        let x = anim(0x55, 1, now);
        let agent = |_: &Uuid| Some(Source::Agent);
        let mut t = Tracker::default();
        assert_eq!(t.update(&[x], &ground(), agent, &[], now).len(), 1);
        let later = now + GRACE + Duration::from_secs(10);
        assert!(t.update(&[], &ground(), agent, &[x.id], later).is_empty(), "stopped by us");
        t.note_command(&NetCommand::AgentAnimation { anim: x.id, start: true }, later);
        let back = t.update(&[x], &ground(), agent, &[], later);
        assert_eq!(back.len(), 1, "our own restart of a known override is still the override");
        let walk = Context {
            intent: keys(control::AT_POS, false),
            ..ground()
        };
        let again = PlayingAnimation { sequence: 2, ..x };
        assert_eq!(
            t.update(&[again], &walk, agent, &[], later)[0].state,
            State::Walking,
            "a new sequence"
        );
        let mut t2 = Tracker::default();
        t2.update(&[x], &ground(), agent, &[], now);
        t2.note_command(&NetCommand::AnimationStateReset, now);
        t2.note_command(&NetCommand::AgentAnimation { anim: x.id, start: true }, now);
        assert!(t2.update(&[x], &ground(), agent, &[], now).is_empty(), "a reset forgets them");
    }

    fn scripted(state: State, signal: PlayingAnimation, source: Source) -> [Override; 1] {
        [Override { state, signal, source }]
    }

    #[test]
    fn the_ao_replaces_a_scripted_stand_and_gives_it_back_when_off() {
        let (_, set) = fixture();
        let config = &set.states[&State::Standing];
        let stands: Vec<Uuid> = config
            .steps
            .iter()
            .flat_map(|s| s.members.iter())
            .chain(config.tracks.iter().flat_map(|t| t.members.iter()))
            .map(|m| m.asset)
            .collect();
        let now = Instant::now();
        let x = anim(0x56, 1, now);
        let raw = [x];
        let overrides = scripted(State::Standing, x, Source::Agent);
        let mut f = frame(&raw);
        f.scripted = &overrides;
        let mut r = Runtime::default();
        let (pose, cmd) = r.update(Some(&set), f, now, |_| true, |_| 0);
        assert_eq!(r.current, Some(State::Standing));
        assert!(cmd.contains(&(x.id, false)), "the override stopped for everyone: {cmd:?}");
        let stand = cmd
            .iter()
            .find(|(id, start)| *start && stands.contains(id))
            .map(|(id, _)| *id)
            .expect("the AO's stand started");
        assert!(pose.iter().any(|a| a.id == stand) && !pose.iter().any(|a| a.id == x.id));
        let echo = now + Duration::from_millis(200);
        let (pose, cmd) = r.update(Some(&set), frame(&[]), echo, |_| true, |_| 0);
        assert_eq!(r.current, Some(State::Standing), "our stop's echo keeps the state");
        assert!(cmd.is_empty(), "{cmd:?}");
        assert!(pose.iter().any(|a| a.id == stand));
        let mut off = frame(&[]);
        off.enabled = false;
        let (pose, cmd) = r.update(Some(&set), off, echo + Duration::from_millis(100), |_| true, |_| 0);
        assert!(cmd.contains(&(x.id, true)), "AO off: the override back: {cmd:?}");
        assert!(cmd.contains(&(stand, false)));
        assert!(pose.iter().any(|a| a.id == x.id));
    }

    #[test]
    fn a_scripted_override_without_an_ao_animation_or_under_a_stock_signal_stays() {
        let (_, mut set) = fixture();
        let now = Instant::now();
        let x = anim(0x57, 1, now);
        let walk = PlayingAnimation::from_signal(State::Walking.motion(), 1, now + Duration::from_millis(10), None);
        let raw = [x, walk];
        let overrides = scripted(State::Standing, x, Source::Agent);
        let mut f = frame(&raw);
        f.scripted = &overrides;
        let mut r = Runtime::default();
        let (pose, cmd) = r.update(Some(&set), f, now, |_| true, |_| 0);
        assert_eq!(r.current, Some(State::Walking), "a stock signal wins");
        assert!(pose.iter().any(|a| a.id == x.id) && !cmd.contains(&(x.id, false)));
        set.states.remove(&State::Standing);
        let raw = [x];
        let mut f = frame(&raw);
        f.scripted = &overrides;
        let mut r = Runtime::default();
        let (pose, cmd) = r.update(Some(&set), f, now, |_| true, |_| 0);
        assert!(pose.iter().any(|a| a.id == x.id), "no AO stand: the override plays");
        assert!(!cmd.contains(&(x.id, false)));
    }

    #[test]
    fn a_scripted_sit_follows_the_sit_options() {
        let (_, set) = fixture();
        let sit = step(&set, State::Sitting, 0);
        let now = Instant::now();
        let x = anim(0x58, 1, now);
        let raw = [x];
        let run = |set: &Set, source: Source, seat_animations: bool| {
            let overrides = scripted(State::Sitting, x, source);
            let mut f = frame(&raw);
            f.on_object = true;
            f.seat_animations = seat_animations;
            f.scripted = &overrides;
            Runtime::default().update(Some(set), f, now, |_| true, |_| 0)
        };
        assert!(set.sits && set.smart);
        for (source, seat) in [(Source::Agent, false), (Source::Seat, true)] {
            let (pose, cmd) = run(&set, source, seat);
            assert!(cmd.is_empty(), "smart sit: the scripted sit stays ({source:?}): {cmd:?}");
            assert!(pose.iter().any(|a| a.id == x.id));
        }
        let (pose, cmd) = run(&set, Source::Attachment, false);
        assert_eq!(cmd, vec![(SIT_GENERIC, true), (sit, true)], "a worn object's sit is not the seat's");
        assert!(!pose.iter().any(|a| a.id == x.id), "masked here, never stopped");
        let mut no_smart = set.clone();
        no_smart.smart = false;
        for (source, seat) in [(Source::Agent, false), (Source::Seat, true)] {
            let (pose, cmd) = run(&no_smart, source, seat);
            assert_eq!(cmd, vec![(SIT_GENERIC, true), (sit, true)], "{source:?}");
            assert!(!pose.iter().any(|a| a.id == x.id));
        }
        let mut no_sits = set.clone();
        no_sits.sits = false;
        let (pose, cmd) = run(&no_sits, Source::Attachment, false);
        assert!(cmd.is_empty() && pose.iter().any(|a| a.id == x.id), "sit override off");
    }
}
