// aurora-local-patch: follow-sim-list-module
//! Preferences › Aurora › Animations › Synchronize, on (local patch set):
//! each avatar's playback follows the simulator's AvatarAnimation list
//! exactly. This is Otoa's rule, not a port: the Legacy Viewer orders its
//! motions by activation (load) time and keeps a re-signalled motion in
//! place (LLMotionController::startMotion), and both differ on purpose here.
//!
//! - What plays is what the simulator lists. The few local exceptions are
//!   the ones she chose: the AO keeps hiding the stock clips it replaces,
//!   and our own local starts (AO, movement prediction, previews) play
//!   before the simulator lists them.
//! - The listed motions are ranked by the simulator's sequence numbers,
//!   newest first, whatever the packet order or the load time. A local start
//!   not listed yet ranks above them all, then takes its sequence's rank.
//!   An expression an animation names stays right above that animation. The
//!   built-in motions (head, eyes, breathing, body noise) are not in
//!   `motions`: the blender puts them below every keyframe state of their
//!   priority, as it already did.
//! - One copy per animation. A new sequence for a copy that plays is a
//!   restart: the copy is replaced at once, without a fade, by a new one
//!   that starts at the old copy's weight. A stop fades the one copy out.
//! - Nothing is sent to the simulator for this: the correction is local.
//!
//! `prepare` runs first in `Controller::sync`, `finish` last. Off, or for an
//! animesh (no `follow_simulator` call), both return at once and `sync` is
//! unchanged.

use super::{Controller, Motion};
use crate::world::PlayingAnimation;
use std::collections::HashMap;
use std::time::Instant;
use uuid::Uuid;

/// What the simulator lists for the avatar, and what follows it.
#[derive(Default)]
pub(super) struct Follow {
    active: bool,
    owner: Uuid,
    /// The simulator's list, without our own local upload preview.
    listed: Vec<PlayingAnimation>,
    /// Stock clips the AO hides from our own avatar (rule exception).
    masked: Vec<Uuid>,
    /// The simulator's sequence each copy plays (none for a local start not
    /// listed yet). A stopping copy keeps the last one, so it keeps its
    /// place while it fades.
    ranks: HashMap<Uuid, i32>,
    /// Listed motions our own drawn list lacks, played anyway this sync.
    extras: Vec<Uuid>,
    previous_extras: Vec<Uuid>,
    /// Copies replaced without a fade this sync.
    replaced: Vec<Uuid>,
    /// The listed or local motions playing (not stopping) before this sync.
    before: Vec<Uuid>,
}

impl Controller {
    /// Follow the simulator's list for this avatar (`Some`: the listed
    /// motions and the stock clips the AO hides), or not (`None`: the
    /// setting is off). Called before each `sync_avatar`.
    pub(crate) fn follow_simulator(&mut self, owner: Uuid, simulator: Option<(&[PlayingAnimation], Vec<Uuid>)>) {
        match simulator {
            Some((listed, masked)) => {
                let f = &mut self.follow;
                f.active = true;
                f.owner = owner;
                f.listed.clear();
                f.listed.extend_from_slice(listed);
                f.masked = masked;
            }
            None if self.follow.active => {
                // Back to the usual playback: no weight carried from a replaced copy.
                self.follow = Follow::default();
                for m in &mut self.motions {
                    m.floor = 0.0;
                }
            }
            None => {}
        }
    }
}

/// First step of `Controller::sync`: replace the copies the simulator
/// restarted, and return the list to play when it differs from `playing`
/// (listed motions our own drawn list lacks). None when off.
pub(super) fn prepare(c: &mut Controller, playing: &[PlayingAnimation], now: Instant) -> Option<Vec<PlayingAnimation>> {
    let Controller {
        follow: f,
        motions,
        signals,
        synchronized,
        paused_at,
        paused_real_at,
        animation_clock,
        preview_start,
        ..
    } = c;
    if !f.active {
        return None;
    }
    f.replaced.clear();
    f.before.clear();
    std::mem::swap(&mut f.extras, &mut f.previous_extras);
    f.extras.clear();
    // The motions' times are on the animation clock (Advanced › Animation
    // Speed), their `_real` twins on the wall clock, as in `Controller::sync`.
    let clock = *animation_clock;
    let real_now = paused_real_at.unwrap_or(now);
    let now = paused_at.unwrap_or_else(|| clock.map(now));
    let previewed = |id: Uuid| preview_start.is_some_and(|(target, _)| target.is_none_or(|t| t == id));
    f.before
        .extend(motions.iter().filter(|m| !m.internal && m.stop.is_none()).map(|m| m.id));
    // A new sequence for a copy that plays, or still fades: the simulator
    // restarted it. The new copy takes the old one's place and weight at
    // once, and its new sequence ranks it in `finish`.
    for m in motions
        .iter_mut()
        .filter(|m| !m.internal && !previewed(m.id) && !f.masked.contains(&m.id))
    {
        let Some(sim) = f.listed.iter().find(|s| s.id == m.id) else {
            continue;
        };
        if f.ranks.get(&m.id).is_none_or(|&old| old == sim.sequence) {
            continue;
        }
        // Our own drawn entry keeps driving the copy (an AO pose entry
        // keeps its own signal); the restart's clock is the simulator's.
        let entry = playing.iter().find(|p| p.id == m.id).copied().unwrap_or(*sim);
        let (start, start_real) = if *synchronized {
            (clock.map(sim.sequence_start), sim.sequence_start)
        } else {
            (now, real_now)
        };
        let continuous = if *synchronized {
            (clock.map(entry.continuous_start), entry.continuous_start)
        } else {
            (now, real_now)
        };
        // A loop listed without a break keeps its phase.
        let (continuous_activation, continuous_activation_real) = if entry.continuous_start == m.signal.continuous_start {
            (m.continuous_activation, m.continuous_activation_real)
        } else {
            continuous
        };
        let weight = m.weight;
        *m = Motion {
            id: m.id,
            anim: m.anim.clone(),
            activation: start,
            continuous_activation,
            activation_real: start_real,
            continuous_activation_real,
            signal: entry,
            stop: None,
            residual: weight,
            weight,
            internal: false,
            recovery: None,
            inserted: now,
            floor: weight,
        };
        signals.insert(m.id, entry);
        f.ranks.insert(m.id, sim.sequence);
        f.replaced.push(m.id);
    }
    // Rule 2: listed and not masked, but missing from our own drawn list
    // (an AO clip it stopped, a stand prediction hid). A copy still
    // playing keeps its signal, so it is not started again.
    let mut list: Option<Vec<PlayingAnimation>> = None;
    for s in &f.listed {
        if f.masked.contains(&s.id) || playing.iter().any(|p| p.id == s.id) {
            continue;
        }
        let entry = motions
            .iter()
            .find(|m| !m.internal && m.id == s.id && m.stop.is_none())
            .map_or(*s, |m| m.signal);
        list.get_or_insert_with(|| playing.to_vec()).push(entry);
        f.extras.push(s.id);
    }
    list
}

/// Last step of `Controller::sync`: one copy per animation, then the
/// simulator's order; one log line when either changed something.
pub(super) fn finish(c: &mut Controller, playing: &[PlayingAnimation], now: Instant) {
    let Controller {
        follow: f,
        motions,
        preview_start,
        ..
    } = c;
    if !f.active {
        return;
    }
    let previewed = |id: Uuid| preview_start.is_some_and(|(target, _)| target.is_none_or(|t| t == id));
    // One copy per animation: the copy that plays stays, the others go
    // without a fade, and it starts at their weight (no dip).
    let mut deduplicated = std::mem::take(&mut f.replaced);
    let mut dropped: Option<Vec<bool>> = None;
    for i in 0..motions.len() {
        let id = motions[i].id;
        if motions[i].internal || dropped.as_ref().is_some_and(|d| d[i]) || !motions[i + 1..].iter().any(|o| !o.internal && o.id == id) {
            continue;
        }
        let copies: Vec<usize> = (i..motions.len())
            .filter(|&k| !motions[k].internal && motions[k].id == id)
            .collect();
        let keep = copies.iter().copied().find(|&k| motions[k].stop.is_none()).unwrap_or(i);
        let d = dropped.get_or_insert_with(|| vec![false; motions.len()]);
        let mut carried = 0.0f32;
        for &k in copies.iter().filter(|&&k| k != keep) {
            d[k] = true;
            carried = carried.max(motions[k].weight);
        }
        let m = &mut motions[keep];
        if m.stop.is_none() && carried > m.floor {
            m.floor = carried;
            m.weight = m.ease_in_weight(now);
            m.residual = m.weight;
        }
        if !deduplicated.contains(&id) {
            deduplicated.push(id);
        }
    }
    if let Some(d) = dropped {
        let mut k = 0;
        motions.retain(|_| {
            k += 1;
            !d[k - 1]
        });
    }
    // The sequence each copy follows. A stopping copy keeps its last one.
    for m in motions.iter().filter(|m| !m.internal && !previewed(m.id)) {
        if let Some(s) = f.listed.iter().find(|s| s.id == m.id)
            && (m.stop.is_none() || !f.ranks.contains_key(&m.id))
        {
            f.ranks.insert(m.id, s.sequence);
        }
    }
    f.ranks
        .retain(|id, _| !previewed(*id) && motions.iter().any(|m| !m.internal && m.id == *id));
    // Newest first: the local starts not listed yet (in their current
    // order), then the listed motions by sequence, highest first; ties by
    // id, so the packet order never matters. An expression an animation
    // names sits right above it.
    let keys: Vec<(u8, i64, u128, u8)> = motions
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let carrier = if m.internal { m.signal.id } else { m.id };
            let sub = u8::from(!m.internal);
            match f.ranks.get(&carrier) {
                Some(&rank) => (1, -i64::from(rank), carrier.as_u128(), sub),
                None => {
                    let at = motions.iter().position(|o| !o.internal && o.id == carrier).unwrap_or(i);
                    (0, 0, at as u128, sub)
                }
            }
        })
        .collect();
    let reordered = !keys.is_sorted();
    if reordered {
        let mut keyed: Vec<_> = keys.into_iter().zip(motions.drain(..)).collect();
        keyed.sort_by_key(|(key, _)| *key);
        motions.extend(keyed.into_iter().map(|(_, m)| m));
    }
    // What the correction changed: listed motions put in (or kept against
    // our own drawn list), motions the simulator no longer lists taken out,
    // copies replaced, and the order.
    let live = |id: &Uuid| motions.iter().any(|m| !m.internal && m.id == *id && m.stop.is_none());
    let listed = |id: &Uuid| f.listed.iter().any(|s| s.id == *id);
    let mut added: Vec<Uuid> = motions
        .iter()
        .filter(|m| !m.internal && m.stop.is_none() && listed(&m.id) && !f.before.contains(&m.id) && !deduplicated.contains(&m.id))
        .map(|m| m.id)
        .collect();
    for id in f.extras.iter().filter(|id| !f.previous_extras.contains(id)) {
        if !added.contains(id) {
            added.push(*id);
        }
    }
    let removed: Vec<Uuid> = f
        .before
        .iter()
        .copied()
        .filter(|id| !live(id) && !playing.iter().any(|p| p.id == *id))
        .collect();
    if added.is_empty() && removed.is_empty() && deduplicated.is_empty() && !reordered {
        return;
    }
    let mut parts = Vec::new();
    if !removed.is_empty() {
        parts.push(format!("removed {removed:?}"));
    }
    if !added.is_empty() {
        parts.push(format!("added {added:?}"));
    }
    if !deduplicated.is_empty() {
        parts.push(format!("deduplicated {deduplicated:?}"));
    }
    if reordered {
        let order: Vec<String> = motions
            .iter()
            .map(|m| match (m.internal, f.ranks.get(&m.id)) {
                (true, _) => format!("{} expression", m.id),
                (false, Some(rank)) => format!("{} #{rank}", m.id),
                (false, None) => format!("{} local", m.id),
            })
            .collect();
        parts.push(format!("reordered, newest first [{}]", order.join(", ")));
    }
    log::info!("follow simulator list: avatar {}: {}", f.owner, parts.join("; "));
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::world::World;

    fn rig() -> Rig {
        let xml = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/character/avatar_skeleton.xml")).expect("skeleton");
        Rig::new(&Skeleton::parse(&xml).expect("parse"))
    }

    fn signal(id: Uuid, sequence: i32, start: Instant) -> PlayingAnimation {
        PlayingAnimation {
            id,
            sequence,
            sequence_start: start,
            continuous_start: start,
        }
    }

    /// The same animation listed again with a new sequence at `at`, never
    /// stopped in between (World keeps its continuous start).
    fn again(s: PlayingAnimation, sequence: i32, at: Instant) -> PlayingAnimation {
        PlayingAnimation {
            sequence,
            sequence_start: at,
            ..s
        }
    }

    /// An animation turning `joint` to `angle` at `priority`, with its
    /// (ease in, ease out).
    fn animation(joint: &str, priority: i32, angle: f32, looping: bool, duration: f32, (ease_in, ease_out): (f32, f32)) -> Animation {
        Animation {
            base_priority: priority,
            duration,
            emote_name: String::new(),
            loop_in: 0.0,
            loop_out: duration,
            looping,
            ends_after: None,
            ease_in,
            ease_out,
            hand_pose: 0,
            joints: vec![aurora_assets::JointMotion {
                joint_name: joint.into(),
                priority: -1,
                rot_keys: vec![aurora_assets::anim::RotKey {
                    time: 0.0,
                    rotation: Quat::from_rotation_z(angle),
                }],
                pos_keys: Vec::new(),
            }],
        }
    }

    fn clip(rig: &Rig, joint: &str, priority: i32, angle: f32, looping: bool, duration: f32, ease: (f32, f32)) -> Arc<BoundAnim> {
        Arc::new(BoundAnim::bind(animation(joint, priority, angle, looping, duration, ease), rig))
    }

    const OWNER: Uuid = Uuid::from_u128(9600);

    /// A controller as the scene drives it: Synchronize sets both the
    /// receipt clock and the list to follow.
    fn controller(on: bool) -> Controller {
        Controller {
            synchronized: on,
            ..Default::default()
        }
    }

    /// One sync of an avatar whose drawn list is the simulator's list.
    fn sync(
        c: &mut Controller,
        on: bool,
        list: &[PlayingAnimation],
        now: Instant,
        get: impl FnMut(&Uuid) -> Option<Arc<BoundAnim>>,
    ) -> Vec<Uuid> {
        sync_own(c, on, list, list, &[], now, get)
    }

    /// One sync of our own avatar: `drawn` (AO pose, prediction) beside the
    /// simulator's `listed`, with the stock clips the AO hides.
    fn sync_own(
        c: &mut Controller,
        on: bool,
        drawn: &[PlayingAnimation],
        listed: &[PlayingAnimation],
        masked: &[Uuid],
        now: Instant,
        get: impl FnMut(&Uuid) -> Option<Arc<BoundAnim>>,
    ) -> Vec<Uuid> {
        c.follow_simulator(OWNER, on.then(|| (listed, masked.to_vec())));
        c.sync(drawn, now, get)
    }

    fn ids(c: &Controller) -> Vec<Uuid> {
        c.motions.iter().map(|m| m.id).collect()
    }

    fn joint_off(c: &mut Controller, rig: &Rig, now: Instant, joint: &str, angle: f32) -> f32 {
        let mut out = vec![Mat4::IDENTITY.to_cols_array_2d(); PALETTE_JOINTS];
        c.evaluate(rig, now, None, &mut out);
        c.rot[rig.names[joint]].angle_between(Quat::from_rotation_z(angle))
    }

    const A: Uuid = Uuid::from_u128(9601);
    const B: Uuid = Uuid::from_u128(9602);
    const C: Uuid = Uuid::from_u128(9603);

    /// A late download keeps the simulator's order. Off: as today, the
    /// late one goes in as the newest motion.
    #[test]
    fn a_late_loading_animation_keeps_the_simulators_order() {
        let rig = rig();
        let (clip_a, clip_b) = (
            clip(&rig, "mFaceJaw", 4, 0.6, true, 1.0, (0.0, 0.0)),
            clip(&rig, "mFaceJaw", 4, -0.4, true, 1.0, (0.0, 0.0)),
        );
        let t0 = Instant::now();
        let at = |s: f64| t0 + Duration::from_secs_f64(s);
        // A is the older start (#3), B the newer (#5); A's file comes late.
        let list = [signal(A, 3, t0), signal(B, 5, t0)];
        for on in [false, true] {
            let mut c = controller(on);
            sync(&mut c, on, &list, t0, |id| (*id == B).then(|| clip_b.clone()));
            assert_eq!(ids(&c), [B]);
            sync(&mut c, on, &list, at(1.0), |id| {
                Some(if *id == A { clip_a.clone() } else { clip_b.clone() })
            });
            if on {
                assert_eq!(ids(&c), [B, A], "the newer sequence stays first");
                assert!(joint_off(&mut c, &rig, at(1.1), "mFaceJaw", -0.4) < 1e-4);
            } else {
                assert_eq!(ids(&c), [A, B], "off: loaded last, newest, as today");
                assert!(joint_off(&mut c, &rig, at(1.1), "mFaceJaw", 0.6) < 1e-4);
            }
        }
    }

    /// The same set in another packet order gives the same order and pose.
    /// Off: as today, the last listed is the newest.
    #[test]
    fn the_same_set_in_another_packet_order_gives_the_same_pose() {
        let rig = rig();
        let (clip_a, clip_b, clip_c) = (
            clip(&rig, "mFaceJaw", 4, 0.6, true, 1.0, (0.0, 0.0)),
            clip(&rig, "mFaceJaw", 4, -0.4, true, 1.0, (0.0, 0.0)),
            clip(&rig, "mFaceJaw", 4, 0.2, true, 1.0, (0.0, 0.0)),
        );
        let get = |id: &Uuid| {
            Some(match *id {
                A => clip_a.clone(),
                B => clip_b.clone(),
                _ => clip_c.clone(),
            })
        };
        let t0 = Instant::now();
        let (a, b, cc) = (signal(A, 7, t0), signal(B, 2, t0), signal(C, 4, t0));
        for on in [false, true] {
            let mut first = controller(on);
            sync(&mut first, on, &[a, b, cc], t0, get);
            let mut second = controller(on);
            sync(&mut second, on, &[cc, b, a], t0, get);
            if on {
                assert_eq!(ids(&first), [A, C, B]);
                assert_eq!(ids(&second), [A, C, B]);
                let (x, y) = (
                    joint_off(&mut first, &rig, t0, "mFaceJaw", 0.6),
                    joint_off(&mut second, &rig, t0, "mFaceJaw", 0.6),
                );
                assert!(x < 1e-4 && y < 1e-4, "both drawn with A, the newest sequence");
            } else {
                assert_eq!(ids(&first), [C, B, A], "off: packet order, as today");
                assert_eq!(ids(&second), [A, B, C]);
            }
        }
    }

    /// A restart (new sequence for a copy that plays) moves it to the
    /// newest place with exactly one copy, starting at the old copy's weight.
    /// Off: as today (local patch anim-restart-legacy), kept in place.
    #[test]
    fn a_restart_is_one_copy_at_the_newest_place_without_a_dip() {
        let rig = rig();
        let (clip_a, clip_b) = (
            clip(&rig, "mFaceJaw", 4, 0.6, false, 30.0, (0.8, 0.3)),
            clip(&rig, "mFaceJaw", 4, -0.4, false, 30.0, (0.3, 0.3)),
        );
        let get = |id: &Uuid| Some(if *id == A { clip_a.clone() } else { clip_b.clone() });
        let t0 = Instant::now();
        let at = |s: f64| t0 + Duration::from_secs_f64(s);
        for on in [false, true] {
            let mut c = controller(on);
            let sig_a = signal(A, 1, t0);
            let sig_b = signal(B, 2, at(1.0));
            sync(&mut c, on, &[sig_a], t0, get);
            sync(&mut c, on, &[sig_a, sig_b], at(1.0), get);
            sync(&mut c, on, &[sig_a, sig_b], at(2.0), get);
            assert_eq!(ids(&c), [B, A]);
            let old = c.motions[1].weight;
            assert_eq!(old, 1.0);
            let sig_a = again(sig_a, 3, at(2.2));
            sync(&mut c, on, &[sig_a, sig_b], at(2.2), get);
            if on {
                assert_eq!(ids(&c), [A, B], "the restart is the newest");
                let m = &c.motions[0];
                assert!(m.stop.is_none());
                assert_eq!(m.activation, at(2.2), "a new copy, from its start");
                for s in [2.2, 2.3, 2.6, 3.0, 4.0] {
                    sync(&mut c, on, &[sig_a, sig_b], at(s), get);
                    assert_eq!(c.motions.iter().filter(|m| m.id == A).count(), 1, "one copy at {s} s");
                    assert!(c.motions[0].weight >= old, "no dip at {s} s");
                }
                assert!(joint_off(&mut c, &rig, at(4.0), "mFaceJaw", 0.6) < 1e-4);
            } else {
                assert_eq!(ids(&c), [B, A], "off: kept in place");
                assert_eq!(c.motions[1].activation, t0);
                assert!(joint_off(&mut c, &rig, at(2.3), "mFaceJaw", -0.4) < 1e-4);
            }
        }
        // While it still eases in: on, one copy at its weight; off, as
        // today, a new copy while the old one fades out.
        for on in [false, true] {
            let mut c = controller(on);
            let first = signal(A, 1, t0);
            sync(&mut c, on, &[first], t0, get);
            sync(&mut c, on, &[first], at(0.4), get);
            let old = c.motions[0].weight;
            assert!(old > 0.0 && old < 1.0);
            sync(&mut c, on, &[again(first, 2, at(0.5))], at(0.5), get);
            if on {
                assert_eq!(c.motions.len(), 1);
                assert_eq!(c.motions[0].activation, at(0.5));
                assert!(c.motions[0].weight >= old, "no dip");
            } else {
                assert_eq!(c.motions.len(), 2);
                assert_eq!(c.motions[1].stop, Some(at(0.5)));
            }
        }
    }

    /// A stop fades the single copy out. Listed again while it fades, the
    /// one copy plays again from its fading weight (off: a second copy).
    #[test]
    fn a_stop_fades_the_single_copy_out() {
        let rig = rig();
        let clip_a = clip(&rig, "mFaceJaw", 4, 0.6, true, 1.0, (0.2, 1.0));
        let get = |_: &Uuid| Some(clip_a.clone());
        let t0 = Instant::now();
        let at = |s: f64| t0 + Duration::from_secs_f64(s);
        for on in [false, true] {
            let mut c = controller(on);
            let first = signal(A, 1, t0);
            sync(&mut c, on, &[first], t0, get);
            sync(&mut c, on, &[first], at(1.0), get);
            sync(&mut c, on, &[], at(2.0), get);
            assert_eq!(c.motions.len(), 1);
            assert_eq!(c.motions[0].stop, Some(at(2.0)));
            sync(&mut c, on, &[], at(2.5), get);
            let fading = c.motions[0].weight;
            assert!(fading > 0.0 && fading < 1.0, "it fades, {fading}");
            let back = signal(A, 4, at(2.6));
            sync(&mut c, on, &[back], at(2.6), get);
            if on {
                assert_eq!(c.motions.len(), 1, "one copy");
                assert!(c.motions[0].stop.is_none());
                assert!(c.motions[0].weight >= c.motions[0].floor && c.motions[0].floor > 0.0);
            } else {
                assert_eq!(c.motions.len(), 2, "off: the old copy keeps fading beside the new one");
            }
            sync(&mut c, on, &[], at(3.0), get);
            sync(&mut c, on, &[], at(5.0), get);
            assert!(c.motions.is_empty(), "gone once faded");
        }
    }

    /// Unlisted: taken out. Listed but missing from our own drawn list, and
    /// not hidden on purpose (a stand a predicted start left out): played,
    /// on only.
    #[test]
    fn unlisted_motions_go_and_listed_missing_ones_play() {
        let rig = rig();
        let clip_a = clip(&rig, "mFaceJaw", 4, 0.6, true, 1.0, (0.0, 0.0));
        let get = |_: &Uuid| Some(clip_a.clone());
        let t0 = Instant::now();
        let at = |s: f64| t0 + Duration::from_secs_f64(s);
        for on in [false, true] {
            let mut c = controller(on);
            sync(&mut c, on, &[signal(A, 1, t0), signal(B, 2, t0)], t0, get);
            sync(&mut c, on, &[signal(A, 1, t0)], at(0.1), get);
            sync(&mut c, on, &[signal(A, 1, t0)], at(0.2), get);
            assert_eq!(ids(&c), [A], "B is no longer listed");
        }
        for on in [false, true] {
            let mut c = controller(on);
            let listed = [signal(A, 1, t0), signal(B, 2, t0)];
            sync_own(&mut c, on, &listed[..1], &listed, &[], t0, get);
            if on {
                assert_eq!(ids(&c), [B, A], "listed, so played");
            } else {
                assert_eq!(ids(&c), [A], "off: our drawn list only");
            }
        }
    }

    /// Hidden on purpose while still listed (a clip the AO stopped; a walk an
    /// Instant movement's predicted stop left out): it goes at once, on our
    /// screen only. Not hidden (Server movement): the listed walk plays on
    /// until the simulator drops it.
    #[test]
    fn a_clip_hidden_on_purpose_goes_at_once_while_still_listed() {
        let rig = rig();
        let clip_a = clip(&rig, "mFaceJaw", 4, 0.6, true, 1.0, (0.0, 0.0));
        let get = |_: &Uuid| Some(clip_a.clone());
        let t0 = Instant::now();
        let at = |s: f64| t0 + Duration::from_secs_f64(s);
        let listed = [signal(A, 1, t0), signal(B, 2, t0)];
        for (hidden, on) in [(true, true), (false, true), (true, false), (false, false)] {
            let mut c = controller(on);
            sync_own(&mut c, on, &listed, &listed, &[], t0, get);
            assert_eq!(ids(&c), [B, A]);
            // our drawn list drops B; the simulator still lists it
            let masked: &[Uuid] = if hidden { &[B] } else { &[] };
            sync_own(&mut c, on, &listed[..1], &listed, masked, at(0.1), get);
            sync_own(&mut c, on, &listed[..1], &listed, masked, at(0.2), get);
            if on && !hidden {
                assert_eq!(ids(&c), [B, A], "strict: still listed, so it plays");
                assert!(c.motions[0].stop.is_none());
            } else {
                assert_eq!(ids(&c), [A], "hidden at once (on {on}, hidden {hidden})");
            }
        }
    }

    /// The built-in motions rank below every listed one: breathing turns the
    /// chest at MEDIUM_PRIORITY, a listed animation of that priority keeps it.
    #[test]
    fn built_in_motions_rank_below_listed_ones() {
        let rig = rig();
        let chest = clip(&rig, "mChest", MEDIUM_PRIORITY, 0.3, true, 1.0, (0.0, 0.0));
        let t0 = Instant::now();
        let at = |s: f64| t0 + Duration::from_secs_f64(s);
        for on in [false, true] {
            let mut c = controller(on);
            let list = [signal(A, 1, t0)];
            sync(&mut c, on, &list, t0, |_| Some(chest.clone()));
            for s in [0.5, 1.7, 2.9] {
                sync(&mut c, on, &list, at(s), |_| Some(chest.clone()));
                assert!(joint_off(&mut c, &rig, at(s), "mChest", 0.3) < 1e-4, "on {on}, {s} s");
            }
        }
    }

    /// The stock clip the AO replaces stays hidden although the simulator
    /// lists it; a local start ranks newest until listed, then takes its
    /// sequence's place.
    #[test]
    fn the_ao_keeps_hiding_its_stock_clip_and_local_starts_rank_newest_until_listed() {
        let rig = rig();
        let clip_a = clip(&rig, "mFaceJaw", 4, 0.6, true, 1.0, (0.0, 0.0));
        let get = |_: &Uuid| Some(clip_a.clone());
        let t0 = Instant::now();
        let at = |s: f64| t0 + Duration::from_secs_f64(s);
        let stock = crate::ao::State::Standing.motion();
        let ao = Uuid::from_u128(9610);
        for on in [false, true] {
            let mut c = controller(on);
            // the AO's stand, local (sequence 0), not listed yet
            let drawn = [signal(A, 5, t0), signal(ao, 0, t0)];
            let listed = [signal(stock, 2, t0), signal(A, 5, t0)];
            sync_own(&mut c, on, &drawn, &listed, &[stock], t0, get);
            assert!(!ids(&c).contains(&stock), "the stock stand stays hidden");
            assert_eq!(ids(&c), [ao, A], "the local start is the newest");
            // a newer listed start still ranks under the local one
            let drawn = [signal(A, 5, t0), signal(ao, 0, t0), signal(B, 9, at(0.5))];
            let listed = [signal(stock, 2, t0), signal(A, 5, t0), signal(B, 9, at(0.5))];
            sync_own(&mut c, on, &drawn, &listed, &[stock], at(0.5), get);
            if on {
                assert_eq!(ids(&c), [ao, B, A]);
            } else {
                assert_eq!(ids(&c), [B, ao, A], "off: insertion order, as today");
            }
            // the echo lists the AO's stand at #3: it takes that place
            let listed = [
                signal(stock, 2, t0),
                signal(A, 5, t0),
                signal(ao, 3, at(0.6)),
                signal(B, 9, at(0.5)),
            ];
            sync_own(&mut c, on, &drawn, &listed, &[stock], at(0.6), get);
            if on {
                assert_eq!(ids(&c), [B, A, ao]);
                assert_eq!(c.motions.iter().filter(|m| m.id == ao).count(), 1);
                assert_eq!(c.motions[2].activation, t0, "listing it is not a restart");
            } else {
                assert_eq!(ids(&c), [B, ao, A]);
            }
        }
    }

    /// An expression an animation names stays right above it.
    #[test]
    fn an_expression_stays_right_above_its_animation() {
        let rig = rig();
        let mut streamer = AnimStreamer::new(PathBuf::new());
        default_motions::register(&mut streamer, &rig);
        let wink = default_motions::emote_named("express_wink_emote").expect("wink");
        let mut carrier = animation("mFaceJaw", 4, 0.6, true, 1.0, (0.0, 0.0));
        carrier.emote_name = "express_wink_emote".into();
        let carrier = Arc::new(BoundAnim::bind(carrier, &rig));
        let other = clip(&rig, "mFaceJaw", 4, -0.4, true, 1.0, (0.0, 0.0));
        let t0 = Instant::now();
        let at = |s: f64| t0 + Duration::from_secs_f64(s);
        let mut c = controller(true);
        let list = [signal(A, 2, t0), signal(B, 8, t0)];
        // the carrier loads after B
        sync(&mut c, true, &list, t0, |id| (*id == B).then(|| other.clone()));
        sync(&mut c, true, &list, at(0.1), |id| match *id {
            A => Some(carrier.clone()),
            B => Some(other.clone()),
            _ => streamer.get(id),
        });
        assert_eq!(ids(&c), [B, wink, A]);
        assert!(c.motions[1].internal);
    }

    /// The simulator's list of our own avatar leaves out our local upload
    /// preview, and the AO hides what its pose leaves out.
    #[test]
    fn the_simulators_own_list_and_the_ao_masks() {
        let mut world = World::new(Arc::new(crate::scene::avatar::AvatarLibrary::load()));
        world.agent_id = Uuid::from_u128(9620);
        let me = world.agent_id;
        let other = Uuid::from_u128(9621);
        let stock = crate::ao::State::Standing.motion();
        let clip_x = Uuid::from_u128(9622);
        for avatar in [me, other] {
            world.apply(aurora_net::NetEvent::AvatarAnimations {
                avatar,
                anims: vec![(stock, 1), (clip_x, 2)],
                sources: Vec::new(),
            });
        }
        let ids = |l: &[PlayingAnimation]| l.iter().map(|a| a.id).collect::<Vec<_>>();
        let (listed, masked) = world.simulator_list_of(&other);
        assert_eq!(ids(listed), [stock, clip_x]);
        assert!(masked.is_empty());
        let preview = Uuid::from_u128(9623);
        world.preview_animation(preview, true);
        assert!(ids(world.animations_of(&me)).contains(&preview));
        let (listed, masked) = world.simulator_list_of(&me);
        assert_eq!(ids(listed), [stock, clip_x], "the preview is ours, not listed");
        assert!(masked.is_empty(), "no AO pose");
        let ao = Uuid::from_u128(9624);
        // the AO keeps the preview, replaces the stock stand and stopped clip_x
        let now = Instant::now();
        world.ao_pose = Some(vec![
            PlayingAnimation::from_signal(preview, 0, now, None),
            PlayingAnimation::from_signal(ao, 0, now, None),
        ]);
        let (_, masked) = world.simulator_list_of(&me);
        assert_eq!(masked, [stock, clip_x], "what the pose leaves out stays hidden");
    }
}
