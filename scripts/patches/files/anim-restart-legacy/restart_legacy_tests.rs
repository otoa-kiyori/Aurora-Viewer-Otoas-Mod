// aurora-local-patch: anim-restart-legacy-tests
//! A motion the simulator signals again (a new sequence id) while it plays:
//! LLMotionController::startMotion (indra/llcharacter/llmotioncontroller.cpp)
//! lets an active, fully eased-in instance keep playing, and starts a new one
//! only while it eases in or out, once it has stopped or ended, and always for
//! an expression (LLEmote::canDeprecate). Then the 0 s one-shots, which end
//! themselves once drawn (LLKeyframeMotion::onUpdate). A child of `anim`, like
//! its `tests`.

use super::*;

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

/// The same animation signalled again with a new sequence id at `at`
/// (World keeps its continuous start: it was never stopped).
fn again(s: PlayingAnimation, sequence: i32, at: Instant) -> PlayingAnimation {
    PlayingAnimation {
        sequence,
        sequence_start: at,
        ..s
    }
}

/// A Bento face animation: it turns the jaw to `angle` at `priority`.
fn face(rig: &Rig, priority: i32, angle: f32, looping: bool, duration: f32, ease_in: f32, ease_out: f32) -> Arc<BoundAnim> {
    let anim = Animation {
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
            joint_name: "mFaceJaw".into(),
            priority: -1,
            rot_keys: vec![aurora_assets::anim::RotKey {
                time: 0.0,
                rotation: Quat::from_rotation_z(angle),
            }],
            pos_keys: Vec::new(),
        }],
    };
    Arc::new(BoundAnim::bind(anim, rig))
}

fn controller(synchronized: bool) -> Controller {
    Controller {
        synchronized,
        ..Default::default()
    }
}

/// The jaw's distance (radians) from the pose of `angle` after evaluating at `now`.
fn jaw_off(c: &mut Controller, rig: &Rig, now: Instant, angle: f32) -> f32 {
    let mut out = vec![Mat4::IDENTITY.to_cols_array_2d(); PALETTE_JOINTS];
    c.evaluate(rig, now, None, &mut out);
    c.rot[rig.names["mFaceJaw"]].angle_between(Quat::from_rotation_z(angle))
}

fn ids(c: &Controller) -> Vec<Uuid> {
    c.motions.iter().map(|m| m.id).collect()
}

/// (a) The mesh-head case: the head's script plays a long face animation
/// again and again; an expression started after it, at its priority, keeps
/// the face. Before: each re-signal put the old animation first again.
#[test]
fn a_resignalled_one_shot_keeps_its_place_under_a_newer_expression() {
    let rig = rig();
    let (a, b) = (Uuid::from_u128(9201), Uuid::from_u128(9202));
    let clip_a = face(&rig, 4, 0.6, false, 30.0, 0.3, 0.3);
    let clip_b = face(&rig, 4, -0.4, false, 30.0, 0.3, 0.3);
    let get = |id: &Uuid| Some(if *id == a { clip_a.clone() } else { clip_b.clone() });
    let t0 = Instant::now();
    let at = |s: f64| t0 + Duration::from_secs_f64(s);
    for synchronized in [false, true] {
        let mut c = controller(synchronized);
        let mut sig_a = signal(a, 1, t0);
        let sig_b = signal(b, 1, at(1.0));
        c.sync(&[sig_a], t0, get);
        c.sync(&[sig_a, sig_b], at(1.0), get);
        assert!(c.sync(&[sig_a, sig_b], at(2.0), get).is_empty());
        assert!(jaw_off(&mut c, &rig, at(2.0), -0.4) < 1e-4, "the newer expression has the face");
        // every 2.2 s, as the reported log shows
        for (n, s) in (2..).zip([2.2, 4.4, 6.6, 8.8, 11.0, 13.2, 15.4, 17.6, 19.8]) {
            sig_a = again(sig_a, n, at(s));
            assert!(c.sync(&[sig_a, sig_b], at(s), get).is_empty(), "synchronized {synchronized}, {s} s");
            assert_eq!(
                ids(&c),
                [b, a],
                "kept in place, not started again (synchronized {synchronized}, {s} s)"
            );
            let kept = &c.motions[1];
            assert_eq!(kept.activation, t0, "its clock goes on");
            assert!(kept.stop.is_none());
            assert_eq!(c.signals[&a], sig_a, "the new sequence is recorded once");
            assert!(c.sync(&[sig_a, sig_b], at(s + 0.1), get).is_empty());
            assert_eq!(c.motions.len(), 2);
            assert!(
                jaw_off(&mut c, &rig, at(s + 0.1), -0.4) < 1e-4,
                "synchronized {synchronized}, {s} s"
            );
        }
        // it still stops itself once, at the end of its first start
        assert!(c.sync(&[sig_a, sig_b], at(29.6), get).is_empty());
        assert_eq!(c.sync(&[sig_a, sig_b], at(29.75), get), [a]);
        assert!(c.sync(&[sig_a, sig_b], at(29.8), get).is_empty());
    }
}

/// (b) Easing in, easing out or after its timed stop: a new instance, the
/// old one fades out. Paused (LLCharacter::requestPause), nothing is
/// replaced. An expression is always started again.
#[test]
fn a_one_shot_resignalled_while_blending_or_stopping_starts_again() {
    let rig = rig();
    let id = Uuid::from_u128(9203);
    let clip = face(&rig, 4, 0.6, false, 4.0, 1.0, 0.5);
    let get = |_: &Uuid| Some(clip.clone());
    let t0 = Instant::now();
    let at = |s: f64| t0 + Duration::from_secs_f64(s);
    for synchronized in [false, true] {
        // easing in
        let mut c = controller(synchronized);
        let first = signal(id, 1, t0);
        c.sync(&[first], t0, get);
        c.sync(&[first], at(0.5), get);
        assert!(c.motions[0].weight > 0.0 && c.motions[0].weight < 1.0);
        let next = again(first, 2, at(0.6));
        c.sync(&[next], at(0.6), get);
        assert_eq!(c.motions.len(), 2, "synchronized {synchronized}");
        assert_eq!(c.motions[0].activation, at(0.6));
        assert!(c.motions[0].stop.is_none());
        assert_eq!(c.motions[1].stop, Some(at(0.6)));
        // easing out, after the timed stop (duration - ease out = 3.5 s)
        let mut c = controller(synchronized);
        c.sync(&[first], t0, get);
        c.sync(&[first], at(2.0), get);
        assert_eq!(c.motions[0].weight, 1.0);
        assert_eq!(c.sync(&[first], at(3.5), get), [id]);
        assert_eq!(c.motions[0].stop, Some(at(3.5)));
        c.sync(&[again(first, 2, at(3.7))], at(3.7), get);
        assert_eq!(c.motions.len(), 2, "synchronized {synchronized}");
        assert_eq!(c.motions[0].activation, at(3.7));
        assert!(c.motions[0].stop.is_none());
        assert_eq!(c.motions[1].stop, Some(at(3.5)), "the old one keeps easing out");
        // paused during the ease-in: kept (startMotion deprecates nothing while mPaused)
        let mut c = controller(synchronized);
        c.sync(&[first], t0, get);
        c.sync(&[first], at(0.5), get);
        c.set_paused(true, at(0.5));
        c.sync(&[again(first, 2, at(0.7))], at(0.7), get);
        assert_eq!(c.motions.len(), 1, "synchronized {synchronized}");
        assert_eq!(c.motions[0].activation, t0);
        assert!(c.motions[0].stop.is_none());
    }
    // an expression at full weight starts again (LLEmote::canDeprecate is false)
    let mut streamer = AnimStreamer::new(PathBuf::new());
    default_motions::register(&mut streamer, &rig);
    let wink = default_motions::emote_named("express_wink_emote").expect("wink");
    for synchronized in [false, true] {
        let mut c = controller(synchronized);
        let first = signal(wink, 1, t0);
        c.sync(&[first], t0, |m| streamer.get(m));
        c.sync(&[first], at(1.0), |m| streamer.get(m));
        assert_eq!(c.motions[0].weight, 1.0);
        c.sync(&[again(first, 2, at(1.1))], at(1.1), |m| streamer.get(m));
        assert_eq!(c.motions.len(), 2, "synchronized {synchronized}");
        assert_eq!(c.motions[0].activation, at(1.1));
    }
}

/// (c) After its end a one-shot starts again, also when no frame ran
/// between its end and the new signal.
#[test]
fn a_one_shot_resignalled_after_it_ended_starts_again() {
    let rig = rig();
    let id = Uuid::from_u128(9204);
    let clip = face(&rig, 4, 0.6, false, 2.0, 0.2, 0.5);
    let get = |_: &Uuid| Some(clip.clone());
    let t0 = Instant::now();
    let at = |s: f64| t0 + Duration::from_secs_f64(s);
    for synchronized in [false, true] {
        let mut c = controller(synchronized);
        let first = signal(id, 1, t0);
        c.sync(&[first], t0, get);
        assert_eq!(c.sync(&[first], at(1.5), get), [id]);
        c.sync(&[first], at(2.5), get);
        assert!(c.motions.is_empty());
        let next = again(first, 2, at(3.0));
        assert!(c.sync(&[next], at(3.0), get).is_empty());
        assert_eq!(c.motions.len(), 1, "synchronized {synchronized}");
        assert_eq!(c.motions[0].activation, at(3.0));
        c.sync(&[next], at(3.3), get);
        assert_eq!(c.motions[0].weight, 1.0);
        // its end (4.5 s) passed with no frame before the next signal
        let last = again(next, 3, at(4.6));
        assert!(c.sync(&[last], at(4.6), get).is_empty(), "no stale completion");
        assert_eq!(c.motions.len(), 2, "synchronized {synchronized}");
        assert_eq!(c.motions[0].activation, at(4.6));
        assert!(c.motions[0].stop.is_none());
        assert_eq!(c.motions[1].stop, Some(at(4.6)));
        assert_eq!(c.sync(&[last], at(6.1), get), [id], "the new start completes on its own clock");
    }
}

/// (d) Loops are unchanged: a re-signalled loop keeps its place and its
/// continuous clock, also during its ease-in (Aurora keeps loops continuous).
#[test]
fn a_resignalled_loop_is_kept_as_before() {
    let rig = rig();
    let (l, b) = (Uuid::from_u128(9205), Uuid::from_u128(9206));
    let clip_l = face(&rig, 4, 0.6, true, 1.0, 0.5, 0.5);
    let clip_b = face(&rig, 4, -0.4, false, 30.0, 0.3, 0.3);
    let get = |id: &Uuid| Some(if *id == l { clip_l.clone() } else { clip_b.clone() });
    let t0 = Instant::now();
    let at = |s: f64| t0 + Duration::from_secs_f64(s);
    for synchronized in [false, true] {
        let mut c = controller(synchronized);
        let first = signal(l, 1, t0);
        c.sync(&[first], t0, get);
        // during its ease-in
        let second = again(first, 2, at(0.25));
        c.sync(&[second], at(0.25), get);
        assert_eq!(c.motions.len(), 1, "synchronized {synchronized}");
        assert_eq!(c.motions[0].continuous_activation, t0);
        // under a newer one-shot of its priority
        let sig_b = signal(b, 1, at(1.0));
        c.sync(&[second, sig_b], at(1.0), get);
        c.sync(&[second, sig_b], at(2.0), get);
        let third = again(first, 3, at(2.2));
        c.sync(&[third, sig_b], at(2.2), get);
        assert_eq!(ids(&c), [b, l]);
        assert_eq!(c.motions[1].continuous_activation, t0);
        assert_eq!(
            c.motions[1].activation,
            at(2.2),
            "as before: the loop's activation follows the sequence"
        );
        assert_eq!(c.signals[&l], third);
        assert!(jaw_off(&mut c, &rig, at(2.3), -0.4) < 1e-4);
    }
}

// ------------------------------------------------------------ 0 s one-shots
//
// A non-looping animation of 0 s (a one-frame pose, as many Bento face
// expressions are) ends itself on its first update after the start:
// LLKeyframeMotion::onUpdate returns false once its time passes 0, and
// LLMotionController::updateMotionsByType stops it and asks the simulator to
// stop it. The joints it alone drives keep its pose.

/// The next frame.
const FRAME: f64 = 0.016;

/// An animation turning each `(joint, angle)` at `priority`.
fn keyed(rig: &Rig, priority: i32, joints: &[(&str, f32)], looping: bool, duration: f32, ease_in: f32, ease_out: f32) -> Arc<BoundAnim> {
    let anim = Animation {
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
        joints: joints
            .iter()
            .map(|&(joint, angle)| aurora_assets::JointMotion {
                joint_name: joint.into(),
                priority: -1,
                rot_keys: vec![aurora_assets::anim::RotKey {
                    time: 0.0,
                    rotation: Quat::from_rotation_z(angle),
                }],
                pos_keys: Vec::new(),
            })
            .collect(),
    };
    Arc::new(BoundAnim::bind(anim, rig))
}

/// The distance (radians) of `joint` from the pose of `angle` after
/// evaluating at `now`.
fn off(c: &mut Controller, rig: &Rig, now: Instant, joint: &str, angle: f32) -> f32 {
    let mut out = vec![Mat4::IDENTITY.to_cols_array_2d(); PALETTE_JOINTS];
    c.evaluate(rig, now, None, &mut out);
    c.rot[rig.names[joint]].angle_between(Quat::from_rotation_z(angle))
}

/// (e) Drawn once at full weight, ended at the next sync with a simulator
/// stop, and its pose held on the joints nothing else drives.
#[test]
fn a_zero_second_pose_is_drawn_once_then_ends_and_its_pose_holds() {
    let rig = rig();
    let id = Uuid::from_u128(9207);
    let pose = keyed(&rig, 6, &[("mFaceJaw", 0.5)], false, 0.0, 0.0, 0.0);
    let get = |_: &Uuid| Some(pose.clone());
    let t0 = Instant::now();
    let at = |s: f64| t0 + Duration::from_secs_f64(s);
    for synchronized in [false, true] {
        let mut c = controller(synchronized);
        let first = signal(id, 1, t0);
        assert!(c.sync(&[first], t0, get).is_empty(), "not before it is drawn");
        assert_eq!(c.motions[0].weight, 1.0);
        assert!(off(&mut c, &rig, t0, "mFaceJaw", 0.5) < 1e-4, "synchronized {synchronized}");
        assert_eq!(c.sync(&[first], at(FRAME), get), [id], "the simulator stop, once");
        assert!(c.motions.is_empty());
        for s in [2.0 * FRAME, 1.0, 10.0] {
            assert!(c.sync(&[first], at(s), get).is_empty());
            assert!(off(&mut c, &rig, at(s), "mFaceJaw", 0.5) < 1e-4, "held at {s} s");
        }
    }
}

/// (f) A lower-priority idle on a bone the pose also turned takes it back;
/// the pose holds on the bone the idle leaves alone.
#[test]
fn a_lower_priority_idle_takes_its_bones_back_after_a_zero_second_pose() {
    let rig = rig();
    let (idle, pose) = (Uuid::from_u128(9208), Uuid::from_u128(9209));
    let brow = "mFaceEyebrowInnerLeft";
    let idle_clip = keyed(&rig, 2, &[(brow, -0.3)], true, 2.0, 0.0, 0.0);
    let pose_clip = keyed(&rig, 6, &[("mFaceJaw", 0.5), (brow, 0.4)], false, 0.0, 0.0, 0.0);
    let get = |id: &Uuid| Some(if *id == idle { idle_clip.clone() } else { pose_clip.clone() });
    let t0 = Instant::now();
    let at = |s: f64| t0 + Duration::from_secs_f64(s);
    for synchronized in [false, true] {
        let mut c = controller(synchronized);
        let sig_idle = signal(idle, 1, t0);
        let sig_pose = signal(pose, 1, at(1.0));
        c.sync(&[sig_idle], t0, get);
        assert!(c.sync(&[sig_idle, sig_pose], at(1.0), get).is_empty());
        assert!(off(&mut c, &rig, at(1.0), brow, 0.4) < 1e-4, "the pose wins its frame");
        assert!(off(&mut c, &rig, at(1.0), "mFaceJaw", 0.5) < 1e-4);
        assert_eq!(c.sync(&[sig_idle, sig_pose], at(1.0 + FRAME), get), [pose]);
        assert_eq!(ids(&c), [idle], "synchronized {synchronized}");
        assert!(off(&mut c, &rig, at(1.0 + FRAME), brow, -0.3) < 1e-4, "the idle has the brow back");
        assert!(off(&mut c, &rig, at(1.0 + FRAME), "mFaceJaw", 0.5) < 1e-4, "the jaw holds the pose");
    }
}

/// (g) Signalled again after it ended (a click on the same expression), it
/// starts again first, over a newer motion of its priority; signalled again
/// on the frame it ends, it starts again too.
#[test]
fn a_zero_second_pose_signalled_again_starts_again_in_front() {
    let rig = rig();
    let (a, b) = (Uuid::from_u128(9210), Uuid::from_u128(9211));
    let clip_a = keyed(&rig, 6, &[("mFaceJaw", 0.5)], false, 0.0, 0.0, 0.0);
    let clip_b = keyed(&rig, 6, &[("mFaceJaw", -0.4)], false, 30.0, 0.0, 0.0);
    let get = |id: &Uuid| Some(if *id == a { clip_a.clone() } else { clip_b.clone() });
    let t0 = Instant::now();
    let at = |s: f64| t0 + Duration::from_secs_f64(s);
    for synchronized in [false, true] {
        let mut c = controller(synchronized);
        let first = signal(a, 1, t0);
        c.sync(&[first], t0, get);
        assert_eq!(c.sync(&[first], at(FRAME), get), [a]);
        let sig_b = signal(b, 1, at(1.0));
        c.sync(&[first, sig_b], at(1.0), get);
        assert!(off(&mut c, &rig, at(1.0), "mFaceJaw", -0.4) < 1e-4, "the newer motion has the jaw");
        let second = again(first, 2, at(2.0));
        assert!(c.sync(&[second, sig_b], at(2.0), get).is_empty());
        assert_eq!(ids(&c), [a, b], "synchronized {synchronized}");
        assert!(off(&mut c, &rig, at(2.0), "mFaceJaw", 0.5) < 1e-4, "the click shows");
        // signalled again on the frame it ends: not kept, started again
        let third = again(first, 3, at(2.0 + FRAME));
        assert!(c.sync(&[third, sig_b], at(2.0 + FRAME), get).is_empty());
        assert_eq!(ids(&c), [a, b], "the ended instance went, the new one is first");
        assert_eq!(c.motions[0].inserted, at(2.0 + FRAME));
        assert!(off(&mut c, &rig, at(2.0 + FRAME), "mFaceJaw", 0.5) < 1e-4);
        assert_eq!(c.sync(&[third, sig_b], at(2.0 + 2.0 * FRAME), get), [a]);
        assert!(
            off(&mut c, &rig, at(2.0 + 2.0 * FRAME), "mFaceJaw", -0.4) < 1e-4,
            "then the other motion again"
        );
    }
}

/// (h) With an ease-in its weight is still 0 when it ends: it shows
/// nothing, as in the Legacy Viewer (one update, then the stop).
#[test]
fn a_zero_second_pose_with_an_ease_in_ends_at_once() {
    let rig = rig();
    let id = Uuid::from_u128(9212);
    let pose = keyed(&rig, 6, &[("mFaceJaw", 0.5)], false, 0.0, 0.5, 0.5);
    let get = |_: &Uuid| Some(pose.clone());
    let t0 = Instant::now();
    let at = |s: f64| t0 + Duration::from_secs_f64(s);
    for synchronized in [false, true] {
        let mut c = controller(synchronized);
        let first = signal(id, 1, t0);
        assert!(c.sync(&[first], t0, get).is_empty());
        assert_eq!(c.motions[0].weight, 0.0);
        assert_eq!(c.sync(&[first], at(FRAME), get), [id], "synchronized {synchronized}");
        assert!(c.motions.iter().all(|m| m.weight < 0.01));
        assert!(c.sync(&[first], at(1.0), get).is_empty());
        assert!(c.motions.is_empty());
    }
}

/// (i) A 0 s loop (the hand poses of a mesh body) holds its time at 0 and
/// never ends, also when signalled again.
#[test]
fn a_zero_second_loop_keeps_playing() {
    let rig = rig();
    let id = Uuid::from_u128(9213);
    let pose = keyed(&rig, 6, &[("mFaceJaw", 0.5)], true, 0.0, 0.0, 0.0);
    let get = |_: &Uuid| Some(pose.clone());
    let t0 = Instant::now();
    let at = |s: f64| t0 + Duration::from_secs_f64(s);
    for synchronized in [false, true] {
        let mut c = controller(synchronized);
        let mut sig = signal(id, 1, t0);
        for (n, s) in (1..).zip([0.0, FRAME, 1.0, 5.0, 10.0]) {
            if s == 5.0 {
                sig = again(sig, n, at(s));
            }
            assert!(c.sync(&[sig], at(s), get).is_empty(), "synchronized {synchronized}, {s} s");
            assert_eq!(c.motions.len(), 1);
            assert!(c.motions[0].stop.is_none());
            assert_eq!(c.motions[0].weight, 1.0);
            assert_eq!(c.motions[0].inserted, t0, "kept, not started again");
            assert!(off(&mut c, &rig, at(s), "mFaceJaw", 0.5) < 1e-4);
        }
    }
}
