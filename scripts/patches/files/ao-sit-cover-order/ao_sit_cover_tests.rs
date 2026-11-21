// aurora-local-patch: ao-sit-cover-order-blend-tests
//! The tie rule the AO's sit cover order relies on (ao::runtime puts
//! ANIM_AGENT_SIT_GENERIC before the AO's sit): of two motions of a priority
//! that Controller::sync puts in together, the one listed last is the newest
//! and has the joint (LLJointStateBlender::addJointState,
//! indra/llcharacter/llpose.cpp). A child of `anim`, like its `tests`.

use super::*;

fn rig() -> Rig {
    let xml = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/character/avatar_skeleton.xml")).expect("skeleton");
    Rig::new(&Skeleton::parse(&xml).expect("parse"))
}

fn signal(id: Uuid, start: Instant) -> PlayingAnimation {
    PlayingAnimation {
        id,
        sequence: 0,
        sequence_start: start,
        continuous_start: start,
    }
}

/// A looping one-key sit pose that turns the jaw to `angle` at `priority`.
fn pose(rig: &Rig, priority: i32, angle: f32) -> Arc<BoundAnim> {
    let anim = Animation {
        base_priority: priority,
        duration: 1.0,
        emote_name: String::new(),
        loop_in: 0.0,
        loop_out: 1.0,
        looping: true,
        ends_after: None,
        ease_in: 0.0,
        ease_out: 0.0,
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

/// The jaw's distance (radians) from the pose of `angle` after evaluating at `now`.
fn jaw_off(c: &mut Controller, rig: &Rig, now: Instant, angle: f32) -> f32 {
    let mut out = vec![Mat4::IDENTITY.to_cols_array_2d(); PALETTE_JOINTS];
    c.evaluate(rig, now, None, &mut out);
    c.rot[rig.names["mFaceJaw"]].angle_between(Quat::from_rotation_z(angle))
}

#[test]
fn of_two_motions_of_a_priority_put_in_together_the_last_listed_wins() {
    let rig = rig();
    let (cover, sit) = (crate::ao::SIT_GENERIC, Uuid::from_u128(9401));
    let (cover_angle, sit_angle) = (-0.4, 0.6);
    let cover_clip = pose(&rig, 3, cover_angle);
    let sit_clip = pose(&rig, 3, sit_angle);
    let get = |id: &Uuid| Some(if *id == cover { cover_clip.clone() } else { sit_clip.clone() });
    let t0 = Instant::now();
    let later = t0 + Duration::from_secs(1);
    // the AO's order: the cover first, then the AO's sit
    let mut c = Controller::default();
    let listed = [signal(cover, t0), signal(sit, t0)];
    c.sync(&listed, t0, get);
    c.sync(&listed, later, get);
    assert!(jaw_off(&mut c, &rig, later, sit_angle) < 1e-4, "the AO's sit has the joint");
    // the old order: the cover listed last took the joint
    let mut c = Controller::default();
    let listed = [signal(sit, t0), signal(cover, t0)];
    c.sync(&listed, t0, get);
    c.sync(&listed, later, get);
    assert!(
        jaw_off(&mut c, &rig, later, cover_angle) < 1e-4,
        "the cover listed last wins the tie"
    );
}
