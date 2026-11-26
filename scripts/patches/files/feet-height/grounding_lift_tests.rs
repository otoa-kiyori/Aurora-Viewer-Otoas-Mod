// aurora-local-patch: feet-height-tests
//! Feet height (Preferences › Avatars): the grounded avatar moves as a whole
//! by the shift, at the grounding's own pace, and not at all without it.

use super::*;

struct Fixture {
    rig: Arc<Rig>,
    base: SkeletonBase,
    floor: f32,
}

impl Fixture {
    fn new() -> Self {
        let rig = crate::scene::avatar::AvatarLibrary::load().rig.clone();
        let base = SkeletonBase {
            local_pos: rig.local_pos.clone(),
            scale: rig.scale.clone(),
        };
        let floor = Setup::new(&rig, &base, None).expect("humanoid").floor;
        Self { rig, base, floor }
    }

    /// Standing on flat ground at z = 0, turned a little.
    fn frame(&self, lift: f32) -> Frame {
        Frame {
            to_world: Mat4::from_rotation_translation(Quat::from_rotation_z(0.4), -Vec3::Z * self.floor),
            contact: None,
            velocity_z: 0.0,
            sole_height: None,
            max_height: crate::settings::DEFAULT_GROUND_FEET_HEIGHT,
            standing_anchor: false,
            lift,
        }
    }

    fn sample(&self, placement: &mut Placement, frame: Option<Frame>, at: Instant) -> Option<Placed> {
        placement.apply(
            &self.rig,
            &self.base,
            Pose {
                rotations: &self.rig.local_rot,
                positions: &self.base.local_pos,
                pelvis: Vec3::ZERO,
                reference_offset: Vec3::ZERO,
                allowed: [true; 2],
            },
            frame,
            at,
            &flat,
        )
    }

    /// Every joint in world space, as drawn (no correction: the source pose).
    fn joints(&self, placed: Option<&Placed>, frame: Frame) -> Vec<Mat4> {
        let (rotations, pelvis) = placed.map_or((self.rig.local_rot.as_slice(), Vec3::ZERO), |placed| {
            (placed.rotations.as_slice(), placed.pelvis)
        });
        self.rig
            .world_with(rotations, pelvis, &self.base.local_pos, &self.base.scale)
            .into_iter()
            .map(|matrix| frame.to_world * matrix)
            .collect()
    }

    /// The joints that hang from the pelvis (`rig.order` has parents first).
    fn body(&self) -> Vec<bool> {
        let mut body = vec![false; self.rig.len()];
        for &joint in &self.rig.order {
            body[joint] = joint == self.rig.pelvis || self.rig.parent[joint].is_some_and(|parent| body[parent]);
        }
        assert!(body.iter().filter(|b| **b).count() > 100, "the body is most of the skeleton");
        body
    }

    /// The lowest corner of each sole, in world space.
    fn soles(&self, placement: &Placement, joints: &[Mat4]) -> [f32; 2] {
        placement.setup.as_ref().expect("setup").legs.map(|leg| {
            let ankle = joints[leg.joints[2]];
            footprint(ankle.w_axis.truncate(), orientation(ankle), leg.sole)
                .into_iter()
                .map(|corner| corner.z)
                .fold(f32::INFINITY, f32::min)
        })
    }

    /// Settles a placement for two seconds at 60 frames per second.
    fn settle(&self, placement: &mut Placement, lift: f32, from: Instant) -> (Option<Placed>, Instant) {
        let mut placed = None;
        let mut at = from;
        for i in 0..=120 {
            at = from + Duration::from_secs_f32(i as f32 / 60.0);
            placed = self.sample(placement, Some(self.frame(lift)), at);
        }
        (placed, at)
    }
}

fn flat(p: Vec3) -> Option<Surface> {
    Some(Surface {
        position: p.with_z(0.0),
        normal: Vec3::Z,
    })
}

#[test]
fn feet_height_zero_changes_nothing_and_a_shift_back_to_zero_leaves_no_trace() {
    let f = Fixture::new();
    let now = Instant::now();
    let mut untouched = Placement::default();
    let mut shifted = Placement::default();
    let (expected, at) = f.settle(&mut untouched, 0.0, now);
    let expected = expected.expect("grounded");
    // a negative zero, a non-number and a zero are all no shift, bit for bit
    for lift in [0.0, -0.0, f32::NAN] {
        let mut placement = Placement::default();
        let (placed, _) = f.settle(&mut placement, lift, now);
        let placed = placed.expect("grounded");
        assert_eq!(placed.pelvis, expected.pelvis, "{lift}");
        assert_eq!(placed.rotations, expected.rotations, "{lift}");
    }
    // up by 10 cm, then back to zero: the same pose as never moved
    let (_, later) = f.settle(&mut shifted, 0.1, now);
    let (back, _) = f.settle(&mut shifted, 0.0, later + Duration::from_millis(16));
    let back = back.expect("grounded");
    let (expected, _) = f.settle(&mut untouched, 0.0, at + Duration::from_millis(16));
    let expected = f.joints(expected.as_ref(), f.frame(0.0));
    for (joint, (a, b)) in f.joints(Some(&back), f.frame(0.0)).iter().zip(&expected).enumerate() {
        assert!(
            a.w_axis.distance(b.w_axis) < 2e-4,
            "joint {joint} kept {} m of the old shift",
            a.w_axis.distance(b.w_axis)
        );
    }
}

#[test]
fn feet_height_moves_the_whole_grounded_avatar_by_the_shift() {
    let f = Fixture::new();
    let now = Instant::now();
    let mut reference = Placement::default();
    let (unshifted, _) = f.settle(&mut reference, 0.0, now);
    let unshifted_joints = f.joints(unshifted.as_ref(), f.frame(0.0));
    let unshifted_soles = f.soles(&reference, &unshifted_joints);
    for sole in unshifted_soles {
        assert!(sole.abs() < 0.002, "the unshifted sole stands on the ground: {sole}");
    }
    for lift in [0.1, 0.05, 0.01, -0.01, -0.05, -0.1] {
        let mut placement = Placement::default();
        let (placed, _) = f.settle(&mut placement, lift, now);
        let placed = placed.expect("grounded");
        let joints = f.joints(Some(&placed), f.frame(lift));
        // the pelvis and everything under it (legs, head, the attachment
        // points and what rigged mesh follows): straight up or down by the
        // shift; the skeleton's root above it (Avatar Center) stays, as it
        // does for any grounding correction
        let body = f.body();
        for (joint, (a, b)) in joints.iter().zip(&unshifted_joints).enumerate() {
            let moved = a.w_axis.truncate() - b.w_axis.truncate();
            let expected = if body[joint] { Vec3::Z * lift } else { Vec3::ZERO };
            assert!(
                (moved - expected).length() < 0.001,
                "joint {joint} moved {moved:?} for a {lift} m shift"
            );
            assert!(
                orientation(*a).angle_between(orientation(*b)) < 0.002,
                "joint {joint} turned for a {lift} m shift"
            );
        }
        for (sole, unshifted) in f.soles(&placement, &joints).into_iter().zip(unshifted_soles) {
            assert!(
                (sole - unshifted - lift).abs() < 0.001,
                "the sole stands at {sole} m for a {lift} m shift"
            );
        }
        // only the vertical: the pose-space offset is the shift
        assert!((placed.pelvis.z - lift).abs() < 0.0015, "{lift}: pelvis {:?}", placed.pelvis);
    }
}

#[test]
fn feet_height_eases_in_and_out_with_the_grounding() {
    let f = Fixture::new();
    let now = Instant::now();
    let lift = 0.1;
    let mut placement = Placement::default();
    let step = Duration::from_secs_f32(1.0 / 60.0);
    // not grounded yet (walking with "while moving" off, sitting, flying):
    // no shift at all
    for i in 0..30 {
        assert!(f.sample(&mut placement, None, now + step * i).is_none());
    }
    // grounding starts: from the drawn pose, no pop, settled within twice
    // the grounding's transition
    let start = now + step * 30;
    let mut last = 0.0_f32;
    let mut settled_at = None;
    for i in 0..=120 {
        let at = start + step * i;
        let placed = f.sample(&mut placement, Some(f.frame(lift)), at).expect("grounded");
        let height = placed.pelvis.z;
        if i == 0 {
            assert!(height.abs() < 0.005, "the first grounded frame jumped to {height} m");
        }
        assert!((height - last).abs() < 0.01, "a {} m pop at frame {i}", height - last);
        if settled_at.is_none() && (height - lift).abs() < 0.003 {
            settled_at = Some(i);
        }
        last = height;
    }
    let settled_at = settled_at.expect("the shift settles");
    assert!(
        settled_at as f32 / 60.0 <= TRANSITION_SECONDS * 2.0,
        "settled after {settled_at} frames"
    );
    assert!((last - lift).abs() < 0.0015, "full shift once settled: {last}");
    // grounding stops: the shift fades with the release weight, then nothing
    let held = last;
    let stop = start + step * 121;
    for i in 0..=60 {
        let at = stop + step * i;
        let placed = f.sample(&mut placement, None, at);
        let weight = 1.0 - cubic_step((step * i).as_secs_f32() / TRANSITION_SECONDS);
        match placed {
            Some(placed) => {
                assert!(
                    (placed.pelvis.z - held * weight).abs() < 1e-4,
                    "frame {i}: {} m at weight {weight}",
                    placed.pelvis.z
                );
                assert!((placed.pelvis.z - last).abs() < 0.01, "a pop while releasing at frame {i}");
                last = placed.pelvis.z;
            }
            None => assert!(weight <= 0.0, "released early at weight {weight}"),
        }
    }
    assert!(f.sample(&mut placement, None, stop + step * 61).is_none(), "no shift once released");
}
