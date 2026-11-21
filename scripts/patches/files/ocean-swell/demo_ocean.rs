//! AURORA_DEMO_OCEAN: the open-sea swell in the water demo (local patch
// aurora-local-patch: ocean-swell-demo-module
//! set, ocean_swell.rs), with `AURORA_DEMO_WATER=ocean`. The camera looks
//! west over the void water beyond the demo region: no land for more than
//! a kilometre ahead, the coast about 300 m behind. The fixtures float 260 m
//! out, west of the region (region coordinates below 0):
//! - a physical box at the surface, a demo avatar sitting on it: both ride
//!   the swell with `ride`;
//! - a demo avatar swimming at the surface: rides with `ride`;
//! - a non-physical box at the surface and a physical box in the air: the
//!   rule's controls, they never ride.
//!
//! Words, comma separated: a number, the swell's depth (m, 0 to 5; 3 when
//! there is none); `ride`: "Objects ride the swell" on; a view: `high`
//! (above the fixtures, looking down), `horizon` (low, toward the far
//! water), `coast` (toward the shore, where the swell must fade out).

use aurora_net::objects::ObjectUpdate;
use glam::{Quat, Vec3};
use std::sync::OnceLock;
use uuid::Uuid;

/// Eye and target of `AURORA_DEMO_WATER=ocean`, by view word.
const VIEW: (Vec3, Vec3) = (Vec3::new(-205.0, 118.0, 24.0), Vec3::new(-262.0, 146.0, 19.5));
const VIEW_HIGH: (Vec3, Vec3) = (Vec3::new(-215.0, 105.0, 60.0), Vec3::new(-262.0, 142.0, 19.0));
const VIEW_HORIZON: (Vec3, Vec3) = (Vec3::new(-240.0, 130.0, 21.5), Vec3::new(-640.0, 150.0, 19.0));
const VIEW_COAST: (Vec3, Vec3) = (Vec3::new(-320.0, 128.0, 30.0), Vec3::new(-40.0, 128.0, 19.0));

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum View {
    Near,
    High,
    Horizon,
    Coast,
}

/// This run's camera (the near view without the variable).
pub fn view() -> (Vec3, Vec3) {
    match options().map_or(View::Near, |o| o.view) {
        View::Near => VIEW,
        View::High => VIEW_HIGH,
        View::Horizon => VIEW_HORIZON,
        View::Coast => VIEW_COAST,
    }
}

/// FLAGS_USE_PHYSICS of the update flags.
const PHYSICS: u32 = 1;
/// The fixtures' local ids (the water demo's own run from 99500).
const RAFT: u32 = 99540;
const STILL: u32 = 99541;
const AIRBORNE: u32 = 99542;
const SWIMMER: u32 = 99543;
const RIDER: u32 = 99544;

#[derive(Debug, PartialEq)]
pub struct Options {
    pub depth: f32,
    pub ride: bool,
    pub view: View,
}

fn parse(value: &str) -> Options {
    let mut o = Options {
        depth: 3.0,
        ride: false,
        view: View::Near,
    };
    for word in value.split(',').map(str::trim) {
        if word == "ride" {
            o.ride = true;
        } else if word == "high" {
            o.view = View::High;
        } else if word == "horizon" {
            o.view = View::Horizon;
        } else if word == "coast" {
            o.view = View::Coast;
        } else if let Ok(depth) = word.parse::<f32>() {
            o.depth = aurora_render::water_effects::ocean::depth(depth);
        }
    }
    o
}

fn options() -> Option<&'static Options> {
    static OPTIONS: OnceLock<Option<Options>> = OnceLock::new();
    OPTIONS
        .get_or_init(|| std::env::var("AURORA_DEMO_OCEAN").ok().map(|v| parse(&v)))
        .as_ref()
}

/// The options into this run's settings (the water demo's setup).
pub fn configure(settings: &mut crate::settings::Settings) {
    let Some(o) = options() else { return };
    settings.water_effects.waves = true;
    settings.water_effects.ocean_depth = o.depth;
    settings.water_effects.ocean_ride = o.ride;
    log::info!("ocean demo: swell depth {} m, objects ride {}", o.depth, o.ride);
}

/// The fixtures, in the water demo's update list (`mode`: its scenario).
pub fn objects(mode: &str, time: f32, objects: &mut Vec<ObjectUpdate>) {
    if !mode.contains("ocean") {
        return;
    }
    let box_shape = super::shape(
        aurora_prim::params::LL_PCODE_PATH_LINE,
        aurora_prim::params::LL_PCODE_PROFILE_SQUARE,
        100,
        0,
        0,
    );
    for (id, pos, scale, color, physical) in [
        (
            RAFT,
            Vec3::new(-262.0, 140.0, 20.0),
            Vec3::new(3.0, 1.6, 0.6),
            [0.85, 0.55, 0.2, 1.0],
            true,
        ),
        (
            STILL,
            Vec3::new(-262.0, 152.0, 20.0),
            Vec3::new(1.5, 1.5, 0.6),
            [0.2, 0.6, 0.9, 1.0],
            false,
        ),
        (
            AIRBORNE,
            Vec3::new(-262.0, 128.0, 24.0),
            Vec3::new(1.5, 1.5, 0.6),
            [0.9, 0.2, 0.3, 1.0],
            true,
        ),
    ] {
        let mut prim = super::prim(
            id,
            pos,
            Quat::from_rotation_z(0.3),
            scale,
            box_shape,
            super::te(color, 0, false, 0.0),
            Default::default(),
            "",
        );
        if physical {
            prim.update_flags |= PHYSICS;
        }
        objects.push(prim);
    }
    for (id, parent, pos, rotation) in [
        // the swimmer drifts slowly along the coast
        (
            SWIMMER,
            0,
            Vec3::new(-250.0, 134.0 + 3.0 * (time * 0.1).sin(), 20.05),
            Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
        ),
        (RIDER, RAFT, Vec3::Z * 1.25, Quat::IDENTITY),
    ] {
        let mut avatar = super::action_avatar(false);
        avatar.local_id = id;
        avatar.full_id = Uuid::from_u128(id as u128);
        avatar.parent_id = parent;
        avatar.position = pos;
        avatar.rotation = rotation;
        avatar.name_values = "FirstName STRING RW SV Ocean\nLastName STRING RW SV Demo".into();
        objects.push(avatar);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_words_give_the_depth_and_the_riding() {
        let o = |depth, ride, view| Options { depth, ride, view };
        assert_eq!(parse(""), o(3.0, false, View::Near));
        assert_eq!(parse("5,ride"), o(5.0, true, View::Near));
        assert_eq!(parse("ride, 1.5,high"), o(1.5, true, View::High));
        assert_eq!(parse("12,horizon"), o(5.0, false, View::Horizon));
        assert_eq!(parse("0,coast"), o(0.0, false, View::Coast));
    }

    #[test]
    fn the_fixtures_come_only_with_the_ocean_scenario() {
        let mut list = Vec::new();
        objects("medium:shore", 0.0, &mut list);
        assert!(list.is_empty());
        objects("medium:ocean", 0.0, &mut list);
        assert_eq!(list.len(), 5);
        let raft = list.iter().find(|o| o.local_id == RAFT).expect("raft");
        assert_eq!(raft.update_flags & PHYSICS, PHYSICS);
        let still = list.iter().find(|o| o.local_id == STILL).expect("still");
        assert_eq!(still.update_flags & PHYSICS, 0);
        let rider = list.iter().find(|o| o.local_id == RIDER).expect("rider");
        assert_eq!(rider.parent_id, RAFT);
    }
}
