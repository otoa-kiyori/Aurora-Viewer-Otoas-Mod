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
//!   rule's controls, they never ride;
//! - a keyframed boat (not physical, moving with a velocity, as a
//!   keyframed motion is reported): rides with `ride`, never calms the water;
//! - a pier crossing the water: static, the water calms around it.
//!
//! Words, comma separated: a number, Ocean Swell Down (m, 0 to 10; 3 when
//! there is none); `up=<m>`: Ocean Swell Up (0 to 10, 0 when absent);
//! `calm=<m>`: Calm distance (0 to 100, 100 when absent); `edge=<share>`:
//! Calm at the edge (0 to 1, 0 when absent); `ride`: "Objects ride the
//! swell" on; a view: `high` (above the fixtures, looking down), `horizon`
//! (low, toward the far water), `coast` (toward the shore, where the swell
//! calms down).

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
const BOAT: u32 = 99545;
const PIER: u32 = 99546;

#[derive(Debug, PartialEq)]
pub struct Options {
    pub depth: f32,
    pub up: f32,
    pub calm: f32,
    pub edge: f32,
    pub ride: bool,
    pub view: View,
}

fn parse(value: &str) -> Options {
    use aurora_render::water_effects::ocean;
    let mut o = Options {
        depth: 3.0,
        up: 0.0,
        calm: ocean::CALM_REACH_DEFAULT,
        edge: ocean::CALM_EDGE_DEFAULT,
        ride: false,
        view: View::Near,
    };
    let number = |word: &str, name: &str| word.strip_prefix(name).and_then(|v| v.parse::<f32>().ok());
    for word in value.split(',').map(str::trim) {
        if word == "ride" {
            o.ride = true;
        } else if word == "high" {
            o.view = View::High;
        } else if word == "horizon" {
            o.view = View::Horizon;
        } else if word == "coast" {
            o.view = View::Coast;
        } else if let Some(up) = number(word, "up=") {
            o.up = ocean::up(up);
        } else if let Some(calm) = number(word, "calm=") {
            o.calm = ocean::calm_reach(calm);
        } else if let Some(edge) = number(word, "edge=") {
            o.edge = ocean::calm_edge(edge);
        } else if let Ok(depth) = word.parse::<f32>() {
            o.depth = ocean::depth(depth);
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
    settings.water_effects.ocean_up = o.up;
    settings.water_effects.calm_reach = o.calm;
    settings.water_effects.calm_edge = o.edge;
    settings.water_effects.ocean_ride = o.ride;
    log::info!(
        "ocean demo: swell down {} m, up {} m, calm {} m at {}, objects ride {}",
        o.depth,
        o.up,
        o.calm,
        o.edge,
        o.ride
    );
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
    // the boat goes back and forth along 40 m at 0.6 m/s
    let leg = (time * 0.6) % 80.0;
    let (boat_x, boat_v) = if leg < 40.0 { (-280.0 + leg, 0.6) } else { (-200.0 - leg, -0.6) };
    for (id, pos, scale, color, physical, velocity) in [
        (
            RAFT,
            Vec3::new(-262.0, 140.0, 20.0),
            Vec3::new(3.0, 1.6, 0.6),
            [0.85, 0.55, 0.2, 1.0],
            true,
            0.0,
        ),
        (
            STILL,
            Vec3::new(-262.0, 152.0, 20.0),
            Vec3::new(1.5, 1.5, 0.6),
            [0.2, 0.6, 0.9, 1.0],
            false,
            0.0,
        ),
        (
            AIRBORNE,
            Vec3::new(-262.0, 128.0, 24.0),
            Vec3::new(1.5, 1.5, 0.6),
            [0.9, 0.2, 0.3, 1.0],
            true,
            0.0,
        ),
        (
            BOAT,
            Vec3::new(boat_x, 164.0, 20.0),
            Vec3::new(3.0, 1.4, 0.8),
            [0.9, 0.9, 0.9, 1.0],
            false,
            boat_v,
        ),
        (
            PIER,
            Vec3::new(-236.0, 150.0, 19.0),
            Vec3::new(2.0, 24.0, 3.0),
            [0.55, 0.45, 0.35, 1.0],
            false,
            0.0,
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
        prim.velocity = Vec3::X * velocity;
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
    fn the_words_give_the_swell_the_calming_and_the_riding() {
        let o = |depth, ride, view| Options {
            depth,
            up: 0.0,
            calm: 100.0,
            edge: 0.0,
            ride,
            view,
        };
        assert_eq!(parse(""), o(3.0, false, View::Near));
        assert_eq!(parse("5,ride"), o(5.0, true, View::Near));
        assert_eq!(parse("ride, 1.5,high"), o(1.5, true, View::High));
        assert_eq!(parse("12,horizon"), o(10.0, false, View::Horizon));
        assert_eq!(parse("0,coast"), o(0.0, false, View::Coast));
        assert_eq!(
            parse("2,up=4,calm=30,edge=0.25"),
            Options {
                up: 4.0,
                calm: 30.0,
                edge: 0.25,
                ..o(2.0, false, View::Near)
            }
        );
        assert_eq!(parse("up=40,calm=-3,edge=2").up, 10.0);
        assert_eq!(parse("calm=-3").calm, 0.0);
        assert_eq!(parse("edge=2").edge, 1.0);
    }

    #[test]
    fn the_fixtures_come_only_with_the_ocean_scenario() {
        let mut list = Vec::new();
        objects("medium:shore", 0.0, &mut list);
        assert!(list.is_empty());
        objects("medium:ocean", 0.0, &mut list);
        assert_eq!(list.len(), 7);
        let boat = list.iter().find(|o| o.local_id == BOAT).expect("boat");
        assert_eq!(boat.update_flags & PHYSICS, 0);
        assert!(boat.velocity.length() > 0.5, "the boat moves as a keyframed motion is reported");
        let pier = list.iter().find(|o| o.local_id == PIER).expect("pier");
        assert_eq!((pier.update_flags & PHYSICS, pier.velocity), (0, Vec3::ZERO));
        let raft = list.iter().find(|o| o.local_id == RAFT).expect("raft");
        assert_eq!(raft.update_flags & PHYSICS, PHYSICS);
        let still = list.iter().find(|o| o.local_id == STILL).expect("still");
        assert_eq!(still.update_flags & PHYSICS, 0);
        let rider = list.iter().find(|o| o.local_id == RIDER).expect("rider");
        assert_eq!(rider.parent_id, RAFT);
    }
}
