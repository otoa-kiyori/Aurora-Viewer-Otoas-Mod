//! Cross-region voice (local patch set, 112-voice-neighbours): on the estate
//! (region-wide) channel the viewer keeps a voice connection to the agent's
//! region, the primary one, and one to each neighbouring region within reach,
//! so that residents on the other side of a region border are heard. This
//! module holds the pure rules; `crates/aurora-viewer/src/voice.rs` owns the
//! connections.
//!
//! - which regions: port of LLWebRTCVoiceClient::updateNeighboringRegions
//!   (indra/newview/llvoicewebrtc.cpp, originally LGPL 2.1): the regions
//!   under eight points at 2 * MAX_AUDIO_DIST (100 m) around the speaker
//!   (the avatar's head), since the camera ear can stray 50 m away;
//! - the microphone: LLVoiceWebRTCSpatialConnection::setMuteMic and the
//!   VOICE_STATE_SESSION_UP check of
//!   LLVoiceWebRTCConnection::connectionStateMachine
//!   (`setMute(mMuted || !mPrimary)`): a connection to another region than
//!   the agent's never carries the microphone, peers there would hear the
//!   agent twice;
//! - the join: LLVoiceWebRTCConnection::sendJoin (`{"j":{"p":true}}` on the
//!   primary connection, `{"j":{}}` elsewhere), sent again when the primary
//!   state changes;
//! - the participants: one list for the channel, as the Legacy Viewer keeps
//!   one per session (LLWebRTCVoiceClient::addParticipantByID and
//!   removeParticipantByID): each connection only lists the residents whose
//!   primary server it is (ParticipantTable::apply), and the lists are
//!   merged here.
//!
//! Copyright (C) 2023, Linden Research, Inc.
//! Licensed under the GNU Lesser General Public License, version 2.1.
// aurora-local-patch: voice-neighbours-module

use std::collections::{BTreeMap, BTreeSet};

use aurora_audio::MicCapture;
use glam::DVec3;

use crate::Participant;
use crate::protocol::{self, MAX_AUDIO_DIST};

/// The eight directions of `updateNeighboringRegions`, with its rounded
/// diagonals (0.707).
pub const NEIGHBOUR_DIRECTIONS: [(f64, f64); 8] = [
    (0.0, 1.0),
    (0.707, 0.707),
    (1.0, 0.0),
    (0.707, -0.707),
    (0.0, -1.0),
    (-0.707, -0.707),
    (-1.0, 0.0),
    (-0.707, 0.707),
];

/// How far from the speaker the eight points are: every region within
/// 100 m, because the listener may be tethered up to 50 m away
/// (`2 * MAX_AUDIO_DIST`).
pub const NEIGHBOUR_REACH: f64 = 2.0 * MAX_AUDIO_DIST;

/// The ground a region covers, in global metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegionArea {
    pub handle: u64,
    pub origin_x: f64,
    pub origin_y: f64,
    pub width: f64,
    pub depth: f64,
}

impl RegionArea {
    /// A region from its handle (global origin in its high and low 32 bits)
    /// and its size in metres.
    pub fn from_handle(handle: u64, width: f64, depth: f64) -> Self {
        Self {
            handle,
            origin_x: (handle >> 32) as f64,
            origin_y: (handle & 0xffff_ffff) as f64,
            width,
            depth,
        }
    }

    /// LLWorld::getRegionFromPosGlobal: the origin edge belongs to the
    /// region, the far edge to the next one.
    fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.origin_x && x < self.origin_x + self.width && y >= self.origin_y && y < self.origin_y + self.depth
    }
}

/// The regions other than `own` that lie under one of the eight points
/// [`NEIGHBOUR_REACH`] away from `speaker` (global metres), among the
/// regions the viewer is connected to. The Legacy Viewer's set also holds
/// the agent's region; here that one is the primary connection, kept apart.
pub fn neighbouring_regions(own: u64, speaker: DVec3, regions: &[RegionArea]) -> BTreeSet<u64> {
    let mut out = BTreeSet::new();
    if !speaker.is_finite() {
        return out;
    }
    for (dx, dy) in NEIGHBOUR_DIRECTIONS {
        let (x, y) = (speaker.x + NEIGHBOUR_REACH * dx, speaker.y + NEIGHBOUR_REACH * dy);
        if let Some(r) = regions.iter().find(|r| r.contains(x, y))
            && r.handle != own
        {
            out.insert(r.handle);
        }
    }
    out
}

/// What the sender of a connection takes this frame: the microphone and
/// the transmit switch on the primary connection, nothing on any other
/// (the muted track's silence is sent there, as libwebrtc does for a
/// muted track). A neighbour connection does not even read the capture,
/// which must have a single reader.
pub(crate) fn outgoing(primary: bool, transmit: bool, mic: Option<MicCapture>) -> (Option<MicCapture>, bool) {
    if primary {
        (mic, transmit)
    } else {
        (None, false)
    }
}

/// The join to send on a spatial connection: always once the data channel
/// opens (`sent` is `None`), then again whenever the primary state is no
/// longer the one the server was told.
pub(crate) fn rejoin(sent: Option<bool>, primary: bool) -> Option<String> {
    (sent != Some(primary)).then(|| protocol::join_message(primary))
}

/// The role of a connection in its `voice stats` log line.
pub(crate) fn role(spatial: bool, primary: bool) -> &'static str {
    match (spatial, primary) {
        (false, _) => "call",
        (true, true) => "primary",
        (true, false) => "neighbour",
    }
}

/// The `voice stats` line of one connection, tagged with its role.
pub(crate) fn tag_stats(line: &str, role: &str) -> String {
    match line.strip_prefix("voice stats:") {
        Some(rest) => format!("voice stats ({role}):{rest}"),
        None => format!("({role}) {line}"),
    }
}

/// The channel's participants from the lists of its connections, the
/// primary connection's first. A resident listed by several connections
/// (around a region change, until the old server reports the leave) is
/// listed once: the loudest level, speaking when any connection says so,
/// and the moderator mute of the first list that has the resident, so the
/// primary server's wins (OnDataReceivedImpl ignores the mute flags of a
/// non-primary server).
pub fn merge_participants<I>(lists: I) -> Vec<Participant>
where
    I: IntoIterator<Item = Vec<Participant>>,
{
    let mut merged: BTreeMap<uuid::Uuid, Participant> = BTreeMap::new();
    for list in lists {
        for p in list {
            match merged.get_mut(&p.agent_id) {
                Some(seen) => {
                    seen.level = seen.level.max(p.level);
                    seen.speaking |= p.speaking;
                }
                None => {
                    merged.insert(p.agent_id, p);
                }
            }
        }
    }
    merged.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    const SIDE: u64 = 256;

    /// A handle of the region at grid cell (x, y).
    fn cell(x: u64, y: u64) -> u64 {
        ((x * SIDE) << 32) | (y * SIDE)
    }

    /// The 3 x 3 regions around cell (10, 10).
    fn grid() -> Vec<RegionArea> {
        let mut v = Vec::new();
        for x in 9..=11 {
            for y in 9..=11 {
                v.push(RegionArea::from_handle(cell(x, y), SIDE as f64, SIDE as f64));
            }
        }
        v
    }

    fn at(local_x: f64, local_y: f64) -> DVec3 {
        DVec3::new((10 * SIDE) as f64 + local_x, (10 * SIDE) as f64 + local_y, 25.0)
    }

    #[test]
    fn the_middle_of_a_region_has_no_neighbour_in_reach() {
        let own = cell(10, 10);
        assert!(neighbouring_regions(own, at(128.0, 128.0), &grid()).is_empty());
    }

    #[test]
    fn an_edge_reaches_the_region_beyond_it_within_100_m() {
        let own = cell(10, 10);
        // 30 m from the east edge: the east region only
        assert_eq!(neighbouring_regions(own, at(226.0, 128.0), &grid()), BTreeSet::from([cell(11, 10)]));
        // 30 m from the south edge: the south region only
        assert_eq!(neighbouring_regions(own, at(128.0, 30.0), &grid()), BTreeSet::from([cell(10, 9)]));
        // just over 100 m from the west edge: nothing; just under: the west region
        assert!(neighbouring_regions(own, at(100.5, 128.0), &grid()).is_empty());
        assert_eq!(neighbouring_regions(own, at(99.5, 128.0), &grid()), BTreeSet::from([cell(9, 10)]));
        // the far edge belongs to the next region: exactly 100 m from the
        // east edge reaches it
        assert_eq!(neighbouring_regions(own, at(156.0, 128.0), &grid()), BTreeSet::from([cell(11, 10)]));
    }

    #[test]
    fn a_corner_reaches_the_three_regions_around_it() {
        let own = cell(10, 10);
        assert_eq!(neighbouring_regions(own, at(250.0, 250.0), &grid()), BTreeSet::from([cell(10, 11), cell(11, 11), cell(11, 10)]));
        assert_eq!(neighbouring_regions(own, at(5.0, 5.0), &grid()), BTreeSet::from([cell(9, 10), cell(9, 9), cell(10, 9)]));
        // 60 m from both edges: the straight points (100 m) reach both
        // sides, the diagonal one (70.7 m on each axis) reaches the corner
        // region too
        assert_eq!(neighbouring_regions(own, at(196.0, 196.0), &grid()), BTreeSet::from([cell(10, 11), cell(11, 11), cell(11, 10)]));
        // 80 m from both edges: the straight points reach both sides, the
        // diagonal one falls short of the corner region
        assert_eq!(neighbouring_regions(own, at(176.0, 176.0), &grid()), BTreeSet::from([cell(10, 11), cell(11, 10)]));
    }

    #[test]
    fn only_connected_regions_are_chosen_and_never_the_own_one() {
        let own = cell(10, 10);
        // no region known beyond the corner: no connection to make there
        let known: Vec<_> = grid().into_iter().filter(|r| r.handle != cell(11, 11)).collect();
        assert_eq!(neighbouring_regions(own, at(250.0, 250.0), &known), BTreeSet::from([cell(10, 11), cell(11, 10)]));
        // a region alone: nothing, whatever the position
        let alone = [RegionArea::from_handle(own, SIDE as f64, SIDE as f64)];
        assert!(neighbouring_regions(own, at(1.0, 1.0), &alone).is_empty());
        assert!(neighbouring_regions(own, DVec3::NAN, &grid()).is_empty());
    }

    #[test]
    fn a_larger_region_is_found_by_its_size() {
        let own = cell(10, 10);
        // a 512 m region to the south-east, origin at cell (11, 9): the
        // east and south-east points both fall in it
        let big = RegionArea::from_handle(cell(11, 9), 512.0, 512.0);
        let regions = [RegionArea::from_handle(own, SIDE as f64, SIDE as f64), big];
        assert_eq!(neighbouring_regions(own, at(250.0, 200.0), &regions), BTreeSet::from([big.handle]));
    }

    #[test]
    fn a_neighbour_connection_never_sends_the_microphone() {
        let (_feed, mic) = MicCapture::virtual_input();
        // primary: the microphone and the switch as they are
        let (m, t) = outgoing(true, true, Some(mic.clone()));
        assert!(m.is_some() && t);
        let (m, t) = outgoing(true, false, Some(mic.clone()));
        assert!(m.is_some() && !t);
        // a neighbour: neither, even while transmitting
        for transmit in [false, true] {
            let (m, t) = outgoing(false, transmit, Some(mic.clone()));
            assert!(m.is_none() && !t, "a neighbour connection sends silence only");
        }
        assert!(!outgoing(false, true, None).1);
    }

    #[test]
    fn the_join_says_primary_and_is_sent_again_on_a_hand_over() {
        // a neighbour connection's first join
        assert_eq!(rejoin(None, false).as_deref(), Some(r#"{"j":{}}"#));
        // unchanged: nothing more
        assert_eq!(rejoin(Some(false), false), None);
        // the agent crossed into its region: the join again, primary
        assert_eq!(rejoin(Some(false), true).as_deref(), Some(r#"{"j":{"p":true}}"#));
        assert_eq!(rejoin(Some(true), true), None);
        // crossed back out: a neighbour again
        assert_eq!(rejoin(Some(true), false).as_deref(), Some(r#"{"j":{}}"#));
        // the primary connection's first join, as upstream sends it
        assert_eq!(rejoin(None, true).as_deref(), Some(r#"{"j":{"p":true}}"#));
    }

    #[test]
    fn the_stats_line_names_the_connection() {
        assert_eq!(role(true, true), "primary");
        assert_eq!(role(true, false), "neighbour");
        assert_eq!(role(false, true), "call");
        assert_eq!(tag_stats("voice stats: rx 1 packets", "neighbour"), "voice stats (neighbour): rx 1 packets");
        assert_eq!(tag_stats("other", "primary"), "(primary) other");
    }

    fn p(id: u128, level: f32, speaking: bool, muted: bool) -> Participant {
        Participant {
            agent_id: Uuid::from_u128(id),
            level,
            speaking,
            moderator_muted: muted,
        }
    }

    #[test]
    fn participants_of_all_connections_are_listed_once() {
        let primary = vec![p(1, 0.2, false, false), p(2, 0.1, false, true)];
        let east = vec![p(3, 0.5, true, false), p(2, 0.6, true, false)];
        let north = vec![p(4, 0.0, false, true)];
        let merged = merge_participants([primary, east, north]);
        assert_eq!(merged.len(), 4);
        assert_eq!(merged[0], p(1, 0.2, false, false));
        // listed by two servers: loudest level, speaking, the primary's mute
        assert_eq!(merged[1], p(2, 0.6, true, true));
        assert_eq!(merged[2], p(3, 0.5, true, false));
        assert_eq!(merged[3], p(4, 0.0, false, true));
        assert!(merge_participants(Vec::<Vec<Participant>>::new()).is_empty());
    }
}
