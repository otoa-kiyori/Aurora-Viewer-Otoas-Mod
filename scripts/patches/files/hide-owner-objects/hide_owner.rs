//! Hide the rezzed objects of chosen owners (local patch set). The rule
//! itself, the cache of the owners the region told us, and the owner
//! lookups it asks for while at least one rule exists.
//! // aurora-local-patch: hide-owner-objects-world
//!
//! A rule is an entry of the asset blacklist (`derender.rs`, `Kind::Owner`)
//! keyed by `rule_key(owner)`: never the owner's own id, so a rule cannot
//! collide with a derendered resident or with the owner's sound categories,
//! which share the list's keys. "Owner" is the object's owner id: a
//! resident, or the group for a group-owned (deeded) object, whose owner id
//! is the group's. Every rezzed root (not an avatar, not worn) whose owner
//! has a rule is taken out of the world by the derender machinery, with its
//! linked prims; its updates are dropped while the rule lasts, and removing
//! the rule puts it back at once.
//!
//! Never hidden: what the user owns, the linkset the user sits on, and
//! whatever the user has selected (the build tools edit the selection).
//! The Legacy Viewer has no such rule.
use super::World;
use super::derender::{Entry, Kind};
use super::objects::ObjKey;
use aurora_net::RegionHandle;
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Owner lookups in flight at once (RequestObjectPropertiesFamily, one
/// object each, as a hover asks: LLSelectMgr::requestObjectPropertiesFamily).
pub const IN_FLIGHT: usize = 4;
/// Owner lookups sent per second at most: the rate the Monitor's name
/// lookups keep (`app/monitor.rs`, NAMES_PER_SECOND).
pub const PER_SECOND: usize = 5;
/// A lookup the region did not answer in this time frees its place; that
/// object is not asked again during this visit of its region.
pub const REPLY_WAIT: Duration = Duration::from_secs(10);
/// How often the objects present are walked while a rule exists.
pub const WALK_EVERY: Duration = Duration::from_millis(500);
/// How often the count of lookups sent is logged (when there were any).
pub const LOG_EVERY: Duration = Duration::from_secs(60);

/// Masks an owner id into the key of its rule (and back: XOR).
const RULE_MASK: Uuid = Uuid::from_u128(0x6f77_6e65_7220_7275_6c65_2068_6964_6521);

/// The blacklist key of the rule for `owner`.
pub fn rule_key(owner: Uuid) -> Uuid {
    Uuid::from_u128(owner.as_u128() ^ RULE_MASK.as_u128())
}

/// The owner a rule's key stands for.
pub fn owner_of_key(key: Uuid) -> Uuid {
    rule_key(key)
}

/// What the rule says about one rezzed root.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// Out of the world, under the rule of this key.
    Hide(Uuid),
    /// Stays.
    Keep,
    /// Its owner is not known: ask the region.
    Ask,
}

/// The pure rule. `owner`: the root's owner when known (nil counts as
/// unknown); `me`: the agent; `spared`: sat on or selected.
pub fn decide(owner: Option<Uuid>, me: Uuid, spared: bool, has_rule: impl Fn(Uuid) -> bool) -> Decision {
    if spared || me.is_nil() {
        return Decision::Keep;
    }
    match owner.filter(|o| !o.is_nil()) {
        None => Decision::Ask,
        Some(owner) if owner == me => Decision::Keep,
        Some(owner) if has_rule(owner) => Decision::Hide(rule_key(owner)),
        Some(_) => Decision::Keep,
    }
}

/// The rate of the owner lookups: a few in flight, a budget per second.
#[derive(Default, Debug)]
pub struct Limiter {
    in_flight: HashMap<Uuid, Instant>,
    sent: VecDeque<Instant>,
}

impl Limiter {
    /// Frees the places of the lookups never answered.
    pub fn expire(&mut self, now: Instant) {
        self.in_flight.retain(|_, at| now.saturating_duration_since(*at) < REPLY_WAIT);
        while self
            .sent
            .front()
            .is_some_and(|at| now.saturating_duration_since(*at) >= Duration::from_secs(1))
        {
            self.sent.pop_front();
        }
    }

    pub fn can_send(&self) -> bool {
        self.in_flight.len() < IN_FLIGHT && self.sent.len() < PER_SECOND
    }

    pub fn sent(&mut self, object: Uuid, now: Instant) {
        self.in_flight.insert(object, now);
        self.sent.push_back(now);
    }

    /// An answer came: true when it was one of ours.
    pub fn answered(&mut self, object: &Uuid) -> bool {
        self.in_flight.remove(object).is_some()
    }

    pub fn in_flight(&self) -> usize {
        self.in_flight.len()
    }
}

/// The rule's state beside the derender list (`Derender::owner_rule`).
#[derive(Default)]
pub struct OwnerRule {
    /// The list's generation the rule count below was taken at.
    seen: Option<u64>,
    any: bool,
    /// The agent: its own objects are never hidden.
    pub me: Uuid,
    /// Roots that stay this frame: sat on or selected (set by the app).
    spared: HashSet<ObjKey>,
    /// The owners the region told us, per region visit.
    owners: HashMap<(RegionHandle, Uuid), Uuid>,
    /// The roots already asked for during this region visit.
    asked: HashSet<(RegionHandle, Uuid)>,
    limiter: Limiter,
    next_walk: Option<Instant>,
    /// Lookups sent and answered since the last log line.
    log_since: Option<Instant>,
    log_sent: u32,
    log_answered: u32,
}

impl OwnerRule {
    /// Whether at least one rule exists, counted again when the list changed.
    pub fn any(&mut self, generation: u64, mut entries: impl Iterator<Item = Kind>) -> bool {
        if self.seen != Some(generation) {
            self.seen = Some(generation);
            let any = entries.any(|k| k == Kind::Owner);
            if !any && self.any {
                self.clear();
            }
            self.any = any;
        }
        self.any
    }

    /// No rule left: nothing is kept, nothing is asked.
    fn clear(&mut self) {
        self.spared.clear();
        self.owners.clear();
        self.asked.clear();
        self.limiter = Limiter::default();
        self.next_walk = None;
    }

    pub fn spared(&self, key: &ObjKey) -> bool {
        self.spared.contains(key)
    }

    pub fn set_spared(&mut self, spared: HashSet<ObjKey>) {
        self.spared = spared;
    }

    /// The owner the region told us for this root, if any.
    pub fn owner_of(&self, region: RegionHandle, id: Uuid) -> Option<Uuid> {
        self.owners.get(&(region, id)).copied()
    }

    /// An ObjectProperties(Family) reply: the owner is kept for the visit.
    pub fn learn(&mut self, region: RegionHandle, id: Uuid, owner: Uuid) {
        if self.limiter.answered(&id) {
            self.log_answered += 1;
        }
        if !owner.is_nil() {
            self.owners.insert((region, id), owner);
        }
    }

    /// The region left: its answers and its "asked" marks go with it.
    pub fn forget_region(&mut self, region: RegionHandle) {
        self.owners.retain(|(r, _), _| *r != region);
        self.asked.retain(|(r, _)| *r != region);
        self.spared.retain(|k| k.region != region);
    }

    /// The walk is due (every WALK_EVERY, or at once after `walk_soon`).
    pub fn walk_due(&mut self, now: Instant) -> bool {
        if self.next_walk.is_some_and(|at| now < at) {
            return false;
        }
        self.next_walk = Some(now + WALK_EVERY);
        true
    }

    /// A rule was added or owners became known: walk at the next frame.
    pub fn walk_soon(&mut self) {
        self.next_walk = None;
    }

    /// The lookups to send now, nearest first, each root once per visit,
    /// within the limits. `wanted`: (distance, region, root id).
    pub fn pick(&mut self, now: Instant, mut wanted: Vec<(f32, RegionHandle, Uuid)>) -> Vec<(RegionHandle, Uuid)> {
        self.limiter.expire(now);
        wanted.retain(|(_, r, id)| !self.asked.contains(&(*r, *id)));
        wanted.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut out = Vec::new();
        for (_, region, id) in wanted {
            if !self.limiter.can_send() {
                break;
            }
            self.asked.insert((region, id));
            self.limiter.sent(id, now);
            self.log_sent += 1;
            out.push((region, id));
        }
        out
    }

    /// The periodic log line's numbers, when due and when there were lookups.
    pub fn log_due(&mut self, now: Instant) -> Option<(u32, u32)> {
        let since = *self.log_since.get_or_insert(now);
        if now.saturating_duration_since(since) < LOG_EVERY {
            return None;
        }
        self.log_since = Some(now);
        let counts = (std::mem::take(&mut self.log_sent), std::mem::take(&mut self.log_answered));
        (counts.0 > 0).then_some(counts)
    }
}

impl World {
    /// Whether at least one owner rule exists (no rule: nothing else runs).
    pub fn owner_rules_exist(&mut self) -> bool {
        self.derender.any_owner_rule()
    }

    /// A rule for `owner` (a resident, or the group of a group-owned
    /// object), saved with the blacklist. Never for the agent itself.
    pub fn add_owner_rule(&mut self, owner: Uuid, name: String, group: bool) -> bool {
        if owner.is_nil() || owner == self.agent_id {
            return false;
        }
        let entry = Entry {
            name,
            region: String::new(),
            kind: Kind::Owner,
            date: super::derender::now_secs(),
            permanent: true,
        };
        self.derender.add(rule_key(owner), entry);
        log::info!("hide owner objects: a rule added (group: {group})");
        true
    }

    /// The owner of a rezzed root: from its update (sound or particles),
    /// from the region's answer, or from properties the app holds.
    fn root_owner(&self, idx: usize, props_owner: &impl Fn(&Uuid) -> Option<Uuid>) -> Option<Uuid> {
        let o = self.objects.get(idx)?;
        if !o.owner_id.is_nil() {
            return Some(o.owner_id);
        }
        self.derender
            .owner_rule
            .owner_of(o.key.region, o.full_id)
            .or_else(|| props_owner(&o.full_id))
            .filter(|id| !id.is_nil())
    }

    /// The agent sits on a root an owner rule hid (it came before the
    /// agent's own update, at an arrival): it comes back.
    pub fn owner_rule_free_seat(&mut self) {
        let Some(seat) = self
            .objects
            .index_of_uuid(&self.agent_id)
            .and_then(|i| self.objects.get(i))
            .filter(|a| a.parent_id != 0)
            .map(|a| ObjKey {
                region: a.key.region,
                local_id: a.parent_id,
            })
        else {
            return;
        };
        let restore = self.derender.release_owner_root(&seat);
        if !restore.updates.is_empty() {
            log::info!("hide owner objects: the seat came back");
            self.put_back(restore);
        }
    }

    /// The walk over the rezzed roots present: hides those a rule covers,
    /// and returns the roots whose owner is not known, with their distance.
    pub fn owner_rule_walk(&mut self, props_owner: impl Fn(&Uuid) -> Option<Uuid>) -> Vec<(f32, RegionHandle, Uuid)> {
        let me = self.agent_id;
        let here = self.agent.position;
        let mut hide = Vec::new();
        let mut unknown = Vec::new();
        for (idx, o) in self.objects.iter() {
            if o.parent_id != 0 || o.is_avatar() {
                continue;
            }
            let spared = self.derender.owner_rule.spared(&o.key);
            match decide(self.root_owner(idx, &props_owner), me, spared, |owner| {
                self.derender.has_owner_rule(&owner)
            }) {
                Decision::Hide(key) => hide.push((o.full_id, key)),
                Decision::Ask => {
                    let offset = self.region_offset(o.key.region).unwrap_or_default();
                    unknown.push((here.distance(o.position + offset), o.key.region, o.full_id));
                }
                Decision::Keep => {}
            }
        }
        for (id, key) in hide {
            if let Some(idx) = self.objects.index_of_uuid(&id) {
                self.take_out(idx, key);
            }
        }
        unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ME: Uuid = Uuid::from_u128(1);
    const OWNER: Uuid = Uuid::from_u128(2);
    const OTHER: Uuid = Uuid::from_u128(3);

    #[test]
    fn hide_owner_rule_keys_never_collide_with_the_owner_id() {
        assert_ne!(rule_key(OWNER), OWNER);
        assert_eq!(owner_of_key(rule_key(OWNER)), OWNER);
        assert_ne!(rule_key(OWNER), rule_key(OTHER));
    }

    #[test]
    fn hide_owner_rule_decides_on_the_known_owner_only() {
        let rule = |o: Uuid| o == OWNER;
        assert_eq!(decide(Some(OWNER), ME, false, rule), Decision::Hide(rule_key(OWNER)));
        assert_eq!(decide(Some(OTHER), ME, false, rule), Decision::Keep);
        assert_eq!(decide(None, ME, false, rule), Decision::Ask);
        assert_eq!(decide(Some(Uuid::nil()), ME, false, rule), Decision::Ask);
    }

    #[test]
    fn hide_owner_rule_never_hides_our_own_or_a_spared_object() {
        // a rule on ourselves cannot be made, and would not apply
        assert_eq!(decide(Some(ME), ME, false, |_| true), Decision::Keep);
        // sat on or selected: stays, and is not even asked for
        assert_eq!(decide(Some(OWNER), ME, true, |_| true), Decision::Keep);
        assert_eq!(decide(None, ME, true, |_| true), Decision::Keep);
        // before login (no agent): nothing
        assert_eq!(decide(Some(OWNER), Uuid::nil(), false, |_| true), Decision::Keep);
    }

    #[test]
    fn hide_owner_lookups_keep_their_limits() {
        let mut rule = OwnerRule::default();
        let now = Instant::now();
        let wanted: Vec<_> = (0..20u128).map(|i| (20.0 - i as f32, 1, Uuid::from_u128(100 + i))).collect();
        let first = rule.pick(now, wanted.clone());
        assert_eq!(first.len(), IN_FLIGHT, "a few in flight");
        // nearest first
        assert_eq!(first[0].1, Uuid::from_u128(119));
        // nothing more until answers come or the wait ends
        assert!(rule.pick(now, wanted.clone()).is_empty());
        for (region, id) in &first {
            rule.learn(*region, *id, OWNER);
        }
        // the per-second budget: one more fits within the same second
        assert_eq!(rule.pick(now, wanted.clone()).len(), PER_SECOND - IN_FLIGHT);
        // a second later, after the wait: places free again, nobody asked twice
        let later = now + REPLY_WAIT;
        let again = rule.pick(later, wanted.clone());
        assert_eq!(again.len(), IN_FLIGHT);
        assert!(again.iter().all(|a| !first.contains(a)));
        assert_eq!(rule.owner_of(1, first[0].1), Some(OWNER));
    }

    #[test]
    fn hide_owner_lookups_are_asked_once_per_region_visit() {
        let mut rule = OwnerRule::default();
        let now = Instant::now();
        let one = vec![(1.0, 7, OTHER)];
        assert_eq!(rule.pick(now, one.clone()).len(), 1);
        rule.learn(7, OTHER, OWNER);
        let later = now + Duration::from_secs(2);
        assert!(rule.pick(later, one.clone()).is_empty(), "asked once");
        rule.forget_region(7);
        assert_eq!(rule.owner_of(7, OTHER), None, "the cache goes with the region");
        assert_eq!(rule.pick(later, one).len(), 1, "a new visit asks again");
    }

    #[test]
    fn hide_owner_no_rule_costs_nothing_and_clears_the_state() {
        let mut rule = OwnerRule::default();
        assert!(!rule.any(0, [Kind::Object, Kind::Resident].into_iter()));
        assert!(rule.any(1, [Kind::Owner].into_iter()));
        rule.learn(1, OTHER, OWNER);
        assert!(!rule.any(2, std::iter::empty()));
        assert_eq!(rule.owner_of(1, OTHER), None, "the last rule removed drops the cache");
    }

    #[test]
    fn hide_owner_log_counts_without_ids() {
        let mut rule = OwnerRule::default();
        let now = Instant::now();
        assert_eq!(rule.log_due(now), None);
        rule.pick(now, vec![(1.0, 1, OTHER)]);
        rule.learn(1, OTHER, OWNER);
        assert_eq!(rule.log_due(now + LOG_EVERY), Some((1, 1)));
        assert_eq!(rule.log_due(now + LOG_EVERY * 2), None, "nothing sent since");
    }
}
