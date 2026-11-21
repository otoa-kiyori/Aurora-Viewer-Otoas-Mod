//! The app's side of the rule that hides the rezzed objects of chosen
//! owners (world/hide_owner.rs), local patch set: the menu entry, what
//! stays (the selection and our seat), the owner lookups and their log.
//! // aurora-local-patch: hide-owner-objects-app
use super::*;
use crate::world::derender::Kind;
use crate::world::hide_owner;
use crate::world::objects::ObjKey;
use std::collections::HashSet;
use uuid::Uuid;

/// The owner of the menu's object a rule can be made for: (owner, group
/// owned, name, empty when not known yet). None for our own objects and
/// before the properties came.
pub(super) fn menu_target(props: Option<&aurora_net::build::ObjectProps>, me: Uuid) -> Option<(Uuid, bool)> {
    let p = props?;
    if p.owner_id.is_nil() || p.owner_id == me {
        return None;
    }
    // a deeded object: its owner id is its group's (LLPermissions::isGroupOwned)
    Some((p.owner_id, !p.group_id.is_nil() && p.group_id == p.owner_id))
}

impl App {
    /// The owner's name, as the block list takes it: the group's, or the
    /// resident's legacy name. None while not known.
    fn hide_owner_name(&self, owner: Uuid, group: bool) -> Option<String> {
        let w = &self.world;
        let name = if group {
            w.groups
                .groups
                .iter()
                .find(|g| g.id == owner)
                .map(|g| g.name.clone())
                .or_else(|| w.land.group_names.get(&owner).cloned())
        } else {
            w.legacy_name(&owner)
        };
        name.filter(|n| !n.trim().is_empty())
    }

    /// The object menu's entry, from the properties the menu asked for.
    pub(super) fn hide_owner_menu_facts(&self, props: Option<&aurora_net::build::ObjectProps>) -> Option<(Uuid, bool, String)> {
        let (owner, group) = menu_target(props, self.world.agent_id)?;
        Some((owner, group, self.hide_owner_name(owner, group).unwrap_or_default()))
    }

    /// "Hide all objects of this owner": the rule, then a walk at once.
    pub(super) fn hide_owner_objects(&mut self, owner: Uuid, group_owned: bool) {
        let name = self.hide_owner_name(owner, group_owned).unwrap_or_default();
        if self.world.add_owner_rule(owner, name, group_owned) {
            self.world.derender.owner_rule.walk_soon();
        }
    }

    /// Rows removed in the blacklist window: a line for the owner rules.
    pub(super) fn hide_owner_rows_removed(&self, rows: &[crate::world::derender::RowId]) {
        let removed = rows
            .iter()
            .filter(|r| self.world.derender.entries().any(|(id, e)| *id == r.id && e.kind == Kind::Owner))
            .count();
        if removed > 0 {
            log::info!("hide owner objects: {removed} rule(s) removed");
        }
    }

    /// The answers of ObjectProperties(Family): the owners are kept for
    /// the region visit, and the next walk hides what they settle.
    pub(super) fn hide_owner_properties(&mut self, props: &[aurora_net::build::ObjectProps]) {
        if !self.world.owner_rules_exist() {
            return;
        }
        for p in props {
            let Some(region) = self
                .world
                .objects
                .index_of_uuid(&p.object_id)
                .and_then(|i| self.world.objects.get(i))
                .map(|o| o.key.region)
            else {
                continue;
            };
            self.world.derender.owner_rule.learn(region, p.object_id, p.owner_id);
        }
        self.world.derender.owner_rule.walk_soon();
    }

    /// Per frame (send_derender_requests). No rule: nothing runs and
    /// nothing is asked.
    pub(super) fn hide_owner_tick(&mut self) {
        if !self.world.owner_rules_exist() {
            return;
        }
        let now = Instant::now();
        let w = &self.world;
        // what stays: the linksets of the selection (the build tools edit
        // it) and the one we sit on
        let mut spared: HashSet<ObjKey> = self
            .build
            .selection
            .iter()
            .filter_map(|k| w.objects.index_of(k))
            .filter_map(|i| w.objects.get(context_menu::linkset_root(w, i)))
            .map(|o| o.key)
            .collect();
        if let Some(seat) = w
            .objects
            .index_of_uuid(&w.agent_id)
            .and_then(|i| w.objects.get(i))
            .and_then(|a| w.objects.parent_of(a))
            .and_then(|s| w.objects.get(context_menu::linkset_root(w, s)))
        {
            spared.insert(seat.key);
        }
        let me = w.agent_id;
        self.world.derender.owner_rule.me = me;
        self.world.derender.owner_rule.set_spared(spared);
        self.world.owner_rule_free_seat();
        if !self.world.derender.owner_rule.walk_due(now) {
            return;
        }
        let (build, interactions) = (&self.build.props, &self.interactions.props);
        let unknown = self
            .world
            .owner_rule_walk(|id| build.get(id).or_else(|| interactions.get(id)).map(|p| p.owner_id));
        for (handle, object) in self.world.derender.owner_rule.pick(now, unknown) {
            self.send(NetCommand::RequestObjectProperties { handle, object });
        }
        if let Some((sent, answered)) = self.world.derender.owner_rule.log_due(now) {
            log::info!("hide owner objects: {sent} owner lookup(s) sent in the last minute, {answered} answered");
        }
        self.hide_owner_names();
    }

    /// Rules made before the owner's name was known take it once it is.
    fn hide_owner_names(&mut self) {
        let unnamed: Vec<Uuid> = self
            .world
            .derender
            .entries()
            .filter(|(_, e)| e.kind == Kind::Owner && e.name.is_empty())
            .map(|(id, _)| *id)
            .collect();
        for key in unnamed {
            let owner = hide_owner::owner_of_key(key);
            let name = self.hide_owner_name(owner, false).or_else(|| self.hide_owner_name(owner, true));
            if let Some(name) = name {
                self.world.derender.rename(&key, &name);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::menu_target;
    use aurora_net::build::ObjectProps;
    use uuid::Uuid;

    #[test]
    fn hide_owner_menu_names_the_resident_or_the_group_and_never_ourselves() {
        let me = Uuid::from_u128(1);
        let resident = Uuid::from_u128(2);
        let group = Uuid::from_u128(3);
        let props = |owner: Uuid, group_id: Uuid| ObjectProps {
            owner_id: owner,
            group_id,
            ..ObjectProps::default()
        };
        assert_eq!(menu_target(Some(&props(resident, group)), me), Some((resident, false)));
        assert_eq!(menu_target(Some(&props(group, group)), me), Some((group, true)));
        assert_eq!(menu_target(Some(&props(me, group)), me), None);
        assert_eq!(menu_target(Some(&props(Uuid::nil(), group)), me), None);
        assert_eq!(menu_target(None, me), None);
    }
}
