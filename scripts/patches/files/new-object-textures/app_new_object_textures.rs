//! The application side of the new objects' default maps (local patch set,
//! `build::new_prim_maps`): what Preferences › Build shows, the picker it
//! opens, and the offline scenario `AURORA_DEMO_NEW_OBJECT_TEXTURES`.
//! // aurora-local-patch: new-object-textures-app

use super::App;
use crate::build::new_prim_maps::Slot;
use aurora_net::inventory::InvItem;
use uuid::Uuid;

/// The demo's three texture items (full permissions unless `refused`).
const DEMO_ITEMS: [Uuid; 3] = [
    Uuid::from_u128(0xde40_0b1e_0000_4000_8000_0000_0000_0001),
    Uuid::from_u128(0xde40_0b1e_0000_4000_8000_0000_0000_0002),
    Uuid::from_u128(0xde40_0b1e_0000_4000_8000_0000_0000_0003),
];

impl App {
    /// Before the preferences draw: the rows' names and swatches.
    pub(super) fn new_object_maps_refresh(&mut self) {
        if !self.panels.settings && self.options_ui.new_prim_maps.picking.is_none() {
            return;
        }
        self.options_ui
            .new_prim_maps
            .refresh(&self.world.inventory, self.world.agent_id, &self.settings.build, &self.ui_images);
    }

    /// After the preferences: the picker the page opened. True when a
    /// setting changed (saved with the others).
    pub(super) fn new_object_maps_picker(&mut self, ctx: &egui::Context, p: &crate::theme::Palette, window_open: bool) -> bool {
        self.options_ui
            .new_prim_maps
            .show_picker(ctx, p, &mut self.world, &self.ui_images, &mut self.settings.build, window_open)
    }

    /// `AURORA_DEMO_NEW_OBJECT_TEXTURES=1|refused`: three demo textures in
    /// the inventory become the new objects' texture, normal map and
    /// specular map. `refused`: the normal map's item has no transfer
    /// permission and the specular map's item is not in the inventory; both
    /// rows say so and a new prim gets the texture alone. Without the
    /// variable a demo run puts back the defaults a previous run left.
    pub(super) fn demo_new_object_textures(&mut self) {
        static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !self.demo || DONE.load(std::sync::atomic::Ordering::Relaxed) || self.world.inventory.root.is_nil() {
            return;
        }
        DONE.store(true, std::sync::atomic::Ordering::Relaxed);
        let mode = std::env::var("AURORA_DEMO_NEW_OBJECT_TEXTURES").unwrap_or_default();
        if !matches!(mode.as_str(), "1" | "refused") {
            for slot in Slot::ALL {
                if DEMO_ITEMS.contains(&slot.stored(&self.settings.build)) {
                    *slot.stored_mut(&mut self.settings.build) = slot.default_value();
                }
            }
            return;
        }
        let refused = mode == "refused";
        let agent = self.world.agent_id;
        let root = self.world.inventory.root;
        let full = crate::world::inventory::actions::FULL | crate::world::inventory::actions::MOVE;
        let item = |n: usize, asset: Uuid, name: &str, mask: u32| InvItem {
            id: DEMO_ITEMS[n],
            parent: root,
            name: name.to_owned(),
            asset_type: 0,
            inv_type: 0,
            asset_id: asset,
            owner: agent,
            creator: agent,
            base_mask: 0x7fffffff,
            owner_mask: mask,
            next_owner_mask: mask,
            ..Default::default()
        };
        let mut items = vec![
            item(0, crate::demo::TEX_FRAMES, "New object texture", full),
            item(
                1,
                crate::demo::TEX_BUMPS,
                "New object normal map",
                if refused {
                    full & !crate::world::inventory::actions::TRANSFER
                } else {
                    full
                },
            ),
        ];
        if !refused {
            items.push(item(2, crate::demo::TEX_SNAPSHOT, "New object specular map", full));
        }
        self.world.inventory.add_items(items);
        for slot in Slot::ALL {
            *slot.stored_mut(&mut self.settings.build) = DEMO_ITEMS[slot.index()];
        }
        log::info!("demo new object textures: {mode}");
    }
}
