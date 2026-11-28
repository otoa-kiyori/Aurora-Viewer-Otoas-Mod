//! The default texture, normal map and specular map of new objects
//! (Preferences › Build › Look of new objects), local patch set.
//! // aurora-local-patch: new-object-textures-module
//!
//! A prim made with the Create tool starts with them on every face, in the
//! same pass as the rest of the new objects' look (`new_defaults.rs`): the
//! texture goes into the one ObjectImage of that look, the two maps into one
//! RenderMaterials PUT through the Build window's own Blinn-Phong path
//! (`BuildTool::legacy_put`, with `ui.legacy_pending` so the Texture tab
//! shows them before the simulator answers).
//!
//! The Legacy Viewer has a default texture only (FSDefaultObjectTexture,
//! FSCommon::applyDefaultBuildPreferences, indra/newview/fscommon.cpp); the
//! normal and specular maps are an Aurora addition. Aurora's own design:
//! - a setting holds a built-in texture or an inventory item id, never
//!   another asset id: the settings go into the settings backups, an
//!   exported file;
//! - the texture left at Default does not touch the faces: the region made
//!   them with that texture already, and their mapping stays as it is;
//! - the stored choice goes through the shared rule
//!   (`texture_copy::new_object_map`) when it is picked and again for every
//!   new prim: an item gone or whose rights no longer allow it applies
//!   nothing, and the log says why;
//! - the material is the Build window's default one (LLMaterial::LLMaterial,
//!   `LegacyMaterial::default`): repeats 1, offsets 0, rotation 0, white
//!   specular colour, glossiness 51, environment 0. Its alpha mode follows
//!   the diffuse texture once it is loaded (blend when it has alpha, none
//!   otherwise, as the Texture tab's `default_legacy`): the PUT waits for it
//!   up to `ALPHA_WAIT`, then sends none.

use super::texture_copy::{self, NewObjectMapRefusal};
use super::{BuildSettings, BuildTool};
use crate::scene::Scene;
use crate::scene::legacy_mat::{LegacyMaterial, alpha_mode};
use crate::ui::texture_picker::TexturePicker;
use crate::world::World;
use crate::world::inventory::{Inventory, actions};
use aurora_prim::te::{BLANK_TEXTURE, DEFAULT_TEXTURE, TRANSPARENT_TEXTURE, TextureFace};
use std::collections::HashSet;
use std::time::{Duration, Instant};
use uuid::Uuid;

/// How long the maps of a new prim wait for its texture's alpha.
const ALPHA_WAIT: Duration = Duration::from_secs(5);

/// One of the three defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slot {
    Texture,
    Normal,
    Specular,
}

impl Slot {
    pub const ALL: [Slot; 3] = [Slot::Texture, Slot::Normal, Slot::Specular];

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn stored(self, s: &BuildSettings) -> Uuid {
        match self {
            Slot::Texture => s.new_prim_texture,
            Slot::Normal => s.new_prim_normal,
            Slot::Specular => s.new_prim_specular,
        }
    }

    pub fn stored_mut(self, s: &mut BuildSettings) -> &mut Uuid {
        match self {
            Slot::Texture => &mut s.new_prim_texture,
            Slot::Normal => &mut s.new_prim_normal,
            Slot::Specular => &mut s.new_prim_specular,
        }
    }

    /// The setting's default: the region's own texture, no map.
    pub fn default_value(self) -> Uuid {
        match self {
            Slot::Texture => DEFAULT_TEXTURE,
            Slot::Normal | Slot::Specular => Uuid::nil(),
        }
    }

    /// Nothing to apply: the default (or a nil texture, read as it).
    pub fn unset(self, stored: Uuid) -> bool {
        stored.is_nil() || stored == self.default_value()
    }

    /// The picker's "Default" button, as the Texture tab's.
    fn picker_default(self) -> Uuid {
        match self {
            Slot::Texture => DEFAULT_TEXTURE,
            Slot::Normal => crate::build::ui::texture::DEFAULT_NORMAL,
            Slot::Specular => crate::build::ui::texture::DEFAULT_SPECULAR,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Slot::Texture => aurora_i18n::tr!("common-texture"),
            Slot::Normal => aurora_i18n::tr!("ui-options-build-new-normal"),
            Slot::Specular => aurora_i18n::tr!("ui-options-build-new-specular"),
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Slot::Texture => aurora_i18n::tr!("ui-options-build-new-texture-hint"),
            Slot::Normal => aurora_i18n::tr!("ui-options-build-new-normal-hint"),
            Slot::Specular => aurora_i18n::tr!("ui-options-build-new-specular-hint"),
        }
    }

    fn title(self) -> &'static str {
        match self {
            Slot::Texture => aurora_i18n::tr!("build-ui-texture-choose-texture"),
            Slot::Normal => aurora_i18n::tr!("build-ui-texture-choose-normal-map"),
            Slot::Specular => aurora_i18n::tr!("build-ui-texture-choose-specular-map"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Slot::Texture => "texture",
            Slot::Normal => "normal map",
            Slot::Specular => "specular map",
        }
    }
}

/// What a refusal says, on the page and in the picker's status.
pub fn refusal_text(r: NewObjectMapRefusal) -> &'static str {
    match r {
        NewObjectMapRefusal::Missing => aurora_i18n::tr!("ui-options-build-new-map-missing"),
        NewObjectMapRefusal::NotTexture => aurora_i18n::tr!("ui-options-build-new-map-not-texture"),
        NewObjectMapRefusal::NoCopy => aurora_i18n::tr!("ui-options-build-new-map-no-copy"),
        NewObjectMapRefusal::NoTransfer => aurora_i18n::tr!("ui-options-build-new-map-no-transfer"),
    }
}

fn refusal_log(r: NewObjectMapRefusal) -> &'static str {
    match r {
        NewObjectMapRefusal::Missing => "the item is not in the inventory",
        NewObjectMapRefusal::NotTexture => "the item is not a texture",
        NewObjectMapRefusal::NoCopy => "the item has no copy permission",
        NewObjectMapRefusal::NoTransfer => "the item has no transfer permission",
    }
}

/// The asset `stored` puts on a new prim's faces for `slot`: `None` when
/// unset, else the shared rule's answer.
pub fn resolve(inv: &Inventory, agent: Uuid, slot: Slot, stored: Uuid) -> Result<Option<Uuid>, NewObjectMapRefusal> {
    if slot.unset(stored) {
        return Ok(None);
    }
    texture_copy::new_object_map(inv, agent, stored).map(Some)
}

/// What a picker choice stores for `slot`: the inventory item behind the
/// chosen asset (the picker's row, else the agent's copyable item of it),
/// or the built-in texture itself. "None" (nil) and the slot's default put
/// the setting back to its default.
pub fn choice(inv: &Inventory, agent: Uuid, slot: Slot, asset: Uuid, item: Option<Uuid>) -> Result<Uuid, NewObjectMapRefusal> {
    if asset.is_nil() || asset == slot.default_value() {
        return Ok(slot.default_value());
    }
    if texture_copy::new_object_map(inv, agent, asset) == Ok(asset) {
        // a built-in texture
        return Ok(asset);
    }
    let item = item
        .filter(|id| inv.items.contains_key(id))
        .or_else(|| texture_copy::copy_item(inv, agent, asset, false))
        .ok_or(NewObjectMapRefusal::Missing)?;
    texture_copy::new_object_map(inv, agent, item).map(|_| item)
}

/// The maps a new prim takes, resolved once when it arrives.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Plan {
    pub texture: Option<Uuid>,
    pub normal: Option<Uuid>,
    pub specular: Option<Uuid>,
}

impl Plan {
    /// The three settings through the rule; a refused one applies nothing
    /// and is logged.
    pub fn resolve(inv: &Inventory, agent: Uuid, s: &BuildSettings) -> Plan {
        let get = |slot: Slot| match resolve(inv, agent, slot, slot.stored(s)) {
            Ok(asset) => asset,
            Err(r) => {
                log::warn!(
                    "new object {}: nothing applied, {} (item {})",
                    slot.name(),
                    refusal_log(r),
                    slot.stored(s)
                );
                None
            }
        };
        Plan {
            texture: get(Slot::Texture),
            normal: get(Slot::Normal),
            specular: get(Slot::Specular),
        }
    }

    /// The texture on one face; its mapping stays.
    pub fn paint(&self, face: &mut TextureFace) {
        if let Some(texture) = self.texture {
            face.texture = texture;
        }
    }

    pub fn has_maps(&self) -> bool {
        self.normal.is_some() || self.specular.is_some()
    }

    /// The Blinn-Phong material of a face whose diffuse texture has alpha
    /// (`alpha`), or none without maps.
    pub fn material(&self, alpha: bool) -> Option<LegacyMaterial> {
        self.has_maps().then(|| LegacyMaterial {
            normal_map: self.normal.unwrap_or_default(),
            specular_map: self.specular.unwrap_or_default(),
            diffuse_alpha_mode: if alpha { alpha_mode::BLEND } else { alpha_mode::NONE },
            ..LegacyMaterial::default()
        })
    }
}

/// A new prim's maps waiting for its texture's alpha.
#[derive(Debug, Clone)]
pub struct Pending {
    id: Uuid,
    until: Instant,
    plan: Plan,
}

/// Whether a face's texture has alpha; `None` until it is loaded.
fn face_alpha(scene: Option<&Scene>, texture: Uuid) -> Option<bool> {
    match texture {
        DEFAULT_TEXTURE | BLANK_TEXTURE => Some(false),
        TRANSPARENT_TEXTURE => Some(true),
        _ => scene.and_then(|s| s.textures.has_alpha(&texture)),
    }
}

impl BuildTool {
    /// The new prim `idx` gets `plan`'s maps once its texture is known.
    pub(super) fn queue_new_prim_materials(&mut self, world: &World, idx: usize, plan: Plan) {
        if !plan.has_maps() {
            return;
        }
        let Some(o) = world.objects.get(idx) else { return };
        self.new_prim_materials.push(Pending {
            id: o.full_id,
            until: Instant::now() + ALPHA_WAIT,
            plan,
        });
    }

    /// Once per frame: the RenderMaterials PUT of the new prims whose
    /// textures are loaded, or that waited long enough.
    pub(super) fn send_new_prim_materials(&mut self, world: &World, scene: Option<&Scene>) {
        if self.new_prim_materials.is_empty() {
            return;
        }
        let now = Instant::now();
        let mut waiting = Vec::new();
        for p in std::mem::take(&mut self.new_prim_materials) {
            // deleted, or out of view, before its maps went
            let Some(idx) = world.objects.index_of_uuid(&p.id) else { continue };
            let Some(te) = world.objects.get(idx).and_then(|o| o.te.clone()) else {
                continue;
            };
            let faces = self.num_faces(world, idx);
            let alphas: Vec<Option<bool>> = (0..faces).map(|f| face_alpha(scene, te.face(f).texture)).collect();
            if now < p.until && alphas.iter().any(Option::is_none) {
                waiting.push(p);
                continue;
            }
            let mut puts = Vec::new();
            for (face, alpha) in alphas.into_iter().enumerate() {
                let Some(m) = p.plan.material(alpha.unwrap_or(false)) else {
                    continue;
                };
                let face = face as u8;
                self.ui
                    .legacy_pending
                    .insert((p.id, face), (te.face(face as usize).material_id, Some(m.clone())));
                puts.push((idx, face, Some(m)));
            }
            log::info!("new object maps: {} faces of {}", puts.len(), p.id);
            self.legacy_put(world, puts);
        }
        self.new_prim_materials = waiting;
    }
}

/// What a row of the page shows: the item's name (never an asset id), the
/// texture of its swatch, the reason it is refused.
#[derive(Debug, Clone, Default)]
pub struct Shown {
    pub name: String,
    pub asset: Option<Uuid>,
    pub refused: Option<NewObjectMapRefusal>,
    pub unset: bool,
}

pub fn shown(inv: &Inventory, agent: Uuid, slot: Slot, stored: Uuid) -> Shown {
    if slot.unset(stored) {
        return match slot {
            Slot::Texture => Shown {
                name: aurora_i18n::tr!("common-default").to_owned(),
                asset: Some(DEFAULT_TEXTURE),
                refused: None,
                unset: true,
            },
            _ => Shown {
                name: aurora_i18n::tr!("common-none-2").to_owned(),
                unset: true,
                ..Shown::default()
            },
        };
    }
    let builtin = match stored {
        BLANK_TEXTURE => Some(aurora_i18n::tr!("common-blank")),
        TRANSPARENT_TEXTURE => Some(aurora_i18n::tr!("common-transparent")),
        id if id == slot.picker_default() => Some(aurora_i18n::tr!("common-default")),
        id if texture_copy::predefined(id) => Some(aurora_i18n::tr!("ui-options-build-new-map-builtin")),
        _ => None,
    };
    if let Some(name) = builtin {
        return Shown {
            name: name.to_owned(),
            asset: Some(stored),
            refused: None,
            unset: false,
        };
    }
    let name = actions::original(inv, stored)
        .and_then(|id| inv.items.get(&id))
        .map(|it| it.name.clone())
        .unwrap_or_else(|| "—".to_owned());
    match texture_copy::new_object_map(inv, agent, stored) {
        Ok(asset) => Shown {
            name,
            asset: Some(asset),
            refused: None,
            unset: false,
        },
        Err(r) => Shown {
            name,
            asset: None,
            refused: Some(r),
            unset: false,
        },
    }
}

/// The page's side of the three pickers (in `OptionsUi`), driven by the app.
#[derive(Default)]
pub struct PrefsUi {
    /// The shared texture picker, as the Build window's.
    pub picker: TexturePicker,
    /// The slot the open picker chooses for.
    pub picking: Option<Slot>,
    /// Asked by the page: open the picker on this slot.
    pub open: Option<Slot>,
    /// Filled by the app before the window draws.
    pub shown: [Shown; 3],
    /// The last choice the rule refused.
    pub status: String,
    /// The swatches' images: the app's interface images of the three
    /// choices, and those to load.
    pub images: std::collections::HashMap<Uuid, egui::TextureHandle>,
    pub wanted_images: HashSet<Uuid>,
}

impl PrefsUi {
    /// Before the window draws: what each row shows.
    pub fn refresh(
        &mut self,
        inv: &Inventory,
        agent: Uuid,
        s: &BuildSettings,
        images: &std::collections::HashMap<Uuid, egui::TextureHandle>,
    ) {
        self.images.clear();
        for slot in Slot::ALL {
            let row = shown(inv, agent, slot, slot.stored(s));
            if let Some(t) = row.asset.and_then(|asset| images.get(&asset).map(|t| (asset, t.clone()))) {
                self.images.insert(t.0, t.1);
            }
            self.shown[slot.index()] = row;
        }
    }

    /// After the window: open the picker the page asked for, show it, and
    /// store a choice the rule allows. True when a setting changed.
    pub fn show_picker(
        &mut self,
        ctx: &egui::Context,
        p: &crate::theme::Palette,
        world: &mut World,
        images: &std::collections::HashMap<Uuid, egui::TextureHandle>,
        s: &mut BuildSettings,
        window_open: bool,
    ) -> bool {
        if let Some(slot) = self.open.take() {
            let current = self.shown[slot.index()].asset.unwrap_or_default();
            self.picker
                .open_with_defaults(slot.title(), current, Some(slot.picker_default()), slot != Slot::Texture);
            self.picking = Some(slot);
            self.status.clear();
        }
        let Some(slot) = self.picking else { return false };
        if !window_open {
            self.picker.open = false;
            self.picking = None;
            return false;
        }
        // copy and transfer, as the Texture tab's map slots and a worn
        // object's: the others are greyed out and refused when dropped; a
        // click or OK goes through `choice`, which refuses them too
        let both = actions::COPY | actions::TRANSFER;
        self.picker.required_permissions = both;
        self.picker.drop_permissions = both;
        self.picker.owned_target = true;
        let mut changed = false;
        if let Some(asset) = self.picker.show(ctx, p, world, images) {
            let item = self.picker.chosen_item(asset);
            match choice(&world.inventory, world.agent_id, slot, asset, item) {
                Ok(value) => {
                    let stored = slot.stored_mut(s);
                    changed = *stored != value;
                    *stored = value;
                }
                Err(r) => {
                    log::info!("new object {}: choice refused, {}", slot.name(), refusal_log(r));
                    self.status = aurora_i18n::tr!("ui-options-build-new-map-refused", reason = refusal_text(r));
                }
            }
        }
        self.wanted_images.extend(self.picker.wanted_images.drain());
        if !self.picker.open {
            self.picking = None;
        }
        changed
    }
}

#[cfg(test)]
#[path = "new_prim_maps_tests.rs"]
mod tests;
