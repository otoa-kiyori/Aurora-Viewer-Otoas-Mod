// aurora-local-patch: new-object-textures-tests
use super::*;
use crate::build::{FLAGS_CREATE_SELECTED, Tool};
use crate::world::inventory::{FetchState, Folder};
use aurora_net::NetCommand;
use aurora_net::build::BuildCmd;
use aurora_net::inventory::{InvFolder, InvItem};
use std::sync::Arc;

const ASSET: Uuid = Uuid::from_u128(0x7e57_0101);
const NORMAL: Uuid = Uuid::from_u128(0x7e57_0102);
const LIBRARY: Uuid = Uuid::from_u128(0x7e57_01f0);
const OTHER: Uuid = Uuid::from_u128(0x7e57_0109);
const GROUP: Uuid = Uuid::from_u128(0x7e57_0108);
const FULL: u32 = actions::FULL;

fn fixture() -> (World, BuildTool, BuildSettings) {
    let mut world = World::new(Arc::new(crate::scene::avatar::AvatarLibrary::load()));
    for event in crate::demo::events().into_iter().chain(crate::demo::selection_cycle_events()) {
        world.apply(event);
    }
    world.inventory.items.clear();
    let mut tool = BuildTool::default();
    tool.open_build(Tool::Edit);
    (world, tool, BuildSettings::default())
}

/// A texture item of the agent's inventory, `mask` its owner rights.
fn give(w: &mut World, n: u128, asset: Uuid, mask: u32) -> Uuid {
    let id = Uuid::from_u128(n);
    let agent = w.agent_id;
    w.inventory.items.insert(
        id,
        InvItem {
            id,
            asset_type: 0,
            inv_type: 0,
            asset_id: asset,
            owner: agent,
            base_mask: FULL,
            owner_mask: mask,
            name: format!("Texture {n}"),
            ..Default::default()
        },
    );
    id
}

fn library_folder(w: &mut World) {
    w.inventory.lib_root = LIBRARY;
    w.inventory.folders.insert(
        LIBRARY,
        Folder {
            info: InvFolder {
                id: LIBRARY,
                name: "Library".into(),
                type_default: -1,
                ..Default::default()
            },
            children: Vec::new(),
            items: Vec::new(),
            state: FetchState::Fetched,
            library: true,
        },
    );
}

fn rule(w: &World, stored: Uuid) -> Result<Uuid, NewObjectMapRefusal> {
    texture_copy::new_object_map(&w.inventory, w.agent_id, stored)
}

// ---- settings

#[test]
fn the_defaults_change_nothing_and_survive_a_round_trip() {
    let s = BuildSettings::default();
    assert_eq!(s.new_prim_texture, DEFAULT_TEXTURE);
    assert!(s.new_prim_normal.is_nil());
    assert!(s.new_prim_specular.is_nil());
    for slot in Slot::ALL {
        assert!(slot.unset(slot.stored(&s)), "{slot:?}");
        assert_eq!(slot.stored(&s), slot.default_value());
    }
    let (w, _, _) = fixture();
    assert_eq!(Plan::resolve(&w.inventory, w.agent_id, &s), Plan::default());

    let mut chosen = s.clone();
    chosen.new_prim_texture = Uuid::from_u128(1);
    chosen.new_prim_normal = Uuid::from_u128(2);
    chosen.new_prim_specular = aurora_prim::te::BLANK_TEXTURE;
    let json = serde_json::to_string(&chosen).expect("settings json");
    let back: BuildSettings = serde_json::from_str(&json).expect("settings json");
    assert_eq!(back, chosen);

    // a settings file from before the patch takes the defaults
    let mut old = serde_json::to_value(&s).expect("settings json");
    for key in ["new_prim_texture", "new_prim_normal", "new_prim_specular"] {
        old.as_object_mut().expect("settings object").remove(key);
    }
    let old: BuildSettings = serde_json::from_value(old).expect("settings json");
    assert_eq!(old, s);
}

// ---- the permission rule

#[test]
fn a_default_map_needs_a_texture_held_with_copy_and_transfer() {
    let (mut w, _, _) = fixture();
    // full permissions, and no modify (not needed to paint)
    let full = give(&mut w, 1, ASSET, FULL);
    assert_eq!(rule(&w, full), Ok(ASSET));
    let no_modify = give(&mut w, 2, ASSET, actions::COPY | actions::TRANSFER);
    assert_eq!(rule(&w, no_modify), Ok(ASSET));
    // no copy, no transfer: refused (the Legacy Viewer asks copy only)
    let no_copy = give(&mut w, 3, ASSET, actions::MODIFY | actions::TRANSFER);
    assert_eq!(rule(&w, no_copy), Err(NewObjectMapRefusal::NoCopy));
    let no_transfer = give(&mut w, 4, ASSET, actions::COPY | actions::MODIFY);
    assert_eq!(rule(&w, no_transfer), Err(NewObjectMapRefusal::NoTransfer));
    // gone, or not a texture
    assert_eq!(rule(&w, Uuid::from_u128(99)), Err(NewObjectMapRefusal::Missing));
    let note = give(&mut w, 5, ASSET, FULL);
    let it = w.inventory.items.get_mut(&note).expect("item");
    (it.asset_type, it.inv_type) = (7, 7);
    assert_eq!(rule(&w, note), Err(NewObjectMapRefusal::NotTexture));
    // a snapshot is a texture
    let snapshot = give(&mut w, 6, ASSET, FULL);
    w.inventory.items.get_mut(&snapshot).expect("item").inv_type = 15;
    assert_eq!(rule(&w, snapshot), Ok(ASSET));
}

#[test]
fn someone_elses_and_group_items_count_with_their_everyone_rights() {
    let (mut w, _, _) = fixture();
    let other = give(&mut w, 10, ASSET, FULL);
    let it = w.inventory.items.get_mut(&other).expect("item");
    it.owner = OTHER;
    it.everyone_mask = actions::COPY;
    assert_eq!(rule(&w, other), Err(NewObjectMapRefusal::NoTransfer));
    w.inventory.items.get_mut(&other).expect("item").everyone_mask = actions::COPY | actions::TRANSFER;
    assert_eq!(rule(&w, other), Ok(ASSET));

    let group = give(&mut w, 11, ASSET, FULL);
    let it = w.inventory.items.get_mut(&group).expect("item");
    (it.group_owned, it.group_id, it.owner, it.everyone_mask) = (true, GROUP, GROUP, 0);
    assert_eq!(rule(&w, group), Err(NewObjectMapRefusal::NoCopy));
    w.inventory.items.get_mut(&group).expect("item").everyone_mask = FULL;
    assert_eq!(rule(&w, group), Ok(ASSET));
}

#[test]
fn a_library_texture_needs_copy_only_and_built_ins_need_nothing() {
    let (mut w, _, _) = fixture();
    library_folder(&mut w);
    let lib = give(&mut w, 20, ASSET, FULL);
    let it = w.inventory.items.get_mut(&lib).expect("item");
    (it.parent, it.owner, it.everyone_mask) = (LIBRARY, OTHER, actions::COPY);
    assert_eq!(rule(&w, lib), Ok(ASSET));
    w.inventory.items.get_mut(&lib).expect("item").everyone_mask = 0;
    assert_eq!(rule(&w, lib), Err(NewObjectMapRefusal::NoCopy));
    for id in [
        DEFAULT_TEXTURE,
        BLANK_TEXTURE,
        TRANSPARENT_TEXTURE,
        crate::build::ui::texture::DEFAULT_NORMAL,
        crate::build::ui::texture::DEFAULT_SPECULAR,
    ] {
        assert_eq!(rule(&w, id), Ok(id), "{id}");
    }
}

#[test]
fn a_link_follows_its_original_item() {
    let (mut w, _, _) = fixture();
    let original = give(&mut w, 30, ASSET, FULL);
    let link = give(&mut w, 31, original, FULL);
    let it = w.inventory.items.get_mut(&link).expect("item");
    (it.asset_type, it.inv_type) = (24, 0);
    assert_eq!(rule(&w, link), Ok(ASSET));
    w.inventory.items.get_mut(&original).expect("item").owner_mask = actions::COPY;
    assert_eq!(rule(&w, link), Err(NewObjectMapRefusal::NoTransfer));
}

#[test]
fn a_choice_stores_the_item_or_the_built_in_never_another_asset_id() {
    let (mut w, _, _) = fixture();
    let (inv, agent) = (&w.inventory, w.agent_id);
    assert_eq!(choice(inv, agent, Slot::Texture, Uuid::nil(), None), Ok(DEFAULT_TEXTURE));
    assert_eq!(choice(inv, agent, Slot::Normal, Uuid::nil(), None), Ok(Uuid::nil()));
    assert_eq!(choice(inv, agent, Slot::Texture, DEFAULT_TEXTURE, None), Ok(DEFAULT_TEXTURE));
    assert_eq!(choice(inv, agent, Slot::Specular, BLANK_TEXTURE, None), Ok(BLANK_TEXTURE));
    // an asset nobody holds (a typed id) is refused
    assert_eq!(choice(inv, agent, Slot::Texture, ASSET, None), Err(NewObjectMapRefusal::Missing));
    let item = give(&mut w, 40, ASSET, FULL);
    let (inv, agent) = (&w.inventory, w.agent_id);
    assert_eq!(choice(inv, agent, Slot::Texture, ASSET, Some(item)), Ok(item));
    // the picker gave no row: the agent's own item of that asset
    assert_eq!(choice(inv, agent, Slot::Normal, ASSET, None), Ok(item));
    let restricted = give(&mut w, 41, NORMAL, actions::COPY);
    let (inv, agent) = (&w.inventory, w.agent_id);
    assert_eq!(
        choice(inv, agent, Slot::Normal, NORMAL, Some(restricted)),
        Err(NewObjectMapRefusal::NoTransfer)
    );
}

#[test]
fn a_row_shows_names_and_refusals_never_an_asset_id() {
    let (mut w, _, _) = fixture();
    let item = give(&mut w, 50, ASSET, FULL);
    let row = shown(&w.inventory, w.agent_id, Slot::Normal, item);
    assert_eq!(row.name, "Texture 50");
    assert_eq!((row.asset, row.refused, row.unset), (Some(ASSET), None, false));
    w.inventory.items.get_mut(&item).expect("item").owner_mask = actions::COPY;
    let row = shown(&w.inventory, w.agent_id, Slot::Normal, item);
    assert_eq!((row.asset, row.refused), (None, Some(NewObjectMapRefusal::NoTransfer)));
    let row = shown(&w.inventory, w.agent_id, Slot::Texture, DEFAULT_TEXTURE);
    assert!(row.unset);
    assert_eq!(row.asset, Some(DEFAULT_TEXTURE));
    let row = shown(&w.inventory, w.agent_id, Slot::Specular, Uuid::nil());
    assert!(row.unset && row.asset.is_none());
}

// ---- what a new prim gets

#[test]
fn the_texture_paints_every_face_and_keeps_the_mapping() {
    let plan = Plan {
        texture: Some(ASSET),
        ..Plan::default()
    };
    let mut face = TextureFace {
        scale_s: 2.0,
        offset_t: 0.5,
        ..TextureFace::default()
    };
    plan.paint(&mut face);
    assert_eq!(face.texture, ASSET);
    assert_eq!((face.scale_s, face.offset_t), (2.0, 0.5));
    let mut untouched = TextureFace::default();
    Plan::default().paint(&mut untouched);
    assert_eq!(untouched, TextureFace::default());
    assert!(plan.material(false).is_none(), "no map, no material");
}

#[test]
fn the_maps_take_the_build_windows_default_material() {
    let plan = Plan {
        normal: Some(NORMAL),
        ..Plan::default()
    };
    let m = plan.material(false).expect("a material");
    assert_eq!(m.normal_map, NORMAL);
    assert!(m.specular_map.is_nil());
    assert_eq!(m.diffuse_alpha_mode, alpha_mode::NONE);
    assert_eq!((m.specular_color, m.specular_exp, m.env_intensity), ([255; 4], 51, 0));
    assert_eq!((m.normal_st, m.specular_st), ([1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]));
    assert_eq!(plan.material(true).expect("a material").diffuse_alpha_mode, alpha_mode::BLEND);
}

/// A prim the Create tool made arriving, ours with every right.
fn arrive_created(world: &mut World, local_id: u32) -> usize {
    use crate::ui::context::flags::{OBJECT_COPY, OBJECT_MODIFY, OBJECT_TRANSFER, OBJECT_YOU_OWNER};
    let mut update = crate::demo::action_avatar(false);
    update.pcode = aurora_prim::params::LL_PCODE_VOLUME;
    update.local_id = local_id;
    update.full_id = crate::demo::action_id(local_id);
    update.parent_id = 0;
    update.update_flags = FLAGS_CREATE_SELECTED | OBJECT_YOU_OWNER | OBJECT_MODIFY | OBJECT_COPY | OBJECT_TRANSFER;
    let near = world.objects.index_of_uuid(&crate::demo::action_id(99610)).expect("demo object");
    let handle = world.objects.get(near).expect("demo object").key.region;
    world.apply(aurora_net::NetEvent::ObjectUpdates {
        handle,
        objects: vec![update],
    });
    world.objects.index_of_uuid(&crate::demo::action_id(local_id)).expect("new prim")
}

fn material_puts(tool: &BuildTool) -> usize {
    tool.out
        .iter()
        .filter(|c| matches!(c, NetCommand::Build(BuildCmd::Cap { cap, .. }) if cap == "RenderMaterials"))
        .count()
}

#[test]
fn a_created_prim_gets_the_allowed_maps_and_nothing_refused() {
    let (mut world, mut tool, mut settings) = fixture();
    let texture = give(&mut world, 60, ASSET, actions::COPY);
    let normal = give(&mut world, 61, NORMAL, FULL);
    settings.new_prim_texture = texture;
    settings.new_prim_normal = normal;
    settings.new_prim_specular = Uuid::from_u128(62);

    tool.expect_created(&world);
    tool.expect_new_defaults();
    let idx = arrive_created(&mut world, 99960);
    tool.pick_up_created(&mut world, &settings);
    // the texture has no transfer: the faces keep the region's (the demo
    // prim's blank)
    let te = world.objects.get(idx).and_then(|o| o.te.clone()).expect("faces");
    assert!(te.faces.iter().all(|f| f.texture == BLANK_TEXTURE));
    // the normal map goes at once: that texture is opaque
    tool.send_new_prim_materials(&world, None);
    assert_eq!(material_puts(&tool), 1);
    let id = world.objects.get(idx).expect("new prim").full_id;
    let (_, m) = tool.ui.legacy_pending.get(&(id, 0)).expect("pending face");
    let m = m.as_ref().expect("a material");
    assert_eq!(
        (m.normal_map, m.specular_map, m.diffuse_alpha_mode),
        (NORMAL, Uuid::nil(), alpha_mode::NONE)
    );
    tool.send_new_prim_materials(&world, None);
    assert_eq!(material_puts(&tool), 1, "once");
}

#[test]
fn the_maps_wait_for_the_texture_then_go_without_alpha() {
    let (mut world, mut tool, mut settings) = fixture();
    settings.new_prim_texture = give(&mut world, 70, ASSET, FULL);
    settings.new_prim_specular = give(&mut world, 71, NORMAL, FULL);
    tool.expect_created(&world);
    tool.expect_new_defaults();
    let idx = arrive_created(&mut world, 99961);
    tool.pick_up_created(&mut world, &settings);
    let te = world.objects.get(idx).and_then(|o| o.te.clone()).expect("faces");
    assert!(te.faces.iter().all(|f| f.texture == ASSET));
    // not loaded (no scene): it waits
    tool.send_new_prim_materials(&world, None);
    assert_eq!(material_puts(&tool), 0);
    for p in &mut tool.new_prim_materials {
        p.until = Instant::now();
    }
    tool.send_new_prim_materials(&world, None);
    assert_eq!(material_puts(&tool), 1);
    assert!(tool.new_prim_materials.is_empty());
}
