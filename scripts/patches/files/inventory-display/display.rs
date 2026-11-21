//! Two display switches of the inventory window (local patch set), saved
//! with its other preferences (`InventoryPreferences`), both off by
//! default. Display only: what the folder lists show and how a folder's
//! name is drawn, never what the inventory holds, nor any right, asset id
//! or message.
//!
//! "Hide empty system folders" follows the Legacy Viewer's
//! DebugHideEmptySystemFolders: LLInventoryPanel::updateHideEmptySystemFolders
//! (indra/newview/llinventorypanel.cpp) sets FILTERTYPE_EMPTYFOLDERS, then
//! LLInventoryFilter::checkAgainstFilterType (indra/newview/llinventoryfilter.cpp,
//! originally LGPL 2.1) refuses a folder whose preferred type
//! LLViewerFolderType::lookupIsHiddenIfEmpty marks (the "hide if empty"
//! column of LLViewerFolderDictionary, indra/newview/llviewerfoldertype.cpp)
//! while it has no direct descendant, folder or item. Two differences, both
//! hiding less: a folder whose contents are not loaded yet stays (the Legacy
//! Viewer counts it as empty), so that nothing disappears and comes back
//! during the login fetch; and the inventory and Library roots always stay.
//!
//! "Bold folders that hold worn items" is a local addition: the Legacy
//! Viewer bolds the worn items alone (LLItemBridge::getLabelStyle,
//! indra/newview/llinventorybridge.cpp). A folder is bold when a worn item,
//! at any depth, is inside it; links (the Current Outfit's, the saved
//! outfits') are not the items worn and bold nothing. The inventory and
//! Library roots are never bold: holding something worn is always true of
//! the inventory root, so it would say nothing.

// aurora-local-patch: inventory-display-module

use super::view::IdSet;
use super::{FetchState, Inventory, rules};
use crate::world::inventory::Folder;
use std::collections::HashSet;
use uuid::Uuid;

/// The folder types the Legacy Viewer hides while empty: those its
/// LLViewerFolderDictionary marks "hide if empty", Marketplace Listings (53)
/// included as its InventoryOutboxMakeVisible default leaves it. Never the
/// root (8), the Trash (14), the Current Outfit (46), an outfit (47), the
/// Marketplace stock and version folders (54, 55), the ensembles (26 to 45)
/// nor an ordinary folder (-1).
pub(super) fn hidden_when_empty(kind: i32) -> bool {
    matches!(kind, 0..=3 | 5..=7 | 10 | 13 | 15 | 16 | 20 | 21 | 23 | 48..=53 | 56..=60 | 100)
}

/// Whether the folder `id` is left out of the folder lists: of a type
/// hidden while empty, not a root, its contents loaded, and holding nothing,
/// neither in its own lists nor among the `listed` records the view found
/// under it.
fn hides(inv: &Inventory, id: Uuid, folder: &Folder, listed: usize) -> bool {
    hidden_when_empty(folder.info.type_default)
        && id != inv.root
        && id != inv.lib_root
        && folder.state == FetchState::Fetched
        && folder.children.is_empty()
        && folder.items.is_empty()
        && listed == 0
}

/// The empty system folders to leave out of the folder lists; `listed`
/// gives the records the view lists under a folder. The rule is
/// `hides`; every folder is looked at once.
pub(super) fn empty_system_folders(inv: &Inventory, listed: impl Fn(Uuid) -> usize) -> IdSet {
    inv.folders
        .iter()
        .filter(|(id, folder)| hides(inv, **id, folder, listed(**id)))
        .map(|(id, _)| *id)
        .collect()
}

/// The folders above a worn item, at any depth. `worn` holds the originals
/// of what is worn (`Facts::worn`); a link is not an item worn, so it bolds
/// nothing. The inventory and Library roots are never in it (the
/// inventory root holds everything worn, so bolding it says nothing). Each
/// chain of parents is walked once, which also ends a malformed cycle.
pub(super) fn worn_folders(inv: &Inventory, worn: &HashSet<Uuid>) -> IdSet {
    let mut folders = IdSet::default();
    for id in worn {
        let Some(item) = inv.items.get(id) else { continue };
        if matches!(item.asset_type, 24 | 25) {
            continue;
        }
        let mut parent = Some(item.parent);
        while let Some(folder) = parent.filter(|f| *f != inv.root && *f != inv.lib_root && inv.folders.contains_key(f))
            && folders.insert(folder)
        {
            parent = rules::parent(inv, folder);
        }
    }
    folders
}

/// A tile of Aurora's gallery (the gallery view and the combination
/// view's tiles): its name is semi-bold, in the ink of the other folder
/// names, when it is a folder holding a worn item (`worn_folders`, kept by
/// `ViewCache::follow_worn`), as the folder rows are.
pub(super) fn folder_tile_bold(view: &super::view::ViewCache, id: Uuid, is_folder: bool) -> bool {
    is_folder && view.holds_worn(id)
}

#[cfg(test)]
mod tests {
    use super::super::view::{Sort, ViewCache};
    use super::*;
    use aurora_net::inventory::{InvFolder, InvItem};

    fn id(n: u128) -> Uuid {
        Uuid::from_u128(n)
    }

    fn folder(inv: &mut Inventory, n: u128, parent: u128, kind: i32, state: FetchState) {
        inv.folders.insert(
            id(n),
            Folder {
                info: InvFolder {
                    id: id(n),
                    parent: id(parent),
                    type_default: kind,
                    name: format!("folder {n}"),
                    ..Default::default()
                },
                children: Vec::new(),
                items: Vec::new(),
                state,
                library: false,
            },
        );
        if let Some(up) = inv.folders.get_mut(&id(parent)) {
            up.children.push(id(n));
        }
        inv.generation += 1;
    }

    fn item(inv: &mut Inventory, n: u128, parent: u128, asset_type: i32) {
        inv.items.insert(
            id(n),
            InvItem {
                id: id(n),
                parent: id(parent),
                name: format!("item {n}"),
                asset_type,
                inv_type: 6,
                ..Default::default()
            },
        );
        if let Some(up) = inv.folders.get_mut(&id(parent)) {
            up.items.push(id(n));
        }
        inv.generation += 1;
    }

    /// What each folder holds, from its own lists.
    fn listed(inv: &Inventory) -> impl Fn(Uuid) -> usize + '_ {
        move |n| inv.folders.get(&n).map_or(0, |f| f.children.len() + f.items.len())
    }

    fn set(ids: &[u128]) -> IdSet {
        ids.iter().map(|n| id(*n)).collect()
    }

    /// The root (1) and its system folders; the Library root (90), empty,
    /// has a type hidden while empty.
    fn inventory() -> Inventory {
        let mut inv = Inventory {
            root: id(1),
            lib_root: id(90),
            ..Default::default()
        };
        folder(&mut inv, 1, 0, 8, FetchState::Fetched);
        folder(&mut inv, 2, 1, 0, FetchState::Fetched); // Textures, empty
        folder(&mut inv, 3, 1, 3, FetchState::Fetched); // Landmarks, one landmark
        item(&mut inv, 30, 3, 3);
        folder(&mut inv, 4, 1, 16, FetchState::Unknown); // Lost And Found, not loaded
        folder(&mut inv, 5, 1, 14, FetchState::Fetched); // Trash, empty
        folder(&mut inv, 6, 1, 46, FetchState::Fetched); // Current Outfit, empty
        folder(&mut inv, 7, 1, -1, FetchState::Fetched); // an ordinary folder, empty
        folder(&mut inv, 8, 1, 48, FetchState::Fetched); // My Outfits, one empty outfit
        folder(&mut inv, 9, 8, 47, FetchState::Fetched);
        folder(&mut inv, 10, 1, 20, FetchState::Fetching); // Animations, loading
        folder(&mut inv, 90, 0, 0, FetchState::Fetched);
        inv
    }

    #[test]
    fn the_legacy_types_hidden_while_empty() {
        for kind in [0, 1, 3, 6, 10, 16, 20, 23, 48, 49, 50, 53, 56, 57, 100] {
            assert!(hidden_when_empty(kind), "{kind}");
        }
        for kind in [-1, 8, 14, 26, 45, 46, 47, 54, 55] {
            assert!(!hidden_when_empty(kind), "{kind}");
        }
    }

    #[test]
    fn only_a_loaded_empty_system_folder_is_hidden() {
        let mut inv = inventory();
        let hidden = empty_system_folders(&inv, listed(&inv));
        // not 3 (holds a landmark), 4 and 10 (contents unknown), 5, 6, 7 and
        // 9 (types never hidden), 8 (holds an outfit), 1 and 90 (the roots)
        assert_eq!(hidden, set(&[2]));
        // something the view lists under it, not yet in its own lists
        assert!(empty_system_folders(&inv, |n| usize::from(n == id(2))).is_empty());
        // loaded at last, still empty: hidden too
        inv.folders.get_mut(&id(4)).expect("lost and found").state = FetchState::Fetched;
        assert_eq!(empty_system_folders(&inv, listed(&inv)), set(&[2, 4]));
        // it shows again as soon as it holds something
        item(&mut inv, 20, 2, 0);
        assert_eq!(empty_system_folders(&inv, listed(&inv)), set(&[4]));
    }

    #[test]
    fn the_folder_lists_leave_hidden_folders_out_only_with_the_switch() {
        let mut inv = inventory();
        let shown = |view: &ViewCache, inv: &Inventory| -> Vec<Uuid> { view.children_iter(inv.root, true, inv.lib_root).collect() };
        let mut view = ViewCache::default();
        view.prepare(&inv, Sort::default());
        assert!(shown(&view, &inv).contains(&id(2)), "off: every folder, as upstream");
        view.hide_empty_system_folders(true);
        view.prepare(&inv, Sort::default());
        let rows = shown(&view, &inv);
        assert!(!rows.contains(&id(2)));
        assert!(rows.contains(&id(4)) && rows.contains(&id(10)), "contents not known yet: shown");
        assert!(rows.contains(&id(90)), "the Library root stays");
        item(&mut inv, 20, 2, 0);
        view.prepare(&inv, Sort::default());
        assert!(shown(&view, &inv).contains(&id(2)), "it holds something again");
        inv.items.remove(&id(20));
        if let Some(textures) = inv.folders.get_mut(&id(2)) {
            textures.items.clear();
        }
        inv.generation += 1;
        view.prepare(&inv, Sort::default());
        assert!(!shown(&view, &inv).contains(&id(2)));
        view.hide_empty_system_folders(false);
        view.prepare(&inv, Sort::default());
        assert!(shown(&view, &inv).contains(&id(2)), "off again: back at once");
    }

    #[test]
    fn the_folders_above_a_worn_item_at_any_depth() {
        let mut inv = inventory();
        folder(&mut inv, 11, 7, -1, FetchState::Fetched);
        folder(&mut inv, 12, 11, -1, FetchState::Fetched);
        item(&mut inv, 40, 12, 13);
        item(&mut inv, 41, 12, 5);
        let worn = HashSet::from([id(40)]);
        assert_eq!(worn_folders(&inv, &worn), set(&[7, 11, 12]), "never the root");
        let both = HashSet::from([id(40), id(41)]);
        assert_eq!(worn_folders(&inv, &both), set(&[7, 11, 12]), "a shared chain, once");
    }

    #[test]
    fn a_link_to_a_worn_item_bolds_nothing() {
        let mut inv = inventory();
        folder(&mut inv, 11, 7, -1, FetchState::Fetched);
        item(&mut inv, 40, 11, 13);
        // its links in the Current Outfit (6) and in a saved outfit (9)
        item(&mut inv, 41, 6, 24);
        item(&mut inv, 42, 9, 24);
        let worn = HashSet::from([id(40)]);
        assert_eq!(worn_folders(&inv, &worn), set(&[7, 11]));
        let links = HashSet::from([id(41), id(42)]);
        assert!(worn_folders(&inv, &links).is_empty(), "a link given as worn counts for nothing");
    }

    #[test]
    fn nothing_worn_bolds_nothing() {
        let inv = inventory();
        assert!(worn_folders(&inv, &HashSet::new()).is_empty());
        let unknown = HashSet::from([id(777)]);
        assert!(worn_folders(&inv, &unknown).is_empty(), "an item not in the inventory");
    }

    #[test]
    fn the_roots_are_never_bold() {
        let mut inv = inventory();
        item(&mut inv, 50, 1, 6);
        assert!(
            worn_folders(&inv, &HashSet::from([id(50)])).is_empty(),
            "an item in the root bolds nothing"
        );
        // an item of the Library, given as worn: its folder, never the
        // Library root
        folder(&mut inv, 91, 90, -1, FetchState::Fetched);
        item(&mut inv, 51, 91, 13);
        assert_eq!(worn_folders(&inv, &HashSet::from([id(51)])), set(&[91]));
    }

    #[test]
    fn a_gallery_tile_is_bold_for_a_folder_holding_a_worn_item_only() {
        let mut inv = inventory();
        folder(&mut inv, 11, 7, -1, FetchState::Fetched);
        item(&mut inv, 40, 11, 13);
        let worn = HashSet::from([id(40)]);
        let mut view = ViewCache::default();
        view.follow_worn(&inv, &worn, true);
        assert!(folder_tile_bold(&view, id(11), true));
        assert!(folder_tile_bold(&view, id(7), true), "at any depth");
        assert!(!folder_tile_bold(&view, id(3), true), "a folder without a worn item");
        assert!(
            !folder_tile_bold(&view, id(40), false),
            "the worn item's own tile is not this rule's"
        );
        view.follow_worn(&inv, &worn, false);
        assert!(!folder_tile_bold(&view, id(11), true), "the switch off");
    }
}
