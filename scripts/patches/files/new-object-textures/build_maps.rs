//! Preferences › Build › Look of new objects: the default texture, normal
//! map and specular map rows (local patch set, `build::new_prim_maps`).
//! // aurora-local-patch: new-object-textures-page-rows
//!
//! Each row shows the choice's swatch and its item's name, never an asset
//! id; "Choose" opens the shared texture picker (the app shows it after the
//! window), the second button puts the row back to its default.

use super::super::OptionsUi;
use super::super::cx::{Cx, field};
use crate::build::new_prim_maps::{Slot, refusal_text};
use crate::settings::Settings;
use crate::theme::Palette;
use crate::ui::widgets::{ActionKind, action_button};
use egui::{RichText, Vec2};

/// The three rows, then the last refused choice.
pub fn rows(cx: &mut Cx) {
    for slot in Slot::ALL {
        let p = cx.p;
        let track = match slot {
            Slot::Texture => field!(build.new_prim_texture),
            Slot::Normal => field!(build.new_prim_normal),
            Slot::Specular => field!(build.new_prim_specular),
        };
        let words = match slot {
            Slot::Texture => "FSDefaultObjectTexture texture diffuse default",
            Slot::Normal => "normal map bump Blinn-Phong material",
            Slot::Specular => "specular map shine gloss Blinn-Phong material",
        };
        cx.keywords(words)
            .row_with(slot.label(), slot.hint(), &track, |ui, s, st, _| row(ui, &p, slot, s, st));
    }
    let status = cx.st.new_prim_maps.status.clone();
    if !status.is_empty() {
        cx.note(&status);
    }
}

fn row(ui: &mut egui::Ui, p: &Palette, slot: Slot, s: &mut Settings, st: &mut OptionsUi) -> bool {
    let maps = &mut st.new_prim_maps;
    let shown = maps.shown[slot.index()].clone();
    let mut changed = false;
    ui.horizontal(|ui| {
        let swatch = crate::build::ui::texture_swatch(ui, p, &maps.images, &mut maps.wanted_images, shown.asset, "", 40.0, false, true)
            .on_hover_text(&shown.name);
        if swatch.clicked() {
            maps.open = Some(slot);
        }
        ui.vertical(|ui| {
            ui.set_max_width(150.0);
            ui.add(egui::Label::new(RichText::new(&shown.name).color(p.ink)).truncate());
            if let Some(r) = shown.refused {
                ui.add(egui::Label::new(RichText::new(refusal_text(r)).size(11.5).color(p.amber)).wrap());
            }
        });
        let size = Vec2::new(0.0, 26.0);
        if action_button(ui, p, "photo-filled", aurora_i18n::tr!("common-choose"), ActionKind::Flat, size).clicked() {
            maps.open = Some(slot);
        }
        if !shown.unset || shown.refused.is_some() {
            let (icon, label) = match slot {
                Slot::Texture => ("restore", aurora_i18n::tr!("common-default")),
                _ => ("x", aurora_i18n::tr!("common-none-2")),
            };
            if action_button(ui, p, icon, label, ActionKind::Flat, size).clicked() {
                *slot.stored_mut(&mut s.build) = slot.default_value();
                maps.status.clear();
                changed = true;
            }
        }
    });
    changed
}
