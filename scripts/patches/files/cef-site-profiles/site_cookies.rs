// aurora-local-patch: cef-site-profiles-prefs-module
//! Preferences › Network and cache › Web browser › Keep cookies for these
//! sites (local patch set): the list of `media/site_profiles.rs`. Edited
//! like the trusted sites and the picture sites: each site removable, a
//! field to add one. Each site also has its subdomains switch and a "Clear
//! cookies" button, which deletes its kept profile; both the button and
//! the removal (which deletes the profile too) are refused while one of the
//! site's pages is open, with a line that says so.

use std::cell::RefCell;

use aurora_i18n::tr;
use egui::{RichText, Vec2};

use super::cx::{Cx, field};
use crate::media::site_profiles::{self, Busy, KeptSite};
use crate::theme::Palette;
use crate::ui::widgets;

/// The button's icon (Tabler, filled; not used for another function).
const CLEAR_ICON: &str = "cookie-filled";

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

#[derive(Default)]
struct State {
    /// The site being typed.
    typed: String,
    /// The result of the last action, under the list.
    message: Option<String>,
}

/// The row, in the Web browser group.
pub fn rows(cx: &mut Cx) {
    let p = cx.p;
    let words =
        cx.s.keep_cookies_sites
            .iter()
            .map(|s| s.domain.as_str())
            .collect::<Vec<_>>()
            .join(" ");
    cx.wide().below().keywords(&words).row_with(
        tr!("ui-options-keep-cookies"),
        tr!("ui-options-keep-cookies-hint"),
        &field!(keep_cookies_sites),
        |ui, s, _, _| STATE.with_borrow_mut(|state| list(ui, &p, &mut s.keep_cookies_sites, state)),
    );
}

/// What a refused or failed deletion says.
fn busy_text(domain: &str, e: &Busy) -> String {
    match e {
        Busy::InUse => tr!("ui-options-keep-cookies-in-use", domain = domain.to_owned()),
        Busy::BadName | Busy::Io(_) => tr!("ui-options-keep-cookies-failed", domain = domain.to_owned(), error = e.to_string()),
    }
}

/// The sites, then the field to add one. True on a change of the list.
fn list(ui: &mut egui::Ui, p: &Palette, sites: &mut Vec<KeptSite>, state: &mut State) -> bool {
    let root = site_profiles::sites_root();
    let mut changed = false;
    ui.vertical(|ui| {
        if sites.is_empty() {
            ui.label(RichText::new(tr!("ui-options-keep-cookies-none")).size(12.0).color(p.muted));
        }
        let mut removed = None;
        for (i, site) in sites.iter_mut().enumerate() {
            ui.push_id(("keep-cookies", i), |ui| {
                ui.horizontal(|ui| {
                    if widgets::icon_button(ui, p, "trash-filled", tr!("ui-options-keep-cookies-remove"), true) {
                        removed = Some(i);
                    }
                    ui.label(RichText::new(site.domain.as_str()).size(12.5).color(p.ink));
                    let hint = tr!("ui-options-keep-cookies-subdomains-hint", domain = site.domain.clone());
                    if widgets::switch_labeled(ui, p, &mut site.subdomains, tr!("ui-options-keep-cookies-subdomains"))
                        .on_hover_text(hint)
                        .changed()
                    {
                        changed = true;
                    }
                    let kept = site_profiles::has_profile(&root, &site.domain);
                    let clear = widgets::action_button_if(
                        ui,
                        p,
                        CLEAR_ICON,
                        tr!("ui-options-keep-cookies-clear"),
                        widgets::ActionKind::FlatDanger,
                        Vec2::new(0.0, 24.0),
                        kept,
                    )
                    .on_hover_text(tr!("ui-options-keep-cookies-clear-hint"))
                    .clicked();
                    if clear {
                        state.message = Some(match site_profiles::clear(&root, &site.domain) {
                            Ok(_) => tr!("ui-options-keep-cookies-cleared", domain = site.domain.clone()),
                            Err(e) => busy_text(&site.domain, &e),
                        });
                    }
                });
            });
        }
        if let Some(i) = removed {
            // the site's profile goes with it; never while a page has it
            let domain = sites[i].domain.clone();
            match site_profiles::clear(&root, &domain) {
                Ok(_) => {
                    sites.remove(i);
                    changed = true;
                    state.message = None;
                }
                Err(e) => state.message = Some(busy_text(&domain, &e)),
            }
        }
        if let Some(message) = &state.message {
            ui.label(RichText::new(message.as_str()).size(12.0).color(p.muted));
        }
        ui.add_space(4.0);
        let domain = site_profiles::normalize(&state.typed);
        let listed = domain.as_ref().is_some_and(|d| sites.iter().any(|s| s.domain == *d));
        ui.horizontal(|ui| {
            let field = ui.add(
                egui::TextEdit::singleline(&mut state.typed)
                    .desired_width(260.0)
                    .hint_text(tr!("ui-options-keep-cookies-add-hint")),
            );
            let entered = field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            let add = ui
                .add_enabled_ui(domain.is_some() && !listed, |ui| {
                    widgets::action_button(
                        ui,
                        p,
                        "circle-plus-filled",
                        tr!("ui-options-keep-cookies-add"),
                        widgets::ActionKind::Flat,
                        Vec2::new(90.0, 26.0),
                    )
                })
                .inner
                .clicked();
            if let Some(domain) = domain.as_ref()
                && !listed
                && (add || entered)
            {
                sites.push(KeptSite {
                    domain: domain.clone(),
                    subdomains: true,
                });
                changed = true;
                state.typed.clear();
                state.message = None;
            }
        });
        if let Some(domain) = domain.as_ref().filter(|_| listed) {
            ui.label(
                RichText::new(tr!("ui-options-keep-cookies-duplicate", domain = domain.clone()))
                    .size(12.0)
                    .color(p.muted),
            );
        } else if domain.is_none() && !state.typed.trim().is_empty() {
            ui.label(RichText::new(tr!("ui-options-keep-cookies-invalid")).size(12.0).color(p.muted));
        }
    });
    changed
}
