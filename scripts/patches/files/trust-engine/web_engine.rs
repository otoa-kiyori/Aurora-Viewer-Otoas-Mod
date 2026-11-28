// aurora-local-patch: trust-engine-prefs-module
//! Preferences › Media › Plugins (local patch set, 77-trust-engine): the
//! engine of web pages (TRust only, Chromium only, TRust then Chromium;
//! `media/engine_choice.rs`), the sites always shown with Chromium, and
//! the TRust host program. The site list is edited like the kept cookies
//! list (`site_cookies.rs`, patch 76): each site removable, its subdomains
//! switch, a field to add one.

use std::cell::RefCell;

use aurora_i18n::tr;
use egui::{RichText, Vec2};

use super::SLIDER_W;
use super::cx::{Cx, field};
use crate::media::engine_choice::WebEngine;
use crate::media::site_profiles::{self, KeptSite};
use crate::theme::Palette;
use crate::ui::widgets;

thread_local! {
    /// The site being typed.
    static TYPED: RefCell<String> = const { RefCell::new(String::new()) };
}

/// The rows, at the top of the Plugins group.
pub fn rows(cx: &mut Cx) {
    let p = cx.p;
    cx.keywords("TRust Chromium CEF").row(
        tr!("ui-options-web-engine"),
        tr!("ui-options-web-engine-help"),
        &field!(media.web_engine),
        |ui, s| {
            let mut changed = false;
            let selected = s.media.web_engine;
            ui.vertical(|ui| {
                for (engine, label) in [
                    (WebEngine::TrustThenChromium, tr!("ui-options-web-engine-trust-then-chromium")),
                    (WebEngine::TrustOnly, tr!("ui-options-web-engine-trust")),
                    (WebEngine::ChromiumOnly, tr!("ui-options-web-engine-chromium")),
                ] {
                    if widgets::radio(ui, &p, selected == engine, label).clicked() {
                        changed |= s.media.web_engine != engine;
                        s.media.web_engine = engine;
                    }
                }
            });
            changed
        },
    );
    let chromium_only = cx.s.media.web_engine == WebEngine::ChromiumOnly;
    cx.subs(chromium_only, |cx| {
        // the engine names too: a Japanese label has no word break before
        // "Chromium" for the search to find
        let words = ["TRust", "Chromium", "CEF"]
            .into_iter()
            .chain(cx.s.media.web_engine_chromium_sites.iter().map(|s| s.domain.as_str()))
            .collect::<Vec<_>>()
            .join(" ");
        cx.wide().below().keywords(&words).row(
            tr!("ui-options-web-engine-chromium-sites"),
            tr!("ui-options-web-engine-chromium-sites-hint"),
            &field!(media.web_engine_chromium_sites),
            |ui, s| TYPED.with_borrow_mut(|typed| list(ui, &p, &mut s.media.web_engine_chromium_sites, typed)),
        );
        cx.row(
            tr!("ui-options-trust-host"),
            tr!("ui-options-trust-host-help"),
            &field!(media.trust_host),
            |ui, s| {
                ui.add(
                    egui::TextEdit::singleline(&mut s.media.trust_host)
                        .desired_width(SLIDER_W + 60.0)
                        .hint_text(tr!("ui-options-plugins-folder-automatic")),
                )
                .changed()
            },
        );
    });
}

/// The sites, then the field to add one. True on a change of the list.
fn list(ui: &mut egui::Ui, p: &Palette, sites: &mut Vec<KeptSite>, typed: &mut String) -> bool {
    let mut changed = false;
    ui.vertical(|ui| {
        if sites.is_empty() {
            ui.label(
                RichText::new(tr!("ui-options-web-engine-chromium-sites-none"))
                    .size(12.0)
                    .color(p.muted),
            );
        }
        let mut removed = None;
        for (i, site) in sites.iter_mut().enumerate() {
            ui.push_id(("chromium-sites", i), |ui| {
                ui.horizontal(|ui| {
                    if widgets::icon_button(ui, p, "trash-filled", tr!("ui-options-web-engine-chromium-sites-remove"), true) {
                        removed = Some(i);
                    }
                    ui.label(RichText::new(site.domain.as_str()).size(12.5).color(p.ink));
                    let hint = tr!("ui-options-web-engine-chromium-sites-subdomains-hint", domain = site.domain.clone());
                    if widgets::switch_labeled(ui, p, &mut site.subdomains, tr!("ui-options-web-engine-chromium-sites-subdomains"))
                        .on_hover_text(hint)
                        .changed()
                    {
                        changed = true;
                    }
                });
            });
        }
        if let Some(i) = removed {
            sites.remove(i);
            changed = true;
        }
        ui.add_space(4.0);
        let domain = site_profiles::normalize(typed);
        let listed = domain.as_ref().is_some_and(|d| sites.iter().any(|s| s.domain == *d));
        ui.horizontal(|ui| {
            let field = ui.add(
                egui::TextEdit::singleline(typed)
                    .desired_width(260.0)
                    .hint_text(tr!("ui-options-web-engine-chromium-sites-add-hint")),
            );
            let entered = field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            let add = ui
                .add_enabled_ui(domain.is_some() && !listed, |ui| {
                    widgets::action_button(
                        ui,
                        p,
                        "circle-plus-filled",
                        tr!("ui-options-web-engine-chromium-sites-add"),
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
                typed.clear();
            }
        });
        if let Some(domain) = domain.as_ref().filter(|_| listed) {
            ui.label(
                RichText::new(tr!("ui-options-web-engine-chromium-sites-duplicate", domain = domain.clone()))
                    .size(12.0)
                    .color(p.muted),
            );
        } else if domain.is_none() && !typed.trim().is_empty() {
            ui.label(
                RichText::new(tr!("ui-options-web-engine-chromium-sites-invalid"))
                    .size(12.0)
                    .color(p.muted),
            );
        }
    });
    changed
}
