// aurora-local-patch: shader-look-viewer-module
//! "Otoa's Shader" as one of the photo's looks (local patch set,
//! 52-shader-photo-look): aurora-shader's whole effect chain as one layer of
//! the look stack, listed last in the Creative section of Aurora's looks.
//! It is chosen, ticked off, moved, taken out and saved by its id like any
//! other look; its strength lays the chain's result over the image the
//! layers before left (`aurora_render::looks::LookOp::UserChain`).
//!
//! It is listed only while aurora-shader is installed (the folder the host
//! reads has a `config.toml`: the rule of its Preferences page and toolbar
//! button); otherwise the lists are exactly upstream's, and a stack saved
//! with it keeps the layer as an unknown look: named "No filter", drawn as
//! nothing. With the master switch off it stays listed and, like a table
//! whose file is missing, draws nothing. The chain's content and settings
//! stay on the Preferences page, which its settings area opens.

use super::looks::{AURORA, Entry, Resolved};
use crate::theme::Palette;
use crate::ui::snapshot::Action;
use crate::ui::widgets;
use aurora_render::looks::{AuroraSection, LookOp};
use std::borrow::Cow;
use std::sync::Arc;

/// The look's name among Aurora's (its id is `aurora:otoa-shader`).
pub const NAME: &str = "otoa-shader";

#[cfg(test)]
thread_local! {
    /// Tests: aurora-shader installed or not, on this test's thread.
    static INSTALLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// aurora-shader is installed: the look is listed and draws.
pub fn listed() -> bool {
    #[cfg(test)]
    {
        INSTALLED.with(|i| i.get())
    }
    #[cfg(not(test))]
    {
        aurora_render::user_post::installed()
    }
}

/// The look's id.
pub fn id() -> String {
    format!("{AURORA}{NAME}")
}

/// Its name and description.
pub fn text() -> (&'static str, &'static str) {
    (
        aurora_i18n::tr!("photo-look-a-otoa-shader"),
        aurora_i18n::tr!("photo-look-a-otoa-shader-tip"),
    )
}

/// Aurora's looks by section, with this one last in Creative when
/// `installed` (a section of its own at the end, should upstream have no
/// Creative section any more).
pub fn add_entry(sections: &mut Vec<(AuroraSection, Vec<Entry>)>, installed: bool) {
    if !installed {
        return;
    }
    let (name, tip) = text();
    let entry = Entry {
        id: id(),
        name: Cow::Borrowed(name),
        tip: Some(tip),
    };
    match sections.iter_mut().find(|(section, _)| *section == AuroraSection::Creative) {
        Some((_, entries)) => entries.push(entry),
        None => sections.push((AuroraSection::Creative, vec![entry])),
    }
}

/// What the look draws: the whole chain at its place; None while
/// aurora-shader is not installed (an unknown look: left out of the frame).
pub fn resolve(installed: bool) -> Option<Resolved> {
    installed.then(|| Resolved {
        ops: Arc::from([LookOp::UserChain]),
        lut: None,
    })
}

/// Under the strength of its layer when it is the chosen one: where its
/// settings are, and the button that opens them (Preferences › Otoa's
/// Shader, as its toolbar button does).
pub fn settings(ui: &mut egui::Ui, p: &Palette, chosen: Option<&str>, out: &mut Vec<Action>) {
    if chosen != Some(id().as_str()) || !listed() {
        return;
    }
    let note = if aurora_render::user_post::running() {
        aurora_i18n::tr!("photo-look-otoa-shader-note")
    } else {
        aurora_i18n::tr!("photo-look-otoa-shader-off")
    };
    ui.add(egui::Label::new(egui::RichText::new(note).size(11.0).color(p.muted)).wrap());
    let open = widgets::action_button(
        ui,
        p,
        crate::ui::options::aurora_shader::ICON,
        aurora_i18n::tr!("photo-look-otoa-shader-open"),
        widgets::ActionKind::Flat,
        egui::Vec2::new(0.0, 24.0),
    );
    if open.clicked() {
        open.surrender_focus();
        out.push(Action::OtoaShaderPage);
    }
}

#[cfg(test)]
mod tests {
    use super::super::looks::{self, Layer, MAX_LAYERS, Mode, Settings};
    use super::super::luts::LutLibrary;
    use super::*;

    fn installed(on: bool) {
        INSTALLED.with(|i| i.set(on));
    }

    fn luts() -> LutLibrary {
        LutLibrary::new(std::env::temp_dir().join("aurora-no-luts"))
    }

    fn ids(stack: &[Layer]) -> Vec<&str> {
        stack.iter().map(|l| l.id.as_str()).collect()
    }

    #[test]
    fn listed_last_in_creative_only_while_installed() {
        installed(false);
        let upstream = looks::aurora_list();
        assert!(upstream.iter().flat_map(|(_, e)| e).all(|e| e.id != id()));
        assert!(looks::name(&id()).is_none());
        assert!(looks::search("otoa", &[]).is_empty());
        installed(true);
        let with = looks::aurora_list();
        assert_eq!(with.len(), upstream.len());
        for ((section, entries), (up_section, up_entries)) in with.iter().zip(&upstream) {
            assert_eq!(section, up_section);
            if *section == AuroraSection::Creative {
                assert_eq!(entries[..entries.len() - 1], up_entries[..]);
                let last = entries.last().expect("an entry");
                assert_eq!(last.id, "aurora:otoa-shader");
                assert_eq!(last.name, text().0);
                assert!(!last.name.starts_with("photo-look") && last.tip.is_some_and(|t| !t.starts_with("photo-look")));
            } else {
                assert_eq!(entries, up_entries);
            }
        }
        assert_eq!(looks::tab_of(&id()), looks::Tab::Aurora);
        assert_eq!(looks::name(&id()).as_deref(), Some(text().0));
        assert!(looks::search("OTOA", &[]).iter().any(|e| e.id == id()));
        // no Creative section: one of its own, at the end
        let mut sections = Vec::new();
        add_entry(&mut sections, true);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].0, AuroraSection::Creative);
    }

    #[test]
    fn a_layer_like_any_other_look() {
        installed(true);
        let settings = Settings::new();
        let mut luts = luts();
        let mut stack = Vec::new();
        looks::choose(&mut stack, "Sepia", Mode::Replace, 0);
        assert_eq!(looks::choose(&mut stack, &id(), Mode::Add, 0), 1);
        looks::choose(&mut stack, "aurora:oil-paint", Mode::Add, 0);
        stack[1].strength = 40;
        let drawn = looks::resolve_stack(&stack, &settings, &mut luts);
        assert_eq!(drawn.len(), 3);
        assert_eq!(*drawn[1].0.ops, [LookOp::UserChain]);
        let place = looks::Placement {
            origin: [0.0, 0.0],
            photo: (64, 32),
            scale: 1.0,
        };
        let look = looks::frame_look(&drawn, place, None, false, None).expect("a look");
        assert_eq!(look.layers[1].intensity, 0.4);
        assert_eq!(*look.layers[1].ops, [LookOp::UserChain]);
        // ticked off: kept in its place, not drawn
        stack[1].on = false;
        let drawn = looks::resolve_stack(&stack, &settings, &mut luts);
        assert_eq!(drawn.len(), 2);
        assert!(drawn.iter().all(|(r, _)| *r.ops != [LookOp::UserChain]));
        assert_eq!(ids(&stack), ["Sepia", "aurora:otoa-shader", "aurora:oil-paint"]);
        // moved: drawn at its new place
        stack[1].on = true;
        stack.swap(1, 2);
        let drawn = looks::resolve_stack(&stack, &settings, &mut luts);
        assert_eq!(*drawn[2].0.ops, [LookOp::UserChain]);
        // taken out
        stack.remove(2);
        assert_eq!(ids(&stack), ["Sepia", "aurora:oil-paint"]);
        // counted in the ceiling like the others
        for _ in 0..10 {
            looks::choose(&mut stack, &id(), Mode::Add, 0);
        }
        assert_eq!(stack.len(), MAX_LAYERS);
        // replaced in its layer like the others
        let at = looks::choose(&mut stack, "Negative", Mode::Replace, 2);
        assert_eq!(stack[at].id, "Negative");
    }

    #[test]
    fn saved_by_its_id_and_kept_as_unknown_without_the_shader() {
        installed(true);
        let mut stack = vec![Layer::new(id()), Layer::new("Sepia")];
        stack[0].strength = 55;
        stack[0].on = false;
        let saved = serde_json::to_string(&stack).expect("saved");
        assert!(saved.contains("\"aurora:otoa-shader\""), "{saved}");
        let loaded: Vec<Layer> = serde_json::from_str(&saved).expect("loaded");
        assert_eq!(loaded, stack);
        // on a machine without aurora-shader: kept, named and drawn as an
        // unknown look is ("No filter", left out of the frame)
        installed(false);
        let mut on = loaded.clone();
        on[0].on = true;
        let drawn = looks::resolve_stack(&on, &Settings::new(), &mut luts());
        assert_eq!(drawn.len(), 1);
        assert_eq!(looks::label(Some(&id())), looks::label(Some("Nothing")));
        assert!(looks::resolve(&id(), &Settings::new(), &mut luts()).is_none());
        assert!(looks::resolve("aurora:nothing", &Settings::new(), &mut luts()).is_none());
        assert_eq!(on.len(), 2);
    }
}
