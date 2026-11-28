// aurora-local-patch: aurora-shader-prefs-module
//! Preferences › Extras › Otoa's Shader: the user post effects of the
//! local patch set (aurora-render `user_post.rs`). The page shows only
//! while the folder the host reads has a `config.toml` (and the host is
//! not turned off by `AURORA_SHADER=0`).
//!
//! - The master switch writes `[host] enabled`: off, the host runs nothing
//!   and frees its GPU objects (the frame is the unpatched one).
//! - The chain is a strip of tabs, one per effect of `order` (each once),
//!   in the order they are drawn, wrapping onto more lines when the page is
//!   narrow; an effect switched off has a dimmed label. Dragging a tab onto
//!   another place reorders `order` (the drag and its drop marker follow the
//!   favourites bar, `bars/favorites.rs`), a click selects it, its
//!   right-click menu moves it left or right or takes it out of `order`.
//!   The "+" at the end of the strip adds one of the folder's other effects
//!   at the end and selects it.
//! - The selected tab's body: its switch (`[<effect>] enabled`) and its way
//!   out of the chain, then a control per parameter from the `[ui]` hints
//!   of its header, the related values (`-group`) in a group of their own,
//!   an ordered chain (`-ordered`, p1 <= p2 <= ...) pushing a neighbour it
//!   passes; then a reset to the header's defaults. The search, the
//!   modified view and the other passes still go through every effect.
//! - The selection lives for the session only: it follows a moved tab,
//!   goes to a neighbour when its effect leaves the chain, and falls back
//!   to the first tab when a hand edit took its effect out.
//!
//! `config.toml` is the only source of truth. The rows show its values
//! (the effect's `//!` defaults where a key is left out) and write a change
//! back into it in place: only the value changes, comments and layout stay
//! (`user_post::edit_config`). A slider held down writes at most every
//! 0.3 s, then once more on release; the write runs on a short-lived thread
//! and replaces the file atomically, and the host's watcher applies it as
//! it applies a hand edit. A hand edit shows here the next time the watcher
//! loads the folder. A file the parser refuses is never written: its error
//! is shown instead.
//!
//! The toolbar's button (`aurora_shader`, hidden while the folder is not
//! installed) opens this page.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

use aurora_i18n::tr;
use aurora_render::user_post::{self as host, ConfigView, Control, Edit, EffectInfo, Param};
use egui::{RichText, vec2};

use super::cx::{Cx, Nothing};
use super::{OptionsUi, Page};
use crate::theme::Palette;
use crate::ui::{menu, widgets};

/// The toolbar command, and its icon (Tabler, filled; the page's too).
pub(crate) const COMMAND: &str = "aurora_shader";
pub(crate) const ICON: &str = "contrast-2-filled";

/// While a control is held, a write at most this often (seconds).
const WRITE_EVERY: f64 = 0.3;
/// The folder's effects are listed again this often while the page shows.
const RESCAN_EVERY: f64 = 2.0;
/// The host runs at most this many effects (`library::MAX_EFFECTS`).
const MAX_EFFECTS: usize = 8;

/// A tab of the chain: its height (`widgets::tabs`), the room around its
/// label, the widest label before it is cut, the room between two tabs.
const TAB_H: f32 = 30.0;
const TAB_PAD: f32 = 24.0;
const TAB_TEXT_MAX: f32 = 220.0;
const TAB_GAP: f32 = 4.0;

/// The toolbar's button asked for this page.
static OPEN: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
thread_local! {
    /// Tests: aurora-shader installed or not, on this test's thread (the
    /// host's own flag is process-wide and set by its watcher).
    static INSTALLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The folder the host reads has a `config.toml`: the page and the toolbar
/// button show.
pub(crate) fn installed() -> bool {
    #[cfg(test)]
    {
        INSTALLED.with(|i| i.get())
    }
    #[cfg(not(test))]
    {
        host::installed()
    }
}

/// The toolbar's button: Preferences (opened by the caller) goes to this
/// page at its next frame.
pub(crate) fn request_open() {
    OPEN.store(true, Ordering::Release);
}

/// At the start of the Preferences' frame: the page the toolbar's button
/// asked for, its master switch lit (like "Go to").
pub(crate) fn take_open(st: &mut OptionsUi) {
    if OPEN.swap(false, Ordering::AcqRel) {
        st.open(Page::OtoaShader, 0);
        st.reveal = Some((Page::OtoaShader, 0, tr!("aurora-shader-prefs-enabled")));
    }
}

/// `config.toml` as this page knows it, and its writes. Kept in the
/// window's `OptionsUi` (`aurora_shader`), like the other pages' state, not
/// in a thread-local: Preferences runs the pages again from inside a row
/// (the label pass of `describe_settings`, with its own throwaway
/// `OptionsUi`), and a state shared by every pass was then borrowed twice.
#[derive(Default)]
pub(super) struct State {
    /// The host's generation last read from (None: never read).
    generation: Option<u64>,
    path: Option<PathBuf>,
    text: String,
    /// The text read with the host's rules (Err: the file is refused, and
    /// nothing is written over it).
    view: Option<Result<ConfigView, String>>,
    /// The folder's effects (stem, title) and when they were listed.
    available: Vec<(String, String)>,
    scanned: Option<f64>,
    /// Changes shown already, waiting for their write.
    pending: Vec<Edit>,
    last_write: f64,
    writing: Option<mpsc::Receiver<Result<String, String>>>,
    /// The last write failed.
    error: Option<String>,
    /// The effect whose tab is selected (this session only, never saved).
    selected: Option<String>,
}

impl State {
    fn set_text(&mut self, text: String) {
        self.view = Some(ConfigView::parse(&text));
        self.text = text;
    }

    fn reload(&mut self) {
        self.path = host::config_path();
        match self.path.as_deref().map(host::read_config) {
            Some(Ok(text)) => self.set_text(text),
            Some(Err(e)) => self.view = Some(Err(e)),
            None => self.view = None,
        }
    }

    /// The write that finished, the file again when the watcher loaded it
    /// (a hand edit) while nothing of ours waits, the folder's effects now
    /// and then.
    fn sync(&mut self, now: f64) {
        if let Some(rx) = &self.writing {
            match rx.try_recv() {
                Ok(result) => {
                    self.writing = None;
                    match result {
                        Ok(text) => {
                            self.error = None;
                            if self.pending.is_empty() {
                                self.set_text(text);
                            }
                        }
                        Err(e) => {
                            log::warn!("aurora-shader: {e}");
                            self.error = Some(e);
                            self.pending.clear();
                            self.reload();
                        }
                    }
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => self.writing = None,
            }
        }
        let generation = host::generation();
        let fresh = self.generation != Some(generation);
        if fresh && self.pending.is_empty() && self.writing.is_none() {
            self.generation = Some(generation);
            self.reload();
        }
        if fresh || self.scanned.is_none_or(|t| now - t >= RESCAN_EVERY) {
            self.scanned = Some(now);
            self.available = self
                .path
                .as_deref()
                .and_then(|p| p.parent())
                .map(host::available)
                .unwrap_or_default();
        }
    }

    /// Shows `edits` at once; they are written by `flush`.
    fn change(&mut self, edits: Vec<Edit>) {
        if !matches!(self.view, Some(Ok(_))) {
            return;
        }
        match host::edit_config(&self.text, &edits) {
            Ok(text) => {
                self.set_text(text);
                for e in edits {
                    self.pending.retain(|p| p.table != e.table || p.key != e.key);
                    self.pending.push(e);
                }
            }
            Err(e) => self.error = Some(e),
        }
    }

    /// Writes the waiting changes into the file as it is on disk now, on a
    /// thread: when nothing is held, or every WRITE_EVERY while it is.
    fn flush(&mut self, now: f64, held: bool) {
        if self.pending.is_empty() || self.writing.is_some() || (held && now - self.last_write < WRITE_EVERY) {
            return;
        }
        let Some(path) = self.path.clone() else {
            self.pending.clear();
            return;
        };
        let edits = std::mem::take(&mut self.pending);
        let (tx, rx) = mpsc::channel();
        let spawned = std::thread::Builder::new().name("aurora-shader-save".into()).spawn(move || {
            let _ = tx.send(host::save_config(&path, &edits));
        });
        match spawned {
            Ok(_) => {
                self.writing = Some(rx);
                self.last_write = now;
            }
            Err(e) => self.error = Some(format!("cannot start the writer: {e}")),
        }
    }

    fn busy(&self) -> bool {
        !self.pending.is_empty() || self.writing.is_some()
    }
}

/// The page (`pages.rs`): nothing while aurora-shader is not installed.
pub(super) fn page(cx: &mut Cx) {
    if !installed() {
        return;
    }
    // The label pass describes the settings each row edits, in a Cx that is
    // thrown away; this page edits config.toml, no setting, so it has
    // nothing to describe and must not read the folder for that pass.
    if cx.collecting() {
        return;
    }
    let ctx = cx.ui.ctx().clone();
    let (now, held) = ctx.input(|i| (i.time, i.pointer.any_down()));
    // Out of `cx` while the rows borrow it, back once they are drawn.
    let mut st = std::mem::take(&mut cx.st.aurora_shader);
    st.sync(now);
    let effects = host::effects();
    let mut edits = Vec::new();
    let mut selected = st.selected.take();
    rows(cx, &st, &effects, &mut edits, &mut selected);
    st.selected = selected;
    if !edits.is_empty() {
        st.change(edits);
        ctx.request_repaint();
    }
    st.flush(now, held);
    if st.busy() {
        ctx.request_repaint_after(std::time::Duration::from_millis(100));
    }
    cx.st.aurora_shader = st;
}

fn edit(table: &str, key: &str, value: String) -> Edit {
    Edit {
        table: table.to_owned(),
        key: key.to_owned(),
        value,
    }
}

fn token(param: &Param, values: &[f64]) -> String {
    if values.len() == 1 {
        host::number_token(values[0], param.integer)
    } else {
        host::numbers_token(values, param.integer)
    }
}

/// The effects of `order`, each once, as indices into it.
fn shown(order: &[String]) -> Vec<usize> {
    let mut shown: Vec<usize> = Vec::new();
    for (i, name) in order.iter().enumerate() {
        if !shown.iter().any(|&j| order[j] == *name) {
            shown.push(i);
        }
    }
    shown
}

// ------------------------------------------------- the chain's tabs, pure

/// A change of the chain asked by its tabs (indices into the chain, each
/// effect once).
#[derive(Clone, Debug, PartialEq)]
enum Change {
    /// The tab at `from` goes to index `to` (counted once it left `from`).
    Move { from: usize, to: usize },
    /// The effect at this index leaves the chain.
    Remove(usize),
    /// This effect of the folder joins the chain at its end.
    Add(String),
}

/// `list` with the item at `from` taken out and put back at index `to`
/// (the end when `to` is past it). Unchanged when `from` is out of it.
fn move_item<T: Clone>(list: &[T], from: usize, to: usize) -> Vec<T> {
    let mut out = list.to_vec();
    if from < out.len() {
        let item = out.remove(from);
        let to = to.min(out.len());
        out.insert(to, item);
    }
    out
}

/// The index a tab dragged from `from` takes when it is dropped in front
/// of the tab at `slot` (`slot` = the number of tabs: at the end).
fn drop_index(from: usize, slot: usize) -> usize {
    if slot > from { slot - 1 } else { slot }
}

/// The tab to show: the selected one while its effect is in the chain,
/// else the first (a hand edit took it out), none for an empty chain.
fn resolve(chain: &[String], selected: Option<&str>) -> Option<String> {
    selected
        .filter(|s| chain.iter().any(|n| n.as_str() == *s))
        .map(str::to_owned)
        .or_else(|| chain.first().cloned())
}

/// The chain after `change`, and the tab selected then: the moved one, the
/// neighbour that takes the place of a removed selected one (the one before
/// it when it was the last), the added one.
fn apply(chain: &[String], selected: Option<&str>, change: &Change) -> (Vec<String>, Option<String>) {
    match change {
        Change::Move { from, to } => {
            let moved = chain.get(*from).cloned();
            let out = move_item(chain, *from, *to);
            let next = moved.or_else(|| resolve(&out, selected));
            (out, next)
        }
        Change::Remove(at) => {
            let Some(gone) = chain.get(*at) else {
                return (chain.to_vec(), resolve(chain, selected));
            };
            let out: Vec<String> = chain.iter().filter(|n| *n != gone).cloned().collect();
            let next = if selected == Some(gone.as_str()) {
                out.get(*at).or_else(|| out.last()).cloned()
            } else {
                resolve(&out, selected)
            };
            (out, next)
        }
        Change::Add(name) => {
            let mut out = chain.to_vec();
            if !out.contains(name) {
                out.push(name.clone());
            }
            (out, Some(name.clone()))
        }
    }
}

/// Where a dragged tab lands, and where its marker stands.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Slot {
    /// In front of the tab of this index (the number of tabs: the end).
    slot: usize,
    /// The marker is beside this tab: at its left edge, or at its right
    /// edge when `after` (the end of a line).
    beside: usize,
    after: bool,
}

/// The slot under `at` among the tabs' `rects`, in chain order and wrapped
/// onto lines: on the line nearest the pointer, in front of the first tab
/// whose middle is past it, else after the line's last tab (`drop_rank` of
/// `bars/favorites.rs`, on more than one line). None without tabs.
fn drop_slot(rects: &[egui::Rect], at: egui::Pos2) -> Option<Slot> {
    // the tabs of a line share their middle
    let line_y = rects
        .iter()
        .map(|r| r.center().y)
        .min_by(|a, b| (a - at.y).abs().total_cmp(&(b - at.y).abs()))?;
    let line: Vec<usize> = (0..rects.len()).filter(|&i| (rects[i].center().y - line_y).abs() < 1.0).collect();
    let last = *line.last()?;
    Some(match line.iter().copied().find(|&i| at.x < rects[i].center().x) {
        Some(i) => Slot {
            slot: i,
            beside: i,
            after: false,
        },
        None => Slot {
            slot: last + 1,
            beside: last,
            after: true,
        },
    })
}

// ------------------------------------------------------------ the page

/// The effect's title: its header's `name`, else its file's stem.
fn title_of(name: &str, effects: &[EffectInfo], available: &[(String, String)]) -> String {
    effects
        .iter()
        .find(|e| e.name == name)
        .map(|e| e.title.clone())
        .or_else(|| available.iter().find(|(stem, _)| stem == name).map(|(_, t)| t.clone()))
        .unwrap_or_else(|| name.to_owned())
}

fn rows(cx: &mut Cx, st: &State, effects: &[EffectInfo], edits: &mut Vec<Edit>, selected: &mut Option<String>) {
    let p = cx.p;
    let view = match &st.view {
        Some(Ok(view)) => view,
        Some(Err(e)) => {
            cx.group(tr!("aurora-shader-prefs-title"), |cx| {
                cx.note(&tr!("aurora-shader-prefs-broken", error = e.as_str()));
            });
            return;
        }
        None => return,
    };
    let host_on = view.host_enabled();
    cx.group(tr!("aurora-shader-prefs-title"), |cx| {
        if let Some(e) = &st.error {
            cx.note(&tr!("aurora-shader-prefs-write-failed", error = e.as_str()));
        }
        let mut on = host_on;
        cx.row(
            tr!("aurora-shader-prefs-enabled"),
            tr!("aurora-shader-prefs-enabled-help"),
            &Nothing,
            |ui, _| {
                if super::toggle(ui, &p, &mut on) {
                    edits.push(edit("host", "enabled", host::bool_token(on)));
                }
                false
            },
        );
        cx.greyed(!host_on);
        let mut photos = view.photos();
        cx.row(
            tr!("aurora-shader-prefs-photos"),
            tr!("aurora-shader-prefs-photos-help"),
            &Nothing,
            |ui, _| {
                if super::toggle(ui, &p, &mut photos) {
                    edits.push(edit("host", "photos", host::bool_token(photos)));
                }
                false
            },
        );
        // under the switches, so turning them off does not move them away
        // from the pointer
        if !host_on {
            cx.note(tr!("aurora-shader-prefs-off"));
        }
    });

    let order = view.order();
    let chain: Vec<String> = shown(&order).into_iter().map(|i| order[i].clone()).collect();
    *selected = resolve(&chain, selected.as_deref());
    let mut change: Option<Change> = None;
    cx.group(tr!("aurora-shader-prefs-chain"), |cx| {
        cx.greyed(!host_on);
        cx.section_body().row(
            tr!("aurora-shader-prefs-chain"),
            tr!("aurora-shader-prefs-chain-help"),
            &Nothing,
            |ui, _| {
                let (pick, asked) = ui
                    .vertical(|ui| strip(ui, &p, view, effects, &st.available, &chain, selected.as_deref()))
                    .inner;
                if pick.is_some() {
                    *selected = pick;
                }
                if asked.is_some() {
                    change = asked;
                }
                false
            },
        );
    });

    // the selected tab's body; the other passes (search, modified view,
    // reset) go through every effect
    for (pos, name) in chain.iter().enumerate() {
        let info = effects.iter().find(|e| e.name == *name);
        let visible = selected.as_deref() == Some(name.as_str());
        cx.conditional(visible, |cx| {
            effect_rows(cx, view, name, pos, info, host_on, edits, &mut change);
        });
    }

    if let Some(change) = change {
        let (new_chain, next) = apply(&chain, selected.as_deref(), &change);
        if new_chain != chain {
            edits.push(edit("host", "order", host::names_token(&new_chain)));
        }
        *selected = next;
    }
}

/// What is dragged: a tab of the chain, by its effect.
#[derive(Clone)]
struct TabDrag {
    name: String,
    title: String,
}

/// The chain as a strip of tabs (the look of `widgets::tabs`), then the
/// "+" that adds an effect. Returns the tab clicked and the change asked.
fn strip(
    ui: &mut egui::Ui,
    p: &Palette,
    view: &ConfigView,
    effects: &[EffectInfo],
    available: &[(String, String)],
    chain: &[String],
    selected: Option<&str>,
) -> (Option<String>, Option<Change>) {
    let mut pick = None;
    let mut change = None;
    let count = chain.len();
    let font = widgets::control_font(ui, 12.0);
    let dragged = egui::DragAndDrop::payload::<TabDrag>(ui.ctx());
    let out: Vec<&(String, String)> = available.iter().filter(|(stem, _)| !chain.contains(stem)).collect();
    let mut rects: Vec<egui::Rect> = Vec::with_capacity(count);
    let wrapped = ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = vec2(TAB_GAP, TAB_GAP);
        for (pos, name) in chain.iter().enumerate() {
            let title = title_of(name, effects, available);
            let on = view.effect_enabled(name);
            let galley =
                egui::WidgetText::from(title.as_str()).into_galley(ui, Some(egui::TextWrapMode::Truncate), TAB_TEXT_MAX, font.clone());
            let text_w = galley.size().x;
            let (rect, _) = ui.allocate_exact_size(vec2(text_w + TAB_PAD, TAB_H), egui::Sense::hover());
            rects.push(rect);
            let resp = ui.interact(
                rect,
                ui.make_persistent_id(("aurora_shader_tab", name.as_str())),
                egui::Sense::click_and_drag(),
            );
            let moving = dragged.as_ref().is_some_and(|d| d.name == *name);
            let hot = dragged.is_none() && resp.hovered();
            let sel = selected == Some(name.as_str());
            if sel || hot {
                let fill = if sel {
                    p.violet.gamma_multiply(0.18)
                } else {
                    p.raised.gamma_multiply(0.25)
                };
                ui.painter().rect_filled(rect, ui.visuals().window_corner_radius, fill);
            }
            underline(ui, p, rect, sel, hot);
            // switched off: kept in the chain, not drawn
            let ink = match (on, sel) {
                (true, true) => p.ink,
                (true, false) | (false, true) => p.muted,
                (false, false) => p.muted_dim,
            };
            let ink = if moving { ink.gamma_multiply(0.35) } else { ink };
            widgets::paint_row_text(ui.painter(), galley, rect.center().x - text_w / 2.0, rect.center().y, ink);

            let file = format!("{name}.wgsl");
            let detail = match effects.iter().find(|e| e.name == *name).and_then(|e| e.problem.as_deref()) {
                Some(problem) => tr!("aurora-shader-prefs-problem", file = file.as_str(), problem = problem),
                None => file,
            };
            let resp = resp.on_hover_text(format!("{title}\n{detail}\n{}", tr!("aurora-shader-prefs-tab-hint")));
            if resp.drag_started() {
                resp.dnd_set_drag_payload(TabDrag {
                    name: name.clone(),
                    title: title.clone(),
                });
            }
            if resp.clicked() {
                pick = Some(name.clone());
            }
            menu::context_menu(&resp, p, |ui| {
                if menu::item_if(ui, p, "arrow-left", tr!("aurora-shader-prefs-move-left"), pos > 0) {
                    change = Some(Change::Move { from: pos, to: pos - 1 });
                }
                if menu::item_if(ui, p, "arrow-right", tr!("aurora-shader-prefs-move-right"), pos + 1 < count) {
                    change = Some(Change::Move { from: pos, to: pos + 1 });
                }
                menu::separator(ui, p);
                if menu::item(ui, p, "x", tr!("aurora-shader-prefs-remove-from-chain")) {
                    change = Some(Change::Remove(pos));
                }
            });
        }

        // the folder's effects out of the chain, added at its end
        let room = count < MAX_EFFECTS;
        let can_add = room && !out.is_empty();
        let sense = if can_add { egui::Sense::click() } else { egui::Sense::hover() };
        let (rect, resp) = ui.allocate_exact_size(vec2(TAB_H, TAB_H), sense);
        let open = egui::Popup::is_id_open(ui.ctx(), egui::Popup::default_response_id(&resp));
        let hot = can_add && (resp.hovered() || open);
        if hot {
            ui.painter()
                .rect_filled(rect, ui.visuals().window_corner_radius, p.raised.gamma_multiply(0.25));
        }
        underline(ui, p, rect, false, hot);
        let ink = match (can_add, hot) {
            (false, _) => p.muted_dim,
            (true, true) => p.ink,
            (true, false) => p.muted,
        };
        if let Some(icon) = crate::ui::icons::global("plus") {
            ui.painter().image(
                icon.id(),
                egui::Rect::from_center_size(rect.center(), egui::Vec2::splat(16.0)),
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                ink,
            );
        }
        let tip = if !room {
            tr!("aurora-shader-prefs-add-full", max = MAX_EFFECTS)
        } else if out.is_empty() {
            tr!("aurora-shader-prefs-add-none").to_owned()
        } else {
            tr!("aurora-shader-prefs-add-effect").to_owned()
        };
        let resp = resp.on_hover_text(tip);
        menu::dropdown(&resp, p, egui::PopupCloseBehavior::CloseOnClick, |ui| {
            menu::section(ui, p, tr!("aurora-shader-prefs-add"));
            for (stem, title) in out.iter().copied() {
                if menu::item_detail(ui, p, "plus", title, &format!("{stem}.wgsl")) {
                    change = Some(Change::Add(stem.clone()));
                }
            }
        });
        if chain.is_empty() {
            ui.label(RichText::new(tr!("aurora-shader-prefs-tabs-empty")).size(11.5).color(p.muted));
        }
    });

    // a tab dragged over the strip: the marker where it would land, the
    // move once it is dropped (`drop_on` of `bars/favorites.rs`)
    if let Some(drag) = dragged {
        if let Some(from) = chain.iter().position(|n| *n == drag.name)
            && let Some(at) = ui.ctx().pointer_interact_pos()
            && ui.rect_contains_pointer(wrapped.response.rect.expand(TAB_GAP))
            && let Some(slot) = drop_slot(&rects, at)
        {
            let to = drop_index(from, slot.slot);
            if to != from {
                paint_drop(ui, p, &rects, slot);
            }
            if ui.input(|i| i.pointer.any_released()) {
                egui::DragAndDrop::clear_payload(ui.ctx());
                if to != from {
                    change = Some(Change::Move { from, to });
                }
            }
        }
        ghost(ui.ctx(), p, &drag.title);
    }
    (pick, change)
}

/// The line under a tab, as `widgets::tabs` draws it: violet under the
/// selected one.
fn underline(ui: &egui::Ui, p: &Palette, rect: egui::Rect, selected: bool, hot: bool) {
    let color = if selected {
        p.violet
    } else if hot {
        p.muted
    } else {
        p.muted_dim.gamma_multiply(0.7)
    };
    let height = if selected { 2.0 } else { 1.0 };
    ui.painter().rect_filled(
        egui::Rect::from_min_size(
            rect.left_bottom() + vec2(6.0, -height),
            vec2((rect.width() - 12.0).max(1.0), height),
        ),
        0.0,
        color,
    );
}

/// The drop marker of `slot`, above the windows (`paint_drop` of
/// `bars/favorites.rs`).
fn paint_drop(ui: &egui::Ui, p: &Palette, rects: &[egui::Rect], slot: Slot) {
    let Some(r) = rects.get(slot.beside) else { return };
    let x = if slot.after {
        r.right() + TAB_GAP / 2.0
    } else {
        r.left() - TAB_GAP / 2.0
    };
    let painter = ui
        .ctx()
        .layer_painter(egui::LayerId::new(egui::Order::Tooltip, egui::Id::unique("aurora_shader_tab_drop")));
    painter.line_segment(
        [egui::pos2(x, r.top() + 4.0), egui::pos2(x, r.bottom() - 4.0)],
        egui::Stroke::new(2.0, p.violet_light),
    );
}

/// The tab being dragged, under the pointer: the violet ghost of the
/// toolbar editor (`toolbar/editor.rs` `ghost`).
fn ghost(ctx: &egui::Context, p: &Palette, title: &str) {
    let Some(pointer) = ctx.pointer_interact_pos() else { return };
    widgets::drag_ghost(ctx, p, "aurora_shader_tab_drag", pointer + vec2(16.0, 20.0), |ui| {
        let font = egui::FontId::new(12.0, crate::ui::fonts::semibold(ui.ctx()));
        let galley = ui.painter().layout_no_wrap(title.to_owned(), font, p.on_accent);
        let (rect, _) = ui.allocate_exact_size(vec2(galley.size().x + 20.0, 28.0), egui::Sense::hover());
        ui.painter()
            .rect_filled(rect, ui.visuals().window_corner_radius, p.violet.gamma_multiply(0.85));
        widgets::paint_row_text(ui.painter(), galley, rect.left() + 10.0, rect.center().y, p.on_accent);
    });
}

/// The top of a tab's body: the effect drawn or not (`[<effect>] enabled`)
/// and its way out of the chain.
fn head(cx: &mut Cx, view: &ConfigView, name: &str, pos: usize, host_on: bool, edits: &mut Vec<Edit>, change: &mut Option<Change>) {
    let p = cx.p;
    cx.greyed(!host_on);
    let mut on = view.effect_enabled(name);
    cx.row(
        tr!("aurora-shader-prefs-effect-on"),
        &format!("{} ([{name}] enabled)", tr!("aurora-shader-prefs-on-tip")),
        &Nothing,
        |ui, _| {
            if super::toggle(ui, &p, &mut on) {
                edits.push(edit(name, "enabled", host::bool_token(on)));
            }
            false
        },
    );
    cx.row(
        tr!("aurora-shader-prefs-remove-from-chain"),
        tr!("aurora-shader-prefs-remove"),
        &Nothing,
        |ui, _| {
            let button = widgets::action_button(ui, &p, "x", tr!("common-remove"), widgets::ActionKind::Flat, vec2(0.0, 26.0));
            if button.clicked() {
                *change = Some(Change::Remove(pos));
            }
            false
        },
    );
}

/// An effect of the chain (the body of its tab): its switch and its way
/// out, its values not grouped, its reset, then a group of rows per set of
/// related values.
#[allow(clippy::too_many_arguments)]
fn effect_rows(
    cx: &mut Cx,
    view: &ConfigView,
    name: &str,
    pos: usize,
    info: Option<&EffectInfo>,
    host_on: bool,
    edits: &mut Vec<Edit>,
    change: &mut Option<Change>,
) {
    let p = cx.p;
    let off = !host_on || !view.effect_enabled(name);
    let title = info.map_or(name, |e| e.title.as_str());
    let Some(info) = info.filter(|e| e.problem.is_none()) else {
        cx.group(title, |cx| {
            head(cx, view, name, pos, host_on, edits, change);
            let problem = info.and_then(|e| e.problem.as_deref()).unwrap_or("");
            cx.note(&tr!(
                "aurora-shader-prefs-problem",
                file = format!("{name}.wgsl"),
                problem = problem
            ));
        });
        return;
    };
    let mut values: Vec<Vec<f64>> = info
        .params
        .iter()
        .map(|prm| view.numbers(name, &prm.name, prm.components).unwrap_or_else(|| prm.default.clone()))
        .collect();
    let mut groups: Vec<Option<&str>> = vec![None];
    for prm in &info.params {
        if let Some(g) = prm.group.as_deref()
            && !groups.contains(&Some(g))
        {
            groups.push(Some(g));
        }
    }
    for group in groups {
        let heading = match group {
            None => title.to_owned(),
            Some(g) => format!("{title} · {g}"),
        };
        cx.group(&heading, |cx| {
            if group.is_none() {
                head(cx, view, name, pos, host_on, edits, change);
            }
            cx.greyed(off);
            let members: Vec<usize> = (0..info.params.len())
                .filter(|&k| info.params[k].group.as_deref() == group)
                .collect();
            let chain: Vec<usize> = members.iter().copied().filter(|&k| info.params[k].ordered).collect();
            for &k in &members {
                let param = &info.params[k];
                if let Some(new) = control(cx, name, param, &values[k]) {
                    values[k] = new;
                    edits.push(edit(name, &param.name, token(param, &values[k])));
                    // an ordered value pushes the neighbours it passed
                    if let Some(at) = chain.iter().position(|&c| c == k) {
                        let mut line: Vec<f64> = chain.iter().map(|&c| values[c][0]).collect();
                        host::push_ordered(&mut line, at);
                        for (&c, v) in chain.iter().zip(line) {
                            if c != k && values[c][0] != v {
                                values[c][0] = v;
                                edits.push(edit(name, &info.params[c].name, token(&info.params[c], &values[c])));
                            }
                        }
                    }
                }
            }
            if group.is_none() {
                // back to the header's defaults: the keys written that differ
                let defaults: Vec<Edit> = info
                    .params
                    .iter()
                    .filter(|prm| view.has(name, &prm.name) && view.numbers(name, &prm.name, prm.components).as_ref() != Some(&prm.default))
                    .map(|prm| edit(name, &prm.name, token(prm, &prm.default)))
                    .collect();
                cx.row(
                    tr!("aurora-shader-prefs-reset"),
                    tr!("aurora-shader-prefs-reset-help"),
                    &Nothing,
                    |ui, _| {
                        let button = widgets::action_button_if(
                            ui,
                            &p,
                            "restore",
                            tr!("aurora-shader-prefs-reset-button"),
                            widgets::ActionKind::Flat,
                            vec2(0.0, 26.0),
                            !defaults.is_empty(),
                        );
                        if button.clicked() {
                            edits.extend(defaults);
                        }
                        false
                    },
                );
            }
        });
    }
}

/// One parameter's row: a slider per component (one row, never a
/// multi-handle widget), a switch or a colour. Returns its new values.
fn control(cx: &mut Cx, effect: &str, param: &Param, values: &[f64]) -> Option<Vec<f64>> {
    let p = cx.p;
    let n = param.components;
    let hint = format!("[{effect}] {}", param.name);
    let mut changed = None;
    match param.control {
        Control::Flag => {
            let mut on = values.first().is_some_and(|v| *v != 0.0);
            cx.row(&param.label, &hint, &Nothing, |ui, _| {
                if super::toggle(ui, &p, &mut on) {
                    changed = Some(vec![if on { 1.0 } else { 0.0 }]);
                }
                false
            });
        }
        Control::Color => {
            let channel = |i: usize, default: f64| (values.get(i).copied().unwrap_or(default).clamp(0.0, 1.0) * 255.0).round() as u8;
            let mut rgba = [channel(0, 0.0), channel(1, 0.0), channel(2, 0.0), channel(3, 1.0)];
            cx.row(&param.label, &hint, &Nothing, |ui, _| {
                ui.spacing_mut().interact_size = vec2(38.0, 18.0);
                let id = ui.scope_id().with(("aurora-shader", effect, param.name.as_str()));
                if widgets::color_button(ui, &p, id, &mut rgba, n == 4).changed() {
                    changed = Some(
                        rgba[..n]
                            .iter()
                            .map(|c| (f64::from(*c) / 255.0 * 1000.0).round() / 1000.0)
                            .collect(),
                    );
                }
                false
            });
        }
        Control::Slider => {
            let mut new = values.to_vec();
            if n > 1 {
                // a slider per component, side by side: both columns
                cx.wide();
            }
            cx.row(&param.label, &hint, &Nothing, |ui, _| {
                let mut moved = false;
                for v in new.iter_mut() {
                    // a value written out of the range stays as it is
                    let lo = param.min.min(*v);
                    let hi = param.max.max(*v);
                    let mut slider = egui::Slider::new(v, lo..=hi)
                        .custom_parser(aurora_i18n::parse_number)
                        .trailing_fill(true);
                    if let Some(step) = param.step {
                        slider = slider.step_by(step);
                    }
                    if param.integer {
                        slider = slider.fixed_decimals(0);
                    }
                    ui.spacing_mut().slider_width = if n == 1 { super::SLIDER_W } else { super::SLIDER_W * 0.6 };
                    moved |= ui.add(slider).changed();
                }
                if moved {
                    changed = Some(new.clone());
                }
                false
            });
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn an_item_moves_to_its_new_index() {
        let list = names(&["a", "b", "c", "d"]);
        assert_eq!(move_item(&list, 0, 2), names(&["b", "c", "a", "d"]));
        assert_eq!(move_item(&list, 3, 0), names(&["d", "a", "b", "c"]));
        assert_eq!(move_item(&list, 1, 1), list, "in place: unchanged");
        assert_eq!(move_item(&list, 1, 9), names(&["a", "c", "d", "b"]), "past the end: at the end");
        assert_eq!(move_item(&list, 7, 0), list, "no such item: unchanged");
    }

    #[test]
    fn a_drop_in_front_of_a_tab_counts_once_the_tab_left() {
        // four tabs, the second one dragged
        assert_eq!(drop_index(1, 0), 0);
        assert_eq!(drop_index(1, 1), 1, "in front of itself: no move");
        assert_eq!(drop_index(1, 2), 1, "right after itself: no move");
        assert_eq!(drop_index(1, 3), 2);
        assert_eq!(drop_index(1, 4), 3, "at the end");
        assert_eq!(drop_index(3, 0), 0);
    }

    #[test]
    fn the_drop_slot_follows_the_line_under_the_pointer() {
        let rect = |x: f32, y: f32| egui::Rect::from_min_size(egui::pos2(x, y), vec2(50.0, 30.0));
        // two tabs on the first line, one on the second
        let rects = [rect(0.0, 0.0), rect(54.0, 0.0), rect(0.0, 34.0)];
        let at = |x: f32, y: f32| drop_slot(&rects, egui::pos2(x, y)).map(|s| (s.slot, s.beside, s.after));
        assert_eq!(at(10.0, 10.0), Some((0, 0, false)));
        assert_eq!(at(70.0, 15.0), Some((1, 1, false)));
        assert_eq!(at(90.0, 15.0), Some((2, 1, true)), "the end of the first line");
        assert_eq!(at(5.0, 50.0), Some((2, 2, false)));
        assert_eq!(at(200.0, 60.0), Some((3, 2, true)), "the end of the chain");
        assert_eq!(at(10.0, -100.0), Some((0, 0, false)), "above: the first line");
        assert_eq!(at(10.0, 500.0), Some((2, 2, false)), "below: the last line");
        assert_eq!(drop_slot(&[], egui::pos2(0.0, 0.0)), None);
    }

    #[test]
    fn the_selection_falls_back_to_the_first_tab() {
        let chain = names(&["bloom", "grain", "vignette"]);
        assert_eq!(resolve(&chain, Some("grain")).as_deref(), Some("grain"));
        assert_eq!(resolve(&chain, None).as_deref(), Some("bloom"));
        // a hand edit took the selected effect out, or reordered the chain
        assert_eq!(resolve(&chain, Some("gone")).as_deref(), Some("bloom"));
        let reordered = names(&["vignette", "grain", "bloom"]);
        assert_eq!(resolve(&reordered, Some("bloom")).as_deref(), Some("bloom"));
        assert_eq!(resolve(&[], Some("bloom")), None);
    }

    #[test]
    fn the_selection_follows_the_moved_tab() {
        let chain = names(&["a", "b", "c"]);
        let (out, next) = apply(&chain, Some("b"), &Change::Move { from: 0, to: 2 });
        assert_eq!(out, names(&["b", "c", "a"]));
        assert_eq!(next.as_deref(), Some("a"));
        let (out, next) = apply(&chain, Some("a"), &Change::Move { from: 2, to: 1 });
        assert_eq!(out, names(&["a", "c", "b"]));
        assert_eq!(next.as_deref(), Some("c"));
    }

    #[test]
    fn a_removed_selected_tab_hands_over_to_a_neighbour() {
        let chain = names(&["a", "b", "c"]);
        let (out, next) = apply(&chain, Some("b"), &Change::Remove(1));
        assert_eq!(out, names(&["a", "c"]));
        assert_eq!(next.as_deref(), Some("c"), "the one that takes its place");
        let (out, next) = apply(&chain, Some("c"), &Change::Remove(2));
        assert_eq!(out, names(&["a", "b"]));
        assert_eq!(next.as_deref(), Some("b"), "the last: the one before it");
        let (_, next) = apply(&chain, Some("a"), &Change::Remove(2));
        assert_eq!(next.as_deref(), Some("a"), "another tab removed: the selection stays");
        let (out, next) = apply(&names(&["a"]), Some("a"), &Change::Remove(0));
        assert!(out.is_empty());
        assert_eq!(next, None);
        let (out, next) = apply(&chain, Some("b"), &Change::Remove(5));
        assert_eq!(out, chain, "no such tab: unchanged");
        assert_eq!(next.as_deref(), Some("b"));
    }

    #[test]
    fn an_added_effect_ends_the_chain_and_is_selected() {
        let chain = names(&["a", "b"]);
        let (out, next) = apply(&chain, Some("a"), &Change::Add("c".to_owned()));
        assert_eq!(out, names(&["a", "b", "c"]));
        assert_eq!(next.as_deref(), Some("c"));
        let (out, next) = apply(&chain, Some("a"), &Change::Add("b".to_owned()));
        assert_eq!(out, chain, "already in the chain: not twice");
        assert_eq!(next.as_deref(), Some("b"));
        let (out, next) = apply(&[], None, &Change::Add("a".to_owned()));
        assert_eq!(out, names(&["a"]));
        assert_eq!(next.as_deref(), Some("a"));
    }

    /// `config.toml` as read, the way `sync` leaves it once the host's
    /// generation is known (no folder is read in a test).
    fn read(text: &str) -> State {
        let mut st = State {
            generation: Some(host::generation()),
            ..State::default()
        };
        st.set_text(text.to_owned());
        st
    }

    /// One frame of `page` in `mode`, with aurora-shader installed.
    fn frame(st: &mut OptionsUi, mode: super::super::cx::Mode) {
        INSTALLED.with(|i| i.set(true));
        let ctx = egui::Context::default();
        let p = crate::theme::Theme::default().palette();
        let mut s = crate::settings::Settings::default();
        let d = crate::settings::Settings::default();
        let mut r = super::super::OptionsResult::default();
        super::super::run_ui(&ctx, egui::RawInput::default(), |ui| {
            let mut cx = Cx::new(ui, p, &mut s, &d, st, &mut r, mode, "");
            super::super::run_page(&mut cx, Page::OtoaShader, None);
        });
    }

    /// The first row drawn on this thread starts the label pass
    /// (`with_setting_labels`), which runs every page again, this one
    /// included: the page must take that, and keep its own state.
    #[test]
    fn the_page_survives_the_label_pass_its_rows_start() {
        let mut st = OptionsUi {
            aurora_shader: State {
                selected: Some("grain".to_owned()),
                ..read("[host]\nenabled = true\norder = [\"bloom\", \"grain\"]\n")
            },
            ..OptionsUi::default()
        };
        frame(&mut st, super::super::cx::Mode::Show);
        let kept = &st.aurora_shader;
        assert!(matches!(kept.view, Some(Ok(_))), "the page's view is kept");
        assert_eq!(kept.selected.as_deref(), Some("grain"), "the selection is kept");
    }

    /// The label pass reads nothing: no file, no folder listing.
    #[test]
    fn the_label_pass_leaves_the_page_alone() {
        let mut st = OptionsUi::default();
        frame(&mut st, super::super::cx::Mode::Collect);
        assert_eq!(st.aurora_shader.generation, None, "not synced");
        assert_eq!(st.aurora_shader.scanned, None, "the folder is not listed");
        assert!(st.aurora_shader.view.is_none());
    }
}
