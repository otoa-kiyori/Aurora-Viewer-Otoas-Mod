#!/usr/bin/env bash
# Build Aurora Viewer from this checkout (plus the private patch set), install
# it in the user's profile and add a launcher entry (the fuzzel / KDE menu).
#
# The build itself (build tree, patches, toolchain, memory cap, cargo lock) is
# scripts/build.sh, which can be used alone. This script passes it the build
# options (--ref, --fetch, --skip, --build-tree, --dry-run), then asks it where
# the executable is (--print-binary) and installs that.
#
# Linux only. Never runs sudo, never edits a shell profile, never touches the
# viewer's data (~/.config/auroraviewer, ~/.cache/auroraviewer,
# ~/.local/share/auroraviewer: settings, logs, the media plugin pack), except
# one link: ~/.local/share/auroraviewer/trust-host/aurora-trust-host, where
# the viewer looks for the TRust web engine's host (local patch set,
# 77-trust-engine). The host itself is installed inside the install folder
# (trust-host/), so a rollback always pairs a viewer with its own host.
#
# Usage: scripts/install-linux.sh --help
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
BUILD_SH="$SCRIPT_DIR/build.sh"

DESKTOP_ID="aurora-viewer-local.desktop"
ICON_NAME="aurora-viewer-local"
LAUNCHER_MARK="# aurora-install-linux launcher"
INSTALL_MARK=".aurora-local-install"

# user-visible locations (the XDG variables are honoured, so a test can point
# them into a scratch folder)
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
PREFIX="$HOME/.local/opt/aurora-viewer"
LAUNCHER_DIR="$HOME/.local/bin"
APP_DIR="$DATA_HOME/applications"
ICON_ROOT="$DATA_HOME/icons/hicolor"

# the TRust host (aurora-trust-host, a local repository beside this one):
# built there, copied here
TRUST_HOST_SRC=""
TRUST_HOST_DEFAULT_SRC="$(dirname "$REPO")/aurora-trust-host/target/release/aurora-trust-host"
TRUST_HOST_LINK_DIR="$DATA_HOME/auroraviewer/trust-host"
TRUST_HOST_NAME="aurora-trust-host"
TRUST_HOST_VERSION=""

REF="HEAD"
BUILD_TREE=""
SKIPS=()
BUILD_ARGS=() # the build options, as given, for build.sh
DRY=0
NO_BUILD=0
BINARY=""
ROLLBACK=0
UNINSTALL=0

# set while running
SRC_TREE=""
BIN=""
BUILD_COMMIT=""

info() { printf '%s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: scripts/install-linux.sh [options]

Builds Aurora Viewer in release mode from a dedicated build tree, applies the
local patches (scripts/patches), installs the result under your profile,
adds a menu entry (fuzzel, KDE) with Aurora's icon. The build is
scripts/build.sh, which you can also run alone.

Options:
  --ref REF            git ref to build (default: HEAD of this checkout)
  --fetch              run "git fetch origin" first (read-only); use with
                       --ref origin/main. Nothing is fetched otherwise.
  --skip ID            skip one patch (repeatable), passed to apply_patches.py
  --build-tree DIR     build tree (default: ../work/aurora-install next to the
                       main checkout)
  --prefix DIR         install folder (default: ~/.local/opt/aurora-viewer)
  --no-build           do not build; install --binary PATH (for testing the
                       install half; the patches are NOT verified)
  --binary PATH        the executable for --no-build
  --trust-host PATH    the TRust host program to install (default: the
                       release build of ../aurora-trust-host next to this
                       checkout); required, the install refuses without it
  --rollback           go back to the previous install (run again to undo)
  --uninstall          remove the install, launcher, menu entry and icons
                       (never the viewer's settings, cache, logs or media pack)
  --dry-run            print what would be done, change nothing
  -h, --help           this help

Needs ~/.cargo/bin/cargo (rustup), python3 >= 3.11, git, pkg-config and the
alsa and libudev development packages. It never calls sudo. Build memory cap
and jobs: AURORA_BUILD_MEM, AURORA_BUILD_JOBS (see scripts/build.sh --help).
EOF
}

# ---------------------------------------------------------------- arguments

parse_args() {
    while (($#)); do
        case "$1" in
        --ref) REF=${2:?--ref needs a value}; BUILD_ARGS+=(--ref "$REF"); shift 2 ;;
        --fetch) BUILD_ARGS+=(--fetch); shift ;;
        --skip) SKIPS+=("${2:?--skip needs an id}"); BUILD_ARGS+=(--skip "$2"); shift 2 ;;
        --build-tree) BUILD_TREE=${2:?--build-tree needs a folder}; BUILD_ARGS+=(--build-tree "$BUILD_TREE"); shift 2 ;;
        --prefix) PREFIX=${2:?--prefix needs a folder}; shift 2 ;;
        --no-build) NO_BUILD=1; shift ;;
        --binary) BINARY=${2:?--binary needs a path}; shift 2 ;;
        --trust-host) TRUST_HOST_SRC=${2:?--trust-host needs a path}; shift 2 ;;
        --rollback) ROLLBACK=1; shift ;;
        --uninstall) UNINSTALL=1; shift ;;
        --dry-run) DRY=1; BUILD_ARGS+=(--dry-run); shift ;;
        -h | --help) usage; exit 0 ;;
        *) die "unknown option: $1 (see --help)" ;;
        esac
    done
    PREFIX=${PREFIX%/}
    if ((NO_BUILD)) && [[ -z $BINARY ]]; then die "--no-build needs --binary PATH"; fi
    if [[ -n $BINARY ]] && ((!NO_BUILD)); then die "--binary only makes sense with --no-build"; fi
    if ((ROLLBACK && UNINSTALL)); then die "--rollback and --uninstall cannot be combined"; fi
}

# ----------------------------------------------------------------- helpers

# kbuildsycoca6 and the icon cache reach the real desktop session: only do them
# when HOME and the XDG folders are the user's real ones.
is_isolated() {
    local real_home
    real_home=$(getent passwd "$(id -u)" | cut -d: -f6)
    [[ $HOME != "$real_home" || $DATA_HOME != "$real_home/.local/share" || $CONFIG_HOME != "$real_home/.config" ]]
}

# Escape a path for a desktop entry Exec line (quoting and % rules of the
# Desktop Entry specification; the same as docs/LINUX.md).
escape_exec() {
    printf '%s' "$1" | sed -e 's/[\\"`$]/\\&/g' -e 's/\\/\\\\/g' -e 's/%/%%/g'
}

# -------------------------------------------------------------- preflight

# The build's own requirements (toolchain, dev packages, python >= 3.11) are
# checked by build.sh; this is what the install half needs.
preflight() {
    command -v git >/dev/null 2>&1 || die "git is not installed (sudo apt install git)"
    command -v python3 >/dev/null 2>&1 || die "python3 is not installed (sudo apt install python3)"
    if ((NO_BUILD)); then
        [[ -f $BINARY && -x $BINARY ]] || die "--binary $BINARY is not an executable file"
    fi
    preflight_trust_host
}

# Before anything is built or changed: the web engine host must be there,
# or the install stops here (a viewer without its host would show no page
# with "TRust only", and the default setting would never use TRust).
preflight_trust_host() {
    TRUST_HOST_SRC=${TRUST_HOST_SRC:-$TRUST_HOST_DEFAULT_SRC}
    if [[ ! -f $TRUST_HOST_SRC || ! -x $TRUST_HOST_SRC ]]; then
        die "the TRust host program is missing: $TRUST_HOST_SRC
Build it first:
  (cd $(dirname "$REPO")/aurora-trust-host && CARGO_BUILD_JOBS=2 ~/.cargo/bin/cargo build --release --offline)
or give its path with --trust-host PATH. Nothing was installed."
    fi
    TRUST_HOST_VERSION=$(binary_version "$TRUST_HOST_SRC")
}

# ------------------------------------------------------ the build's output

# Ask build.sh where it put things instead of repeating its default build tree
# rules; --print-* builds nothing, so this is also right in a dry run.
locate_build() {
    local args=()
    if [[ -n $BUILD_TREE ]]; then args=(--build-tree "$BUILD_TREE"); fi
    SRC_TREE=$("$BUILD_SH" --print-tree "${args[@]}")
    BIN=$("$BUILD_SH" --print-binary "${args[@]}")
    if ((!DRY)); then
        [[ -x $BIN ]] || die "build.sh finished but $BIN is missing"
        BUILD_COMMIT=$(git -C "$SRC_TREE" rev-parse HEAD)
        PATCH_REPORT="$SRC_TREE/.scratch/patch-report.txt"
    fi
}

# ----------------------------------------------------------- the install

binary_version() {
    local out
    out=$(timeout 30 "$1" --version 2>&1) || die "$1 --version failed: $out"
    # the text around the number is translated; the number is not
    if [[ $out =~ ([0-9]+\.[0-9]+\.[0-9]+([+-][0-9A-Za-z.+-]+)?) ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
    else
        die "cannot read a version from: $out"
    fi
}

is_our_dir() { [[ -e $1/$INSTALL_MARK ]]; }

sha256_of() { sha256sum "$1" | cut -d' ' -f1; }

install_files() {
    local stage="$PREFIX.new" version dict name
    if ((DRY)) && [[ ! -x $BIN ]]; then version="(not built in a dry run)"; else version=$(binary_version "$BIN"); fi
    info "Installing version $version into $PREFIX"
    info "TRust host $TRUST_HOST_VERSION from $TRUST_HOST_SRC (sha256 $(sha256_of "$TRUST_HOST_SRC"))"
    if [[ -e $PREFIX ]] && ! is_our_dir "$PREFIX"; then
        die "$PREFIX exists and was not installed by this script; refusing to replace it (use --prefix)"
    fi
    if ((DRY)); then
        info "[dry-run] stage $stage (aurora-viewer, trust-host/$TRUST_HOST_NAME mode 0755), swap with $PREFIX, keep the old one as $PREFIX.previous"
        info "[dry-run] link $TRUST_HOST_LINK_DIR/$TRUST_HOST_NAME -> $PREFIX/trust-host/$TRUST_HOST_NAME"
        return 0
    fi
    if [[ -e $PREFIX/aurora-viewer ]] && cmp -s "$BIN" "$PREFIX/aurora-viewer" \
        && cmp -s "$TRUST_HOST_SRC" "$PREFIX/trust-host/$TRUST_HOST_NAME"; then
        info "The installed viewer and TRust host are identical: keeping them and the previous copy."
        write_info_files "$PREFIX" "$version"
        return 0
    fi
    if command -v pgrep >/dev/null 2>&1 && pgrep -u "$(id -u)" -x aurora-viewer >/dev/null; then
        warn "Aurora Viewer is running; the new version starts the next time you launch it."
    fi
    rm -rf "$stage"
    mkdir -p "$stage/assets/emoji" "$stage/assets/dictionaries"
    install -m 0755 "$BIN" "$stage/aurora-viewer"
    mkdir -p "$stage/trust-host"
    install -m 0755 "$TRUST_HOST_SRC" "$stage/trust-host/$TRUST_HOST_NAME"
    cmp -s "$TRUST_HOST_SRC" "$stage/trust-host/$TRUST_HOST_NAME" || die "the TRust host copy differs from its source; nothing was installed"
    [[ $(binary_version "$stage/trust-host/$TRUST_HOST_NAME") == "$TRUST_HOST_VERSION" ]] || die "the copied TRust host does not run; nothing was installed"
    # what a release archive ships beside the binary (scripts/package.ps1):
    # the licences; the viewer itself needs nothing else at run time
    cp "$SRC_TREE/LICENSE" "$SRC_TREE/NOTICE.md" "$SRC_TREE/README.md" "$stage/"
    cp "$SRC_TREE/assets/emoji/LICENSE-OFL.txt" "$stage/assets/emoji/"
    dict="$SRC_TREE/crates/aurora-viewer/assets/dictionaries"
    cp "$dict/README.md" "$stage/assets/dictionaries/"
    for name in "$dict"/*/; do
        [[ -f ${name}LICENSE.txt ]] && cp "${name}LICENSE.txt" "$stage/assets/dictionaries/$(basename "$name")-LICENSE.txt"
    done
    cp "$SRC_TREE/assets/branding/aurora-logo.svg" "$stage/aurora-viewer.svg"
    write_info_files "$stage" "$version"
    : >"$stage/$INSTALL_MARK"

    mkdir -p "$(dirname "$PREFIX")"
    if [[ -e $PREFIX ]]; then
        rm -rf "$PREFIX.previous"
        mv "$PREFIX" "$PREFIX.previous"
    fi
    if ! mv "$stage" "$PREFIX"; then
        [[ -e $PREFIX.previous ]] && mv "$PREFIX.previous" "$PREFIX"
        die "could not move the new install into place; the old one was put back"
    fi
}

write_info_files() {
    local dir=$1 version=$2 id
    {
        printf 'installed: %s\n' "$(date -Is)"
        printf 'version: %s\n' "$version"
        if ((NO_BUILD)); then
            printf 'source: --no-build --binary %s (patches NOT verified)\n' "$BINARY"
        else
            printf 'ref: %s\ncommit: %s\n' "$REF" "$BUILD_COMMIT"
            printf 'patches from: %s\n' "$REPO"
            for id in "${SKIPS[@]}"; do printf 'skipped patch: %s\n' "$id"; done
        fi
        printf 'trust host: %s\n' "$TRUST_HOST_VERSION"
        printf 'trust host source: %s\n' "$TRUST_HOST_SRC"
        printf 'trust host sha256: %s\n' "$(sha256_of "$TRUST_HOST_SRC")"
        if id=$(git -C "$(dirname "$TRUST_HOST_SRC")" rev-parse HEAD 2>/dev/null); then
            printf 'trust host commit: %s\n' "$id"
        fi
    } >"$dir/local-install.txt"
    if [[ -n ${PATCH_REPORT:-} && -f $PATCH_REPORT ]]; then
        cp "$PATCH_REPORT" "$dir/patch-report.txt"
    fi
}

# The viewer looks for the host at <data>/auroraviewer/trust-host/ (the
# default of Preferences > Media > TRust program): a link to the installed
# folder's copy, which --rollback swaps with the viewer.
install_trust_host_link() {
    local link="$TRUST_HOST_LINK_DIR/$TRUST_HOST_NAME" target="$PREFIX/trust-host/$TRUST_HOST_NAME"
    if [[ -e $link && ! -L $link ]]; then
        warn "$link is a file, not our link: left alone (the viewer uses that program)."
        return 0
    fi
    if ((DRY)); then
        return 0
    fi
    mkdir -p "$TRUST_HOST_LINK_DIR"
    ln -sfn "$target" "$link.tmp"
    mv -fT "$link.tmp" "$link"
}

install_launcher() {
    local target="$LAUNCHER_DIR/aurora-viewer"
    case ":$PATH:" in
    *":$LAUNCHER_DIR:"*) ;;
    *)
        info "$LAUNCHER_DIR is not on PATH: no launcher command installed."
        return 0
        ;;
    esac
    if [[ -e $target || -L $target ]] && ! grep -qF "$LAUNCHER_MARK" "$target" 2>/dev/null; then
        warn "$target exists and is not ours: left alone."
        return 0
    fi
    if ((DRY)); then
        info "[dry-run] write launcher $target"
        return 0
    fi
    mkdir -p "$LAUNCHER_DIR"
    {
        printf '#!/bin/bash\n%s\n' "$LAUNCHER_MARK"
        printf 'exec %q "$@"\n' "$PREFIX/aurora-viewer"
    } >"$target.tmp"
    chmod 0755 "$target.tmp"
    mv -f "$target.tmp" "$target"
}

# The application icon: the PNG frames of assets/branding/aurora.ico, which
# the Windows build embeds (16 to 256 px), installed as hicolor sizes; the
# entry's Icon= is the theme name. Fuzzel and KDE find them through the icon
# theme, which inherits hicolor. No SVG on purpose: fuzzel draws SVG with
# nanosvg, which ignores the <style> classes that aurora-logo.svg colours
# itself with, and it prefers a scalable icon when one exists.
install_icon() {
    local ico="$SRC_TREE/assets/branding/aurora.ico"
    if ((DRY)); then
        info "[dry-run] install the $ICON_NAME icons (frames of $ico) under $ICON_ROOT"
        return 0
    fi
    python3 - "$ico" "$ICON_ROOT" "$ICON_NAME" <<'PY'
import os, struct, sys

ico, root, name = sys.argv[1:4]
data = open(ico, "rb").read()
count = struct.unpack("<H", data[4:6])[0]
for i in range(count):
    width, height, _, _, _, _, size, offset = struct.unpack("<BBBBHHII", data[6 + 16 * i : 22 + 16 * i])
    frame = data[offset : offset + size]
    if frame[:8] != b"\x89PNG\r\n\x1a\n":
        continue  # an old BMP frame: the PNG ones cover every size
    side = width or 256
    folder = os.path.join(root, "%dx%d" % (side, side), "apps")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, name + ".png"), "wb") as out:
        out.write(frame)
PY
}

# Only when a cache already exists (a stale one hides new icons from GTK).
refresh_icon_cache() {
    [[ -f $ICON_ROOT/icon-theme.cache ]] || return 0
    command -v gtk-update-icon-cache >/dev/null 2>&1 || return 0
    if ((DRY)); then
        info "[dry-run] gtk-update-icon-cache $ICON_ROOT"
        return 0
    fi
    gtk-update-icon-cache -f -t -q "$ICON_ROOT" || warn "gtk-update-icon-cache failed; GTK apps may show the icon after the next login."
}

write_desktop_entry() {
    local entry="$APP_DIR/$DESKTOP_ID" tmp exec_path
    exec_path=$(escape_exec "$PREFIX/aurora-viewer")
    if ((DRY)); then
        info "[dry-run] write $entry (Exec=\"$PREFIX/aurora-viewer\")"
        return 0
    fi
    mkdir -p "$APP_DIR"
    tmp="$APP_DIR/.$DESKTOP_ID.tmp"
    {
        printf '[Desktop Entry]\n'
        printf 'Type=Application\n'
        printf 'Name=Aurora Viewer (local)\n'
        printf 'GenericName=Second Life Viewer\n'
        printf 'Comment=Locally patched build of Aurora Viewer\n'
        printf 'Exec="%s"\n' "$exec_path"
        printf 'Icon=%s\n' "$ICON_NAME"
        printf 'Terminal=false\n'
        printf 'Categories=Network;Game;\n'
        printf 'Keywords=Second Life;SL;metaverse;virtual world;\n'
        printf 'StartupWMClass=aurora-viewer\n'
    } >"$tmp"
    if command -v desktop-file-validate >/dev/null 2>&1; then
        desktop-file-validate "$tmp" || {
            rm -f "$tmp"
            die "the generated desktop entry does not validate"
        }
    else
        info "desktop-file-validate is not installed: the entry was not validated."
    fi
    mv -f "$tmp" "$entry"
}

refresh_menu() {
    if is_isolated; then
        info "Isolated environment (HOME/XDG changed): not running kbuildsycoca6."
    elif command -v kbuildsycoca6 >/dev/null 2>&1; then
        if ((DRY)); then info "[dry-run] kbuildsycoca6"; else kbuildsycoca6 >/dev/null 2>&1 || warn "kbuildsycoca6 failed; the menu may update at the next login."; fi
    else
        info "kbuildsycoca6 not found: the menu updates at the next login."
    fi
}

# ------------------------------------------------- rollback and uninstall

version_of_dir() { sed -n 's/^version: //p' "$1/local-install.txt" 2>/dev/null | head -n1; }

do_rollback() {
    local prev="$PREFIX.previous" tmp="$PREFIX.swap"
    [[ -d $prev ]] || die "nothing to roll back to: $prev does not exist"
    is_our_dir "$prev" || die "$prev was not installed by this script"
    info "Rolling back: $(version_of_dir "$PREFIX") -> $(version_of_dir "$prev")"
    if ((DRY)); then
        info "[dry-run] swap $PREFIX and $prev"
        return 0
    fi
    rm -rf "$tmp"
    mv "$PREFIX" "$tmp"
    mv "$prev" "$PREFIX"
    mv "$tmp" "$prev"
    info "Done. Run --rollback again to go forward."
}

do_uninstall() {
    local dir
    info "Uninstalling (your settings, cache, logs and media plugins are NOT touched)."
    if ((DRY)); then
        info "[dry-run] remove $PREFIX, $PREFIX.previous, $LAUNCHER_DIR/aurora-viewer, $APP_DIR/$DESKTOP_ID, the $ICON_NAME icons and the link $TRUST_HOST_LINK_DIR/$TRUST_HOST_NAME"
        return 0
    fi
    for dir in "$PREFIX" "$PREFIX.previous" "$PREFIX.new" "$PREFIX.swap"; do
        [[ -e $dir ]] || continue
        if is_our_dir "$dir" || [[ $dir == *.new || $dir == *.swap ]]; then
            rm -rf "$dir"
        else
            warn "$dir was not installed by this script: left alone."
        fi
    done
    if grep -qF "$LAUNCHER_MARK" "$LAUNCHER_DIR/aurora-viewer" 2>/dev/null; then rm -f "$LAUNCHER_DIR/aurora-viewer"; fi
    if [[ -L $TRUST_HOST_LINK_DIR/$TRUST_HOST_NAME && $(readlink "$TRUST_HOST_LINK_DIR/$TRUST_HOST_NAME") == "$PREFIX/trust-host/$TRUST_HOST_NAME" ]]; then
        rm -f "$TRUST_HOST_LINK_DIR/$TRUST_HOST_NAME"
        rmdir "$TRUST_HOST_LINK_DIR" 2>/dev/null || true
    fi
    rm -f "$APP_DIR/$DESKTOP_ID" "$ICON_ROOT"/*/apps/"$ICON_NAME".png
    refresh_icon_cache
    refresh_menu
    info "Removed. Viewer data stays in ~/.config/auroraviewer, ~/.cache/auroraviewer and ~/.local/share/auroraviewer."
}

# ------------------------------------------------------------------- main

main() {
    parse_args "$@"
    if ((ROLLBACK)); then do_rollback; exit 0; fi
    if ((UNINSTALL)); then do_uninstall; exit 0; fi

    preflight
    if ((NO_BUILD)); then
        BIN=$BINARY
        SRC_TREE=$REPO
        warn "--no-build: the patches are not applied or verified for $BINARY."
    else
        "$BUILD_SH" "${BUILD_ARGS[@]}"
        locate_build
    fi
    install_files
    install_trust_host_link
    install_launcher
    install_icon
    write_desktop_entry
    refresh_icon_cache
    refresh_menu

    info ""
    if ((DRY)); then info "Dry run: nothing was changed."; fi
    info "Installed: $PREFIX/aurora-viewer"
    info "TRust host: $PREFIX/trust-host/$TRUST_HOST_NAME ($TRUST_HOST_VERSION),"
    info "  linked from $TRUST_HOST_LINK_DIR/$TRUST_HOST_NAME"
    info "Menu entry: $APP_DIR/$DESKTOP_ID"
    info "Roll back with: scripts/install-linux.sh --rollback"
    if command -v portmaster-start >/dev/null 2>&1 || [[ -d /var/lib/portmaster || -d /opt/safing/portmaster ]]; then
        info ""
        info "Portmaster: this is a new binary path. If web pages or media stay blank, allow"
        info "localhost for $PREFIX/aurora-viewer and the SLPlugin launcher in Portmaster."
        info "The TRust host fetches web pages itself: it needs its own Internet rule for"
        info "$PREFIX/trust-host/$TRUST_HOST_NAME (LAN and localhost can stay blocked)."
    fi
}

main "$@"
