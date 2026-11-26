#!/usr/bin/env bash
# Build Aurora Viewer (release) from this checkout plus the private patch set,
# in a dedicated build tree. Usable alone; scripts/install-linux.sh calls it
# and then installs the result.
#
# Why a build tree: the checkout running this script stays clean. The patches
# are applied to a dedicated detached git worktree (../work/aurora-install,
# next to the main checkout), reset to the chosen ref on every run. Its
# target/ folder is kept, so a rebuild is incremental.
#
# Three things to remember on this PC, done here: use rustup's toolchain, keep
# the cargo build under a memory cap, and run one build at a time (a lock
# shared by every checkout and worktree, held from the tree reset to the end
# of cargo).
#
# What the build leaves behind, for the caller (install-linux.sh):
#   <tree>/target/release/aurora-viewer    the executable
#   <tree>/.scratch/patch-report.txt       what apply_patches.py reported
# --print-tree and --print-binary print those locations (and do nothing else),
# so nobody has to repeat the default build tree rules.
#
# Linux only. Never runs sudo, never edits a shell profile.
#
# Environment overrides:
#   AURORA_BUILD_MEM   memory cap of the cargo build, a systemd size (default: 20G)
#   AURORA_BUILD_JOBS  parallel cargo jobs (default: 4)
#
# Usage: scripts/build.sh --help
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
PATCH_TOOL="$REPO/scripts/patches/apply_patches.py"

TREE_MARK="aurora-local-install-tree"
CARGO="$HOME/.cargo/bin/cargo"
MEM="${AURORA_BUILD_MEM:-20G}"
JOBS="${AURORA_BUILD_JOBS:-4}"
# Rustup's cargo looks up rustc on PATH, so Debian's /usr/bin/rustc (too old
# for this workspace) would otherwise win.
export PATH="$HOME/.cargo/bin:$PATH"

REF="HEAD"
BUILD_TREE=""
SKIPS=()
DRY=0
DO_FETCH=0
PRINT=""

# set while running
BIN=""
WORK_DIR=""
COMMON_DIR=""

info() { printf '%s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: scripts/build.sh [options]

Builds Aurora Viewer in release mode from a dedicated build tree and applies
the local patches (scripts/patches). Prints the path of the executable at the
end. To install the result too, use scripts/install-linux.sh.

Options:
  --ref REF            git ref to build (default: HEAD of this checkout)
  --fetch              run "git fetch origin" first (read-only); use with
                       --ref origin/main. Nothing is fetched otherwise.
  --skip ID            skip one patch (repeatable), passed to apply_patches.py
  --build-tree DIR     build tree (default: ../work/aurora-install next to the
                       main checkout)
  --print-tree         only print the build tree path, build nothing
  --print-binary       only print the path the built executable has, build
                       nothing
  --dry-run            print what would be done, change nothing
  -h, --help           this help

Environment: AURORA_BUILD_MEM (memory cap of the cargo build, default 20G),
AURORA_BUILD_JOBS (parallel cargo jobs, default 4).

Needs ~/.cargo/bin/cargo (rustup), python3 >= 3.11, git, pkg-config and the
alsa and libudev development packages. It never calls sudo.
EOF
}

# ---------------------------------------------------------------- arguments

parse_args() {
    while (($#)); do
        case "$1" in
        --ref) REF=${2:?--ref needs a value}; shift 2 ;;
        --fetch) DO_FETCH=1; shift ;;
        --skip) SKIPS+=("${2:?--skip needs an id}"); shift 2 ;;
        --build-tree) BUILD_TREE=${2:?--build-tree needs a folder}; shift 2 ;;
        --print-tree) PRINT=tree; shift ;;
        --print-binary) PRINT=binary; shift ;;
        --dry-run) DRY=1; shift ;;
        -h | --help) usage; exit 0 ;;
        *) die "unknown option: $1 (see --help)" ;;
        esac
    done
}

# ----------------------------------------------------------------- helpers

abs_dir() { (cd "$1" 2>/dev/null && pwd -P); }

# Where the build tree and the lock live: next to the main checkout (the
# parent of the git common dir), not next to a worktree running this script.
resolve_paths() {
    local main_repo
    COMMON_DIR=$(git -C "$REPO" rev-parse --path-format=absolute --git-common-dir)
    main_repo=$(dirname "$COMMON_DIR")
    WORK_DIR="$(abs_dir "$main_repo/..")/work"
    if [[ -z $BUILD_TREE ]]; then BUILD_TREE="$WORK_DIR/aurora-install"; fi
    BIN="$BUILD_TREE/target/release/aurora-viewer"
}

# -------------------------------------------------------------- preflight

preflight() {
    local missing=()
    command -v git >/dev/null 2>&1 || die "git is not installed (sudo apt install git)"
    command -v python3 >/dev/null 2>&1 || die "python3 is not installed (sudo apt install python3)"
    python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' ||
        die "python3 >= 3.11 is needed (found $(python3 --version 2>&1))"
    # The release workflow also installs libxkbcommon-dev and libwayland-dev;
    # a build here does not need them (winit loads them at run time).
    command -v pkg-config >/dev/null 2>&1 || missing+=(pkg-config)
    if command -v pkg-config >/dev/null 2>&1; then
        pkg-config --exists alsa || missing+=(libasound2-dev)
        pkg-config --exists libudev || missing+=(libudev-dev)
    fi
    if ((${#missing[@]})); then
        info "Missing build packages. Install them, then run this script again:"
        info "  sudo apt install ${missing[*]}"
        die "build dependencies are missing (this script never runs sudo)"
    fi
    if [[ ! -x $CARGO ]]; then
        die "$CARGO not found. Install the Rust toolchain with rustup (https://rustup.rs). /usr/bin/cargo is too old and is never used."
    fi
}

# ------------------------------------------------------------- build tree

prepare_build_tree() {
    local commit gitdir

    if ((DO_FETCH)); then
        info "Fetching origin (read-only)..."
        if ((DRY)); then info "[dry-run] git fetch origin"; else git -C "$REPO" fetch origin; fi
    fi
    commit=$(git -C "$REPO" rev-parse --verify --quiet "${REF}^{commit}") || die "unknown git ref: $REF"
    info "Build ref: $REF ($(git -C "$REPO" rev-parse --short=9 "$commit"))"
    info "Build tree: $BUILD_TREE"

    if [[ ! -e $BUILD_TREE ]]; then
        if ((DRY)); then
            info "[dry-run] git worktree add --detach $BUILD_TREE $commit"
            return 0
        fi
        mkdir -p "$(dirname "$BUILD_TREE")"
        git -C "$REPO" worktree prune
        git -C "$REPO" worktree add --quiet --detach "$BUILD_TREE" "$commit"
        gitdir=$(git -C "$BUILD_TREE" rev-parse --absolute-git-dir)
        : >"$gitdir/$TREE_MARK"
    else
        verify_own_worktree
        if ((DRY)); then
            info "[dry-run] git reset --hard $commit && git clean -fd (in $BUILD_TREE)"
            return 0
        fi
        git -C "$BUILD_TREE" reset --hard --quiet "$commit"
        git -C "$BUILD_TREE" clean -fdq # no -x: target/ and .scratch/ stay
    fi
}

# Refuse to reset anything that is not this script's own build worktree.
verify_own_worktree() {
    local top gitdir tree_common
    top=$(git -C "$BUILD_TREE" rev-parse --show-toplevel 2>/dev/null) ||
        die "$BUILD_TREE exists but is not a git checkout; refusing to touch it"
    [[ $(abs_dir "$top") == "$(abs_dir "$BUILD_TREE")" ]] ||
        die "$BUILD_TREE is inside another git checkout ($top); refusing to touch it"
    gitdir=$(git -C "$BUILD_TREE" rev-parse --absolute-git-dir)
    tree_common=$(git -C "$BUILD_TREE" rev-parse --path-format=absolute --git-common-dir)
    [[ $tree_common == "$COMMON_DIR" && $gitdir != "$COMMON_DIR" ]] ||
        die "$BUILD_TREE is not a worktree of $REPO; refusing to touch it"
    [[ -e $gitdir/$TREE_MARK ]] ||
        die "$BUILD_TREE has no $TREE_MARK marker in its git dir: it was not created by this script. Remove it yourself (git worktree remove) or pick another --build-tree."
    if git -C "$BUILD_TREE" symbolic-ref -q HEAD >/dev/null; then
        die "$BUILD_TREE is on a branch, not detached; refusing to reset it"
    fi
}

# ---------------------------------------------------------------- patches

apply_patches() {
    local args=(--root "$BUILD_TREE") id rc=0 report
    for id in "${SKIPS[@]}"; do args+=(--skip "$id"); done
    [[ -f $PATCH_TOOL ]] || die "$PATCH_TOOL not found (run this script from a checkout that has the patch set)"
    if ((DRY)); then
        info "[dry-run] python3 $PATCH_TOOL ${args[*]}"
        return 0
    fi
    mkdir -p "$BUILD_TREE/.scratch"
    report="$BUILD_TREE/.scratch/patch-report.txt"
    info "Applying local patches..."
    python3 "$PATCH_TOOL" "${args[@]}" >"$report" 2>&1 || rc=$?
    cat "$report"
    if ((rc != 0)); then
        die "apply_patches.py exited with $rc (report above). Not building: a build without its patches must not be installed."
    fi
    # exit 0 with nothing applied means no manifest was found: same danger
    grep -Eq '^(APPLIED|ALREADY)' "$report" ||
        die "no patch was applied (no manifest in $REPO/scripts/patches?). Not building: a build without its patches must not be installed."
}

# ------------------------------------------------------------------ build

# One cargo build at a time on this PC (the lock is shared by every checkout
# and worktree), and no run resets the build tree or re-applies the patches
# under a build still compiling it: the lock is held from the tree reset to
# the end of cargo, on descriptor 9 until this script exits.
take_lock() {
    mkdir -p "$WORK_DIR"
    exec 9>>"$WORK_DIR/.cargo.lock"
    if ! flock -n 9; then
        info "waiting for another cargo build to finish…"
        flock 9
    fi
}

build() {
    if ((DRY)); then
        info "[dry-run] $CARGO build --release --locked -p aurora-viewer (in $BUILD_TREE)"
        return 0
    fi
    # /tmp is noexec here: build scripts must run from a folder we control.
    mkdir -p "$BUILD_TREE/.scratch/tmp" "$WORK_DIR"
    # Like the release workflow: no debug info in the binary. The version
    # comes from git describe, so a stray tag variable must not leak in.
    # PATH is passed through env explicitly so it surely reaches the build
    # inside the systemd scope.
    local cmd=(env -u AURORA_VERSION_TAG -u AURORA_UPDATE_TEST_VERSION
        "PATH=$PATH" "CARGO_BUILD_JOBS=$JOBS"
        "TMPDIR=$BUILD_TREE/.scratch/tmp" CARGO_PROFILE_RELEASE_STRIP=debuginfo
        "$CARGO" build --release --locked -p aurora-viewer)
    # Why the cap: a full parallel release build once froze the desktop.
    # The scope makes the kernel kill the build instead of swamping the machine.
    if systemd-run --user --scope -q true >/dev/null 2>&1; then
        cmd=(systemd-run --user --scope -q -p "MemoryMax=$MEM" "${cmd[@]}")
    else
        warn "systemd-run --user is not available: building without the $MEM memory cap."
    fi
    info "Building (release). The first build takes a long time; later ones are incremental."
    # 9>&-: cargo does not inherit the lock, so a background process it starts
    # (a compiler cache server, for instance) cannot keep it after this exits.
    (
        cd "$BUILD_TREE"
        "${cmd[@]}" 9>&-
    ) || die "cargo build failed"
    [[ -x $BIN ]] || die "build finished but $BIN is missing"
}

# ------------------------------------------------------------------- main

main() {
    parse_args "$@"
    resolve_paths
    case "$PRINT" in
    tree) printf '%s\n' "$BUILD_TREE"; exit 0 ;;
    binary) printf '%s\n' "$BIN"; exit 0 ;;
    esac

    preflight
    if ((!DRY)); then take_lock; fi
    prepare_build_tree
    apply_patches
    build
    if ((!DRY)); then info "Built: $BIN"; fi
}

main "$@"
