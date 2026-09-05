#!/bin/bash
# install-hooks.sh — link tools/hooks/* into .git/hooks/
#
#   tools/install-hooks.sh          install (idempotent)
#   tools/install-hooks.sh --check   report drift, change nothing (exit 1 if wrong)
#
# WHY A SYMLINK AND NOT A COPY. git does not track `.git/hooks/`, so the
# pre-push hook — which is the only thing standing between a broken CUDA build
# and `main`, since CI has no CUDA runner — lived in exactly one place on one
# machine and would vanish on a re-clone. A COPY would survive that, but copies
# drift: this repo has spent real time on a stale `+nightly` in a script whose
# sibling had been fixed, and on a doc comment that outlived the code it
# described. A symlink cannot drift. Edit tools/hooks/pre-push and the installed
# hook changes with it, because it IS it.
#
# The tradeoff, stated so it is not a surprise: on a branch where
# tools/hooks/pre-push does not exist, the symlink dangles and git skips the
# hook silently. `--check` reports that.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 2

SRC_DIR="tools/hooks"
DST_DIR="$(git rev-parse --git-path hooks)"
CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1

rc=0
for src in "$SRC_DIR"/*; do
    [ -f "$src" ] || continue
    name=$(basename "$src")
    dst="$DST_DIR/$name"
    want="$(cd "$DST_DIR" && realpath --relative-to=. "$OLDPWD/$src" 2>/dev/null || echo "$PWD/$src")"

    if [ -L "$dst" ] && [ "$(readlink "$dst")" = "$want" ]; then
        if [ -e "$dst" ]; then
            echo "ok       $name -> $want"
        else
            echo "DANGLING $name -> $want (source missing on this branch; git will SKIP the hook)"
            rc=1
        fi
        continue
    fi

    if [ "$CHECK" = 1 ]; then
        if [ -e "$dst" ]; then
            echo "DRIFT    $name is a real file, not a link to $SRC_DIR — it can rot independently"
        else
            echo "MISSING  $name is not installed"
        fi
        rc=1
        continue
    fi

    # Never delete a hook without keeping it: an uninstalled local hook may be
    # someone's unpushed work, and this script is not the place to lose it.
    if [ -e "$dst" ] && [ ! -L "$dst" ]; then
        backup="$dst.replaced-$(date +%Y%m%d-%H%M%S)"
        mv "$dst" "$backup"
        echo "backed up existing $name -> $(basename "$backup")"
    fi
    ln -sfn "$want" "$dst"
    chmod +x "$src"
    echo "linked   $name -> $want"
done

if [ "$CHECK" = 1 ] && [ "$rc" = 0 ]; then
    echo "all hooks installed as links to $SRC_DIR"
fi
exit $rc
