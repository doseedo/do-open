#!/usr/bin/env bash
# Fast pre-deploy dependency sanity check. Catches the class of bug where
# a pip package on PyPI is missing a module the code imports (like the
# demucs==4.0.1 / demucs.api incident).
#
# Add a new `check_module` call whenever a deploy breaks because of a
# missing-module issue. 30 seconds of local checking beats 15 minutes of
# Modal rebuild for every reproduction.
#
# Usage: bash scripts/check-deps.sh
# Exit: 0 if all checks pass, 1 if any fails.

set -uo pipefail

FAIL=0
TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

# ─── helpers ────────────────────────────────────────────────────────────
blue()   { printf "\033[34m%s\033[0m\n" "$*"; }
green()  { printf "\033[32m✓ %s\033[0m\n" "$*"; }
red()    { printf "\033[31m✗ %s\033[0m\n" "$*"; }

# Download a pypi source dist WITHOUT installing. Usage:
#   check_sdist_contains <pkg==ver> <path-inside-package>
# Succeeds if the path exists in the tarball. Fails loudly otherwise.
check_sdist_contains() {
  local pkg="$1"
  local path="$2"
  local d="$TMP/$(echo "$pkg" | tr '=/' '__')"
  mkdir -p "$d"
  if ! pip download "$pkg" --no-deps --no-binary :all: -q --dest "$d" 2>/dev/null; then
    red "pip download $pkg failed"
    FAIL=1
    return
  fi
  local tarball=$(ls "$d"/*.tar.gz 2>/dev/null | head -1)
  if [[ -z "$tarball" ]]; then
    red "no sdist tarball found for $pkg"
    FAIL=1
    return
  fi
  if tar tzf "$tarball" 2>/dev/null | grep -q "$path"; then
    green "$pkg contains $path"
  else
    red "$pkg does NOT contain $path — install from git instead"
    FAIL=1
  fi
}

# Check a git-installed package has a given path at a given ref. Usage:
#   check_git_contains <user/repo> <ref> <path>
check_git_contains() {
  local repo="$1"
  local ref="$2"
  local path="$3"
  local code=$(curl -sS -o /dev/null -w "%{http_code}" \
    "https://raw.githubusercontent.com/${repo}/${ref}/${path}")
  if [[ "$code" == "200" ]]; then
    green "github.com/${repo}@${ref} has ${path}"
  else
    red "github.com/${repo}@${ref} does NOT have ${path} (HTTP $code)"
    FAIL=1
  fi
}

# ─── actual checks ──────────────────────────────────────────────────────
blue "=== Modal image dependency checks ==="

# demucs.api — was THE bug on 2026-04-17. PyPI 4.0.1 doesn't have it.
# We install from git+github. Verify the pinned SHA still has api.py.
DEMUCS_SHA=$(grep -oE 'adefossez/demucs(\.git)?@[a-f0-9]{40}' \
  /Users/hydroadmin/Downloads/Do/modal/modal_stemphonic.py | head -1 | cut -d@ -f2)
if [[ -n "$DEMUCS_SHA" ]]; then
  check_git_contains "adefossez/demucs" "$DEMUCS_SHA" "demucs/api.py"
else
  red "could not find pinned demucs SHA in modal_stemphonic.py"
  FAIL=1
fi

# Belt-and-suspenders: the PyPI 4.0.1 version continues to not have api.py.
# If someone ever reverts the git+github install, this will scream.
if grep -q '"demucs==4.0.1"' /Users/hydroadmin/Downloads/Do/modal/modal_stemphonic.py 2>/dev/null; then
  red "modal_stemphonic.py still references demucs==4.0.1 from PyPI — this is known-broken"
  FAIL=1
fi

# ─── done ───────────────────────────────────────────────────────────────
echo
if [[ $FAIL -eq 0 ]]; then
  green "all dependency checks passed"
  exit 0
else
  red "one or more dependency checks FAILED — fix before deploying"
  exit 1
fi
