#!/usr/bin/env bash
set -euo pipefail

# Remotes are configurable only so the same transaction can be exercised
# against local bare repositories. The Actions workflow uses these defaults.
canonical_remote="${CANONICAL_REMOTE:-canonical}"
mirror_remote="${MIRROR_REMOTE:-origin}"
mirror_push_remote="${MIRROR_PUSH_REMOTE:-${mirror_remote}}"
branch="${MIRROR_BRANCH:-master}"
canonical_ref="refs/remotes/${canonical_remote}/${branch}"
mirror_ref="refs/remotes/${mirror_remote}/${branch}"

# Fetching canonical tags without force fails before any push if an existing
# mirror tag has the same name but a different object (tag rewrites are refused).
git fetch --prune "${canonical_remote}" "${branch}" --tags
mirror_head="$(git rev-parse "${mirror_ref}")"
canonical_head="$(git rev-parse "${canonical_ref}")"
if ! git merge-base --is-ancestor "${mirror_head}" "${canonical_head}"; then
  echo "Mirror ${branch} has diverged from canonical ${branch}." >&2
  exit 1
fi

git push --atomic "${mirror_push_remote}" \
  "${canonical_ref}:refs/heads/${branch}" --tags

mirrored_head="$(git ls-remote "${mirror_remote}" "refs/heads/${branch}" | awk '{print $1}')"
test -n "${mirrored_head}"
test "${mirrored_head}" = "${canonical_head}"

while IFS= read -r tag_ref; do
  canonical_tag="$(git rev-parse "${tag_ref}")"
  mirrored_tag="$(git ls-remote "${mirror_remote}" "${tag_ref}" | awk '{print $1}')"
  test -n "${mirrored_tag}"
  test "${mirrored_tag}" = "${canonical_tag}"
done < <(git for-each-ref --format='%(refname)' refs/tags)
