# Public mirror synchronization

The `mlnomadpy/nmn` mirror synchronizes from the public canonical history with
the hourly and manually dispatched `Sync public mirror` workflow. The workflow
fetches `azettaai/nmn` anonymously and can write only to the mirror.

## Mirror-only GitHub App

GitHub's workflow `GITHUB_TOKEN` cannot update files below `.github/workflows/`.
Configure a GitHub App as follows instead:

1. Grant the app repository-level **Contents: write** and **Workflows: write**
   permissions, with no organization permissions.
2. Install it on **only** `mlnomadpy/nmn`.
3. Set the mirror repository variable `MIRROR_APP_CLIENT_ID` to the app's client
   ID and the Actions secret `MIRROR_APP_PRIVATE_KEY` to its private key.

The workflow requests a short-lived installation token with only those two
permissions. It deliberately omits `owner` and `repositories` from
`actions/create-github-app-token`, which scopes the token to the current mirror
repository. The action revokes the token after the job. The workflow's own
`GITHUB_TOKEN` remains read-only.

Checkout credential persistence is disabled. The installation token is passed
only in a dedicated HTTPS push URL for `mlnomadpy/nmn`; neither the anonymous
canonical fetch nor the remote-ref verification inherits an authorization
header.

## Safety properties

`scripts/sync-public-mirror.sh` refuses a non-fast-forward `master` update.
Canonical tags are fetched without force, so an attempt to change an existing
tag fails before the push. The branch and tags are then sent in one atomic push:
if GitHub rejects any ref, none are updated. After pushing, the script verifies
the remote branch and every canonical tag by object ID. A canonical commit that
changes a workflow file is otherwise an ordinary fast-forward and is supported
by the mirror-only app's Workflows permission.
