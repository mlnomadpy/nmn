# Independent repository maintenance

As of the 2026-09-13 separation, `mlnomadpy/nmn` is maintained independently
from `azettaai/nmn`. Neither repository is the canonical upstream of the other.
The former scheduled/manual mirror workflow and synchronization script have
been removed. No branches or tags are automatically imported from the other
repository. Existing history and published tags are preserved.

Open issues and pull requests in `mlnomadpy/nmn` for this distribution.
Future cross-repository changes require an explicit reviewed contribution with
compatible licensing; do not restore automatic branch or tag synchronization.

The old mirror documentation URL is retained to explain the transition.
Changes in this repository do not relicense or modify `azettaai/nmn`.

## External integrations

Repository links and GitHub Pages configuration target `mlnomadpy/nmn`.
Codecov upload steps retain their previous repository guard and therefore stay
inactive here until enrollment for this repository is verified; the local
coverage gate still runs. Package publishing remains tag-triggered; this
separation does not create a release, change PyPI ownership or configure new
trusted publishers. Published packages retain the license shipped with them.
