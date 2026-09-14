# Offline evidence dashboard

Build a portable research index from saved NMN records:

```bash
nmn research native report experiments/ another-run.json --output evidence-dashboard
```

Open `evidence-dashboard/index.html` in a browser. No server, network, PyTorch
installation or publication is required. Move the entire directory to keep links
working. The output directory must be new; reports never overwrite evidence.

Filter by architecture, backend, contract, scope and status, or search records.
Select a result to inspect identity, coverage, limitations and available data.
Each recognized record has a byte-preserved JSON copy and an Obsidian Markdown
note. `catalog.json` exposes the index to other tools. Directory scans skip
unrelated JSON schemas; missing, malformed or unsupported explicit inputs appear
as unavailable. Identical evidence bytes are indexed once.

Supported inputs are the native model/research schemas and
`nmn.finite-contract-evidence.v1`. Native model identities and supplied native
export manifests are checked. Exact finite outcomes retain their saved status;
the dashboard does not replay them or issue certificates. Source modification
age describes file freshness, not the time the model ran. Training completion
and numerical observations are not proofs. Invalid records remain visible.

Labels are rendered as text, embedded JSON is escaped, and no remote assets are
loaded. This report indexes supplied evidence, not every paper's research
protocol. See [the requirements map](research-data-requirements.md) for outstanding
measurement and inference capabilities.

Result previews expose coalition coverage and coefficient availability, donor transfer outcomes, and separate protection accuracy/damage/disagreement rates. Donor/protection previews show at most 25 rows with the full row count; copied JSON retains every row. Replay entries inherit their executed model architecture for filtering.

Certificate rows retain their saved outcome and explicitly state that the dashboard has not checked them. Saved checker reports display the derived contract outcome (including counterexamples and inconclusive coverage) and state that the checker was not rerun. All rational enclosure/certificate types are labeled with an exact-rational backend.

## Rebuild from a portable source list

Save a JSON file with paths relative to that file's directory:

```json
{"schema": "nmn.evidence-sources.v1", "sources": ["run-a/data.json", "run-b/data.json"]}
```

```bash
nmn research native report --sources-file evidence-sources.json --output next-dashboard
```

Explicit positional sources can be combined with this file. Only local paths
are accepted; no network fetch occurs. Missing evidence files become unavailable
entries in the report. Invalid source-list syntax/schema is an input error and
creates no report. A source list is a reusable selection, not an integrity
manifest or a declaration that every literature requirement is covered.

Probe previews separate baseline, frozen edited, and optionally refitted edited
accuracy. They retain eligibility counts and conditional damage. Missing refitted
results are null, not zero accuracy. At most 25 edits are previewed; copied JSON
retains every edit, score, prediction and trace.
