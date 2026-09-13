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
