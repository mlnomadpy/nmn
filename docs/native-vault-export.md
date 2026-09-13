# Export native research records to Obsidian

```bash
nmn research native export observations.json --output research-vault/Experiments/my-run
nmn research native verify-export research-vault/Experiments/my-run
```

Both commands work without PyTorch. Export accepts native model, observation,
donor-study, gate-path, kernel-diagnostic, benchmark and training records. It
creates a **new** directory containing:

- `data.json`: an exact byte copy of the source record.
- `Report.md`: a readable Obsidian note with record-specific tables and scope.
- `manifest.json`: hashes of the data/note plus exporter identity.

No existing note or vault index is edited. Unsupported schemas and mismatched
embedded model parameter/configuration hashes are rejected. An existing output
path is refused. File creation failures remove the newly created partial export.

The note retains all benchmark methods and training seed outcomes, including
failures. It distinguishes missing task labels from measured errors, and nonfinite
measurements from finite numbers. Donor notes report reference error and protected
change separately. Gate-path notes report prediction residuals without presenting
them as certified bounds. Raw per-example data remains linked for full analysis.

`verify-export` checks file hashes, declared record schema and embedded model
identities. Its output explicitly says `recomputed: false`. This is stored-file
integrity checking, not computational reproduction, authorship verification,
semantic validation or an independently checked mathematical certificate. It does
not load model code, rerun training or update a result's verdict. Changes to both
a file and its manifest are not prevented by unsigned hashes.

Python integrations can use `nmn.research.native_export.export_native_record`,
`verify_native_export`, or `render_native_note`. The renderer does not mutate the
record and preserves recorded limitations. Exported Markdown is a summary, not a
replacement for the complete numerical record or the canonical research notes.
