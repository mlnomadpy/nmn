# Research data requirements driven by the Obsidian vault

This maps the seven kernel gap cards and selected method-reading records to
package capabilities. It does not claim every literature protocol is implemented
or that a source theorem has been transferred. Source titles below identify the
existing curated vault notes; their reading-depth qualifications remain binding.

| Research family and vault sources | Required data/interface | Current native NMN support | Remaining implementation |
|---|---|---|---|
| KG01 response spaces; finite edit coverage | Frozen model/input/edit identities, complete output responses, local bank geometry/norms, preparation/query costs | Native collector, replay table, local Gram/norm data, split-aware finite response subspace measurements | Valid whole-response norm budgets, unseen-edit prediction, matched cost benchmarks |
| KG02 closure through depth; predictive-state/reduction literature | Read/write maps, full pre/post states, action, reduced summaries, actual next state and summary residual | Explicit-state residual graphs, dependency maps, resumable suffixes, supplied-state response studies, supplied or fit-only PCA/ridge summaries with held-out next-state/output residuals, and full layer/module traces | Nonlinear/controlled summary adapters, invariant-domain and defect propagation |
| KG03 stable observability; kernel sensors; LEACE | Raw activations, sensor outputs, bank spectra, labels, covariance/noise/separation data | Kernel/sensor values, spectra, Jacobians, covariance, sampled separation, supplied noise-radius diagnostics, fit-only linear state probes with frozen and separately refitted held-out edit comparisons | Nonlinear probe comparators, sensor acquisition, stochastic-noise protocols, stable inverse bounds |
| KG04 interaction structure; Integrated Hessians; HVP patching correction | Joint effects, background-dependent finite differences, gradients, mixed curvature, path and quadrature data | Budgeted module-subset lattices on native graphs, explicit background gates, full-lattice subset coefficients, gate derivatives and joint directional path integration | Scalable HVP batching, sparse recovery and certified residuals |
| KG05 native protected edits; RAVEL; ACDC/EAP-IG | Native edit and donor coordinates, recomputed descendants, per-example target/protection outputs, eligibility | Native gates/replacements, receiving-module read-slot patches, producer-specific residual edge studies and declared donor-edge transfers with CLI/replay/export, donor studies with provenance, full traces and effects, declared classification tasks, per-example damage and semantic strata | Automatic donor selection, recursive path isolation, native feasibility search, source-faithful comparator adapters |
| KG06 acquisition/selection; Learn then Test | Population IDs, independent splits, per-candidate risk/uncertainty, selection/stopping history, full cost | Sample IDs, split/group checks, explicit donor access, executable budgeted candidate selection, frozen-winner validation, ledger, controls and per-example observations | Statistical validation and acquisition algorithms |
| KG07 semantics; causal abstraction/scrubbing; patching illusion | Semantic variables/reference outputs, donor equivalence groups, correspondence hypotheses, predicted/unforced outcomes | Supplied semantic dataset records, finite reference/correspondence tables, versioned reference callback, native donor transfer and counterfactual mismatch reports | Recursive scrubbing, semantic alignment and ambiguity controls |
| Kernelized erasure / preimage reconstruction | Feature projections, realizable native edits, residual feature and nonlinear suffix error | Kernel bank features and executed suffix responses | Projection/preimage solver and utility/probe comparators |
| NAM / IMQ / generic smooth comparison | Matched task, topology, supervision, fit budget, effects and serving costs | Native IMQ/tanh alternatives, CPU replay and explicit bounded training, parameter/precision/cost records | Task-matched fitting studies, acquisition/certification baselines |
| Formal verification literature (AutoBound, Beta-CROWN, certified fidelity) | Domain, arithmetic assurance, sound bounds, counterexamples and unresolved results | Exact rational enclosures, bounded output/output-difference verification with structural zero dependencies and search-independent partition checking for fixed Yat/IMQ graphs | Distinct-model relations, additional operations and runtime-roundoff bounds |
| Repetition/sequence protection literature | Tokenizer/model revision, prompts, decoding law, prefixes, rollout IDs, EOS/length and per-task losses | Not supported by the three-neuron collector | Sequence-model adapter and frozen decoding/evaluation contract |

## Common observation record

Every future adapter should retain model/configuration identity, source/runtime
versions, input/sample identity, intended measurement, controls, complete executed
responses, relevant internal state, supplied labels and their origin, population
and split assignment, selection history, arithmetic assurance and resource costs.
Raw per-example data must remain available even when summaries are produced.

The v1 collector supplies a subset of this record and accepts unvalidated metadata;
it does not enforce statistical independence or semantic validity. Each experiment
must state which fields and obligations remain absent. Extend package capabilities
against a concrete source protocol before describing a benchmark as reproduced.

## Implementation order

1. Native trainable model and numerical observations (implemented for PyTorch's
   three-module reference in this change).
2. General explicit-state residual graph and dataset/reference-model interfaces (implemented).
3. Donor/interchange and gate-path comparisons against direct replay, with a
   candidate ledger and split checks.
4. Training/evaluation lifecycle and matched module baselines once the task and
   supervision contract are selected.
5. Sound bounds and statistical validation as separately scoped adapters.

This is package work driven by research questions. Adding more test counts or
artifact copies is not a substitute for these capabilities.

## Protection task records

[Native protection studies](native-protection.md) execute edits under declared threshold/argmax tasks, preserve labels and full traces, and distinguish accuracy, conditional damage and disagreement with explicit eligible counts. Supplied semantic strata include missing-value groups. Statistical guarantees and sequence protocols remain open.

## Inspecting evidence in the vault

Use [native exports](native-vault-export.md) and the [offline dashboard](native-dashboard.md) to retain raw records beside their scopes and limitations. The report command filters supplied evidence; it does not assert coverage of unimplemented literature protocols.

[Native coalition studies](native-coalitions.md) retain partial coverage and per-example responses; absent coefficients in an incomplete lattice are not zero effects.

[Native replay](native-replay.md) recomputes supported saved observation, donor, protection and coalition records with explicit tolerances. File integrity, numerical reproduction and scientific validity are separate checks.

[Suffix-state studies](native-suffix.md) measure downstream responses and reconstruction error at declared graph boundaries; invariant domains and summary-map identification remain open.

[Semantic correspondence studies](native-semantics.md) separate ordinary predictions from counterfactual tests under supplied tables; unique identification and symbolic reference models remain open.

[Edit selection](native-selection.md) performs selection and frozen validation on separate declared splits. Statistical validation and acquisition algorithms remain separate open capabilities.

[Finite response-space measurements](native-response-space.md) record a fit-only SVD basis and evaluation residuals; they do not establish whole-response rank or reduce the number of evaluated edits.

[Native rational enclosures](native-enclosure.md) implement the supported arithmetic subset for issue #44. The local containment derivation and limitations are explicit; this does not complete the certificate/search requirements.

[Box range verification](native-interval-contract.md) adds complete partition records, exact witness checks and derived verified/counterexample/inconclusive outcomes. Its checker shares the rational arithmetic primitives; no external proof-assistant review is claimed.

[State-summary studies](native-reduction.md) execute supplied affine encoders, decoders and one-layer transitions, retaining separate reconstruction, summary-transition and downstream output residuals. Fitting and closure guarantees remain open.

[Internal-state probes](native-probes.md) preserve feature/label rows, fit-only ridge coefficients, held-out decoding, confusion counts and conditional damage. Probe failure is not an erasure certificate.
