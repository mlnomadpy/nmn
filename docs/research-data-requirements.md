# Research data requirements driven by the Obsidian vault

This maps the seven kernel gap cards and selected method-reading records to
package capabilities. It does not claim every literature protocol is implemented
or that a source theorem has been transferred. Source titles below identify the
existing curated vault notes; their reading-depth qualifications remain binding.

| Research family and vault sources | Required data/interface | Current native NMN support | Remaining implementation |
|---|---|---|---|
| KG01 response spaces; finite edit coverage | Frozen model/input/edit identities, complete output responses, local bank geometry/norms, preparation/query costs | Native collector, replay table, local Gram/norm data | Reduced-response spaces, valid whole-response norm budgets, matched cost benchmarks |
| KG02 closure through depth; predictive-state/reduction literature | Read/write maps, full pre/post states, action, reduced summaries, actual next state and summary residual | Explicit-state residual graphs, dependency maps and full layer/module traces | Summary adapters, invariant-domain and defect propagation |
| KG03 stable observability; kernel sensors; LEACE | Raw activations, sensor outputs, bank spectra, labels, covariance/noise/separation data | Kernel/sensor values, spectra, Jacobians, covariance, sampled separation and supplied noise-radius diagnostics | Probe fitting, sensor acquisition, stochastic-noise protocols, stable inverse bounds |
| KG04 interaction structure; Integrated Hessians; HVP patching correction | Joint effects, background-dependent finite differences, gradients, mixed curvature, path and quadrature data | All eight three-state coalitions, subset coefficients, gate derivatives and joint directional path integration | Scalable HVP batching, sparse recovery and certified residuals |
| KG05 native protected edits; RAVEL; ACDC/EAP-IG | Native edit and donor coordinates, recomputed descendants, per-example target/protection outputs, eligibility | Native gates/replacements, donor studies with provenance, full traces and effects, separate classification protection metrics | Arbitrary edge patching, native feasibility search, source-faithful comparator adapters |
| KG06 acquisition/selection; Learn then Test | Population IDs, independent splits, per-candidate risk/uncertainty, selection/stopping history, full cost | Sample IDs, split/group checks, explicit donor access, frozen-candidate ledger, controls and per-example observations | Statistical validation and acquisition algorithms |
| KG07 semantics; causal abstraction/scrubbing; patching illusion | Semantic variables/reference outputs, donor equivalence groups, correspondence hypotheses, predicted/unforced outcomes | Supplied semantic dataset records, versioned reference callback, native donor transfer | Recursive scrubbing, semantic alignment and ambiguity controls |
| Kernelized erasure / preimage reconstruction | Feature projections, realizable native edits, residual feature and nonlinear suffix error | Kernel bank features and executed suffix responses | Projection/preimage solver and utility/probe comparators |
| NAM / IMQ / generic smooth comparison | Matched task, topology, supervision, fit budget, effects and serving costs | Native IMQ/tanh alternatives, CPU replay and explicit bounded training, parameter/precision/cost records | Task-matched fitting studies, acquisition/certification baselines |
| Formal verification literature (AutoBound, Beta-CROWN, certified fidelity) | Domain, arithmetic assurance, sound bounds, counterexamples and unresolved results | Floating-point observations only | Interval/bound adapters and independently checked certificates |
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
