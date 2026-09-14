# Explicit linear modules in native graphs

`nmn.torch.LinearExpansion` adds an interpretable linear path that can coexist
with Yat, IMQ and tanh modules in `YatGraph`:

```python
YatModuleSpec("p", reads=("v",), writes=("protected",), family="linear")
```

The module computes `sum_i coefficient[j,i] * dot(x, center[i])`, exposing each
signed contribution. `effective_weight = coefficients @ centers` describes the
represented linear map. Centers and coefficients are trainable; there are no
biases or hidden nonlinearities. `num_centers` controls the factorization width
and bounds its rank. `epsilon` remains a graph-spec field but is unused by this
family. Equal factorization width does not imply equal expressiveness with a
nonlinear module.

Collector geometry includes the center Gram matrix, effective linear weight and
local RKHS inner products for the ordinary dot-product kernel. Those norms refer
to the individual module, not a composed nonlinear graph. The ordinary graph
interfaces supply gates, replacements, read/edge patches, live derivatives,
snapshot restore, training, numerical replay and benchmark execution.
The rational enclosure adapter currently rejects this family explicitly; adding
a linear module does not silently extend the certification arithmetic contract.

`examples/research/linear-protected-graph.json` is a mixed graph configuration
accepted by `nmn research native init --graph ...`. Its initialization is random
under the existing graph initializer. To specify a known scalar identity, set
both the one-center prototype and its coefficient to one; this makes p(v)=v.
That explicit assignment is not evidence that training learned the identity.

The matched training study exposed poor Yat-only fitting of a linear protected
target under its fixed small budget. This component enables a follow-up
architecture study. It does not retrospectively improve those results or prove
that the hybrid architecture trains better. Such comparisons require newly
specified fitting/selection and untouched evaluation populations.
