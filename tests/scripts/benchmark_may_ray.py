"""Cross-framework benchmark for MAY / RAY / SLAY spherical-Yat feature maps.

Reproduces the headline result of issue #36: in the deployment regime (bias
``b > 0``) the bias-aware **MAY** (Random Maclaurin) feature map is near-exact and
beats the bias-free **SLAY** anchor method, which floors at a wrong-kernel ceiling.
**RAY** (radial) is the weaker secondary method.

The ground-truth kernel is computed directly in NumPy (bias-aware, with
``eps = median squared distance`` — the Yat epsilon heuristic), so the benchmark
measures *feature-map quality* and is identical across frameworks. Each installed
framework's feature maps are then evaluated against that shared ground truth.

Run:  python3 tests/scripts/benchmark_may_ray.py
Frameworks that are not installed are skipped automatically.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Shared, verified ground truth (bias-aware exact spherical-Yat attention)
# --------------------------------------------------------------------------
D, N = 64, 512
B_VALUES = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
SEEDS = [0, 1, 2]
F = 256  # matched feature budget


def _normalize(x, eps=1e-6):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + eps)


def _median_sq_distance(qn, kn):
    diff = qn[:, None, :] - kn[None, :, :]
    return float(np.median((diff * diff).sum(-1)))


def _exact_attention(q, k, v, b, eps):
    qn, kn = _normalize(q), _normalize(k)
    s = qn @ kn.T
    w = (s + b) ** 2 / ((2.0 + eps) - 2.0 * s)
    w = w / (w.sum(-1, keepdims=True) + 1e-9)
    return w @ v


def _per_token_cos(a, b):
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
    return float(np.mean(num / den))


def _rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))


# --------------------------------------------------------------------------
# Per-framework adapters: numpy [N, d] -> framework -> numpy attention output.
# Each returns dict {method_name: callable(q, k, v, b, eps, seed) -> np [N, d]}
# or None if the framework is not importable.
# --------------------------------------------------------------------------
def _reshape_in(x):
    # attention APIs use [batch, seq, heads, head_dim]; we use B=1, H=1.
    return x.reshape(1, x.shape[0], 1, x.shape[1])


def _reshape_out(y):
    return np.asarray(y).reshape(N, D)


def adapter_nnx():
    try:
        import jax.numpy as jnp
        from jax import random
        from nmn.nnx.layers.attention import maclaurin_yat as m, radial_yat as r
        from nmn.nnx.layers.attention import spherical_yat_performer as s
    except Exception:
        return None

    def to(x):
        return jnp.asarray(_reshape_in(x))

    def may(q, k, v, b, eps, seed):
        p = m.create_maclaurin_projection(random.PRNGKey(seed), D, num_features=F, bias=b, epsilon=eps)
        return _reshape_out(m.maclaurin_yat_attention(to(q), to(k), to(v), p))

    def ray(q, k, v, b, eps, seed):
        p = r.create_radial_projection(random.PRNGKey(seed), D, sketch_m=8, num_radial=4, radial_dim=8, bias=b, epsilon=eps)
        return _reshape_out(r.radial_yat_attention(to(q), to(k), to(v), p))

    def slay(q, k, v, b, eps, seed):
        p = s.create_yat_tp_projection(random.PRNGKey(seed), D, num_prf_features=8, num_quad_nodes=1, num_anchor_features=32, epsilon=eps)
        return _reshape_out(s.yat_tp_attention(to(q), to(k), to(v), p))

    return {"MAY": may, "RAY": ray, "SLAY": slay}


def adapter_linen():
    try:
        import jax.numpy as jnp
        from jax import random
        from nmn.linen import performer_yat as p
    except Exception:
        return None

    def to(x):
        return jnp.asarray(_reshape_in(x))

    def may(q, k, v, b, eps, seed):
        pr = p.create_maclaurin_projection(random.PRNGKey(seed), D, num_features=F, bias=b, epsilon=eps)
        return _reshape_out(p.maclaurin_yat_attention(to(q), to(k), to(v), pr))

    def ray(q, k, v, b, eps, seed):
        pr = p.create_radial_projection(random.PRNGKey(seed), D, sketch_m=8, num_radial=4, radial_dim=8, bias=b, epsilon=eps)
        return _reshape_out(p.radial_yat_attention(to(q), to(k), to(v), pr))

    return {"MAY": may, "RAY": ray}


def adapter_torch():
    try:
        import torch
        from nmn.torch.attention import performer_yat as p
    except Exception:
        return None

    def to(x):
        return torch.from_numpy(_reshape_in(x).astype(np.float32))

    def may(q, k, v, b, eps, seed):
        torch.manual_seed(seed)
        pr = p.create_maclaurin_projection(D, num_features=F, bias=b, epsilon=eps, seed=seed)
        return _reshape_out(p.maclaurin_yat_attention(to(q), to(k), to(v), pr).detach())

    def ray(q, k, v, b, eps, seed):
        pr = p.create_radial_projection(D, sketch_m=8, num_radial=4, radial_dim=8, bias=b, epsilon=eps, seed=seed)
        return _reshape_out(p.radial_yat_attention(to(q), to(k), to(v), pr).detach())

    return {"MAY": may, "RAY": ray}


def adapter_mlx():
    try:
        import mlx.core as mx
        from nmn.mlx import may as mmay, ray as mray, performer as mslay
    except Exception:
        return None

    def to(x):
        return mx.array(_reshape_in(x).astype(np.float32))

    def may(q, k, v, b, eps, seed):
        pr = mmay.create_maclaurin_projection(D, num_features=F, bias=b, epsilon=eps, seed=seed)
        return _reshape_out(mmay.maclaurin_yat_attention(to(q), to(k), to(v), pr))

    def ray(q, k, v, b, eps, seed):
        pr = mray.create_radial_projection(D, sketch_m=8, num_radial=4, radial_dim=8, bias=b, epsilon=eps, seed=seed)
        return _reshape_out(mray.radial_yat_attention(to(q), to(k), to(v), pr))

    def slay(q, k, v, b, eps, seed):
        pr = mslay.create_yat_tp_projection(D, num_prf_features=8, num_quad_nodes=1, num_anchor_features=32, epsilon=eps, seed=seed)
        return _reshape_out(mslay.yat_tp_attention(to(q), to(k), to(v), pr))

    return {"MAY": may, "RAY": ray, "SLAY": slay}


ADAPTERS = {
    "nnx": adapter_nnx,
    "linen": adapter_linen,
    "torch": adapter_torch,
    "mlx": adapter_mlx,
    # tf / keras require tensorflow; their adapters mirror the same API and run in CI.
}


def run():
    print("MAY / RAY / SLAY spherical-Yat feature-map benchmark")
    print(f"d={D}  N={N}  matched F={F}  eps=median-sq-dist  {len(SEEDS)} seeds")
    print("(MAY should be near-exact and beat SLAY at b>=1; SLAY wins only at b=0)\n")

    for name, build in ADAPTERS.items():
        methods = build()
        if methods is None:
            print(f"[{name}] not installed — skipped\n")
            continue
        cols = list(methods.keys())
        print(f"### {name}")
        print(f"{'b':>5} | " + " | ".join(f"{c+' cos/relL2':>16}" for c in cols))
        print("-" * (8 + 19 * len(cols)))
        for b in B_VALUES:
            agg = {c: [[], []] for c in cols}
            for seed in SEEDS:
                rng = np.random.default_rng(100 + seed)
                q = rng.standard_normal((N, D))
                k = rng.standard_normal((N, D))
                v = rng.standard_normal((N, D))
                eps = _median_sq_distance(_normalize(q), _normalize(k))
                exact = _exact_attention(q, k, v, b, eps)
                for c in cols:
                    out = methods[c](q, k, v, b, eps, seed)
                    agg[c][0].append(_per_token_cos(out, exact))
                    agg[c][1].append(_rel_l2(out, exact))
            cells = []
            for c in cols:
                cells.append(f"{np.mean(agg[c][0]):5.2f} / {np.mean(agg[c][1]):5.2f}")
            print(f"{b:5.2f} | " + " | ".join(f"{x:>16}" for x in cells))
        print()


if __name__ == "__main__":
    run()
