from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
SKILL = ROOT / ".claude" / "skills" / "nmn" / "SKILL.md"


def test_nmn_skill_has_portable_frontmatter_and_existing_references():
    text = SKILL.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    frontmatter = text.split("---\n", 2)[1]
    keys = {
        line.split(":", 1)[0]
        for line in frontmatter.splitlines()
        if line and not line.startswith(" ")
    }
    assert keys == {"name", "description"}
    assert "name: nmn" in frontmatter

    references = re.findall(r"\]\((references/[^)]+\.md)\)", text)
    assert len(references) == 7
    assert len(references) == len(set(references))
    for relative in references:
        assert (SKILL.parent / relative).is_file(), relative


def test_nmn_skill_covers_every_backend_and_constructor_gotcha():
    text = SKILL.read_text(encoding="utf-8")
    references = {
        "torch": "pytorch.md",
        "nnx": "flax-nnx.md",
        "linen": "flax-linen.md",
        "keras": "keras.md",
        "tf": "tensorflow.md",
        "mlx": "mlx.md",
    }
    for backend in ("torch", "nnx", "linen", "keras", "tf", "mlx"):
        assert f"nmn[{backend}]" in text
        assert f"references/{references[backend]}" in text
    for spelling in (
        "in_features=128, out_features=64",
        "YatNMN(128, 64, rngs=nnx.Rngs(0))",
        "YatNMN(features=64)",
        "YatNMN(units=64)",
    ):
        assert spelling in text


def test_agent_handoff_copies_the_complete_skill_bundle():
    instructions = (ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert "entire `nmn`" in instructions
    assert "~/.codex/skills/nmn" in instructions
    assert "~/.claude/skills/nmn" in instructions


def test_nmn_skill_locks_current_numeric_and_mask_contracts():
    text = SKILL.read_text(encoding="utf-8")
    shared = (SKILL.parent / "references" / "shared-contract.md").read_text(
        encoding="utf-8"
    )
    combined = text + shared
    for contract in (
        "fully masked",
        "exact-zero",
        "FP16",
        "BF16",
        "preserves genuine NaNs",
        "learnable_epsilon=True",
        "freezes only the kernel",
    ):
        assert contract in combined
