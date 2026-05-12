<!--
Thanks for sending a pull request! Please make sure you've read CONTRIBUTING.md first.
-->

## Summary

<!-- What does this PR do? Why is it needed? -->

## Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation only
- [ ] Tests only
- [ ] Build / packaging / CI

## Frameworks affected

<!-- Check all that apply -->

- [ ] PyTorch (`nmn.torch`)
- [ ] Flax NNX (`nmn.nnx`)
- [ ] Flax Linen (`nmn.linen`)
- [ ] Keras (`nmn.keras`)
- [ ] TensorFlow (`nmn.tf`)
- [ ] All / cross-framework

## Test plan

<!-- How did you verify this? -->

- [ ] Added/updated unit tests under `tests/test_<framework>/`
- [ ] Added/updated cross-framework consistency tests under `tests/integration/` (if behavior is shared)
- [ ] Ran `pytest tests/test_<framework>/ -v` locally
- [ ] Ran `black src/ tests/` and `isort src/ tests/`

## Checklist

- [ ] My code follows the style guidelines (`black`, `isort`)
- [ ] I have updated [`CHANGELOG.md`](../CHANGELOG.md) under `## [Unreleased]`
- [ ] I have updated the [`README.md`](../README.md) Layer Support Matrix (if adding a new layer)
- [ ] I have updated [`EXAMPLES.md`](../EXAMPLES.md) or `docs/guides/` (if adding a new feature)
- [ ] I have added type hints to public APIs
- [ ] No new warnings introduced in test output

## Related issues

<!-- Closes #123, Fixes #456, Relates to #789 -->
