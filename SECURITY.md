# Security Policy

## Supported Versions

NMN follows semantic versioning. Security fixes are applied to the latest minor release.

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, report them privately by emailing **taha@azetta.ai** with the subject line `[SECURITY] nmn`.

Please include as much of the following as you can:

- A description of the issue and its potential impact
- Steps to reproduce, including code, inputs, and environment (OS, Python version, framework version)
- Whether the issue is already public or known elsewhere
- Whether you would like to be credited in the disclosure

You should receive an acknowledgment within **72 hours**. We will follow up with a more detailed response within **7 days** indicating next steps.

## Disclosure Policy

- We will work with you to understand and reproduce the issue.
- We will prepare a fix in a private branch and coordinate a release.
- Once a fix is released, we will publicly disclose the vulnerability via a GitHub Security Advisory and a release note in [`CHANGELOG.md`](CHANGELOG.md).
- We aim for **90 days** from report to public disclosure as a maximum; faster if a patch is straightforward.

## Scope

In scope:

- The `nmn` Python package and its public API surface
- Build, packaging, and release configuration (`pyproject.toml`, `.github/workflows/`)
- Documentation that could lead to insecure usage

Out of scope:

- Vulnerabilities in upstream dependencies (PyTorch, JAX, TensorFlow, etc.) — report those to their respective projects
- Issues in user-provided code or model checkpoints
- Theoretical numerical-stability issues that do not lead to a security impact

Thank you for helping keep NMN and its users safe.
