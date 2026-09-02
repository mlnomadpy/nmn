# Website dependency security

The documentation site is built from reviewed files in this repository. CI
runs `npm run audit:ci` after every clean install and rejects critical findings
or any high-severity advisory that is not listed in the audit script.

## Applied remediations

- `serialize-javascript` is overridden to 7.1.1, removing the RCE and CPU
  exhaustion advisories inherited through Docusaurus's webpack plugins.
- `uuid` is overridden to 11.1.1, the first patched release for the buffer
  bounds advisory inherited through `sockjs`.
- `qs` is overridden to 6.16.0, the first patched release for
  `GHSA-4mjr-xmp4-gh2g`, inherited through the Docusaurus development server.

Both overrides are exercised by the production Docusaurus build. Dependabot
continues to update the npm lockfile weekly.

## Reviewed residual advisories

Docusaurus 3.10.2 currently resolves `image-size` 2.0.2, which is also the latest
published release. No patched version exists for these build-time parser denial
of service advisories:

- `GHSA-w3rx-r6r6-pgpr` — ICNS parser infinite loop;
- `GHSA-5p2g-fcmc-qvqq` — JXL/HEIF parser infinite loops.

Exposure is limited because the production build processes only versioned,
reviewed repository images and MDX; it does not accept user uploads or fetch
untrusted images. This is a containment measure, not a suppression. The audit
script requires the exact two-advisory allowlist: a new high finding fails CI,
and a patched upstream release also fails CI until the obsolete allowance is
removed.
