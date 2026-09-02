import {spawnSync} from 'node:child_process';

const acceptedHighAdvisories = new Set([
  'GHSA-5p2g-fcmc-qvqq',
  'GHSA-w3rx-r6r6-pgpr',
]);

const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';
const result = spawnSync(npm, ['audit', '--omit=dev', '--json'], {
  cwd: process.cwd(),
  encoding: 'utf8',
});

if (result.error || ![0, 1].includes(result.status)) {
  console.error(result.stderr || result.error || 'npm audit failed unexpectedly');
  process.exit(1);
}

let report;
try {
  report = JSON.parse(result.stdout);
} catch (error) {
  console.error(`npm audit did not return valid JSON: ${error.message}`);
  process.exit(1);
}

if (
  report.error ||
  typeof report.vulnerabilities !== 'object' ||
  typeof report.metadata?.vulnerabilities !== 'object'
) {
  const detail = report.error?.detail || report.error?.summary || report.message;
  console.error(`npm audit did not return a vulnerability report: ${detail || 'unknown error'}`);
  process.exit(1);
}

const activeHighAdvisories = new Set();
for (const vulnerability of Object.values(report.vulnerabilities ?? {})) {
  for (const advisory of vulnerability.via ?? []) {
    if (typeof advisory !== 'object' || advisory === null) continue;
    if (!['high', 'critical'].includes(advisory.severity)) continue;
    const match = advisory.url?.match(/GHSA-[\w-]+$/);
    activeHighAdvisories.add(match?.[0] ?? advisory.url ?? advisory.title);
  }
}

const unreviewed = [...activeHighAdvisories].filter(
  advisory => !acceptedHighAdvisories.has(advisory),
);
const stale = [...acceptedHighAdvisories].filter(
  advisory => !activeHighAdvisories.has(advisory),
);
const critical = report.metadata?.vulnerabilities?.critical ?? 0;

if (critical > 0 || unreviewed.length > 0 || stale.length > 0) {
  if (critical > 0) console.error(`npm audit reports ${critical} critical findings`);
  if (unreviewed.length > 0) {
    console.error(`Unreviewed high advisories: ${unreviewed.join(', ')}`);
  }
  if (stale.length > 0) {
    console.error(`Remove resolved advisories from the allowlist: ${stale.join(', ')}`);
  }
  process.exit(1);
}

console.log(
  `Dependency audit passed with ${activeHighAdvisories.size} explicitly reviewed high advisories.`,
);
