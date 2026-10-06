import assert from 'node:assert/strict';
import { test } from 'node:test';
import { build } from 'esbuild';

const bundle = await build({
  entryPoints: ['src/lib/agenticOrchestrator.ts'], bundle: true, platform: 'browser',
  format: 'esm', write: false, tsconfig: 'tsconfig.app.json', define: { 'import.meta.env': '{}' },
});
const code = bundle.outputFiles[0].text;
const { createInitialState, mergeProfiles } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

const update = (field, value, status, quote) => ({ field, value, status, quote, source: 'note' });

test('later explicit correction clears an earlier clinical assertion and preserves IDs', () => {
  const initial = createInitialState().profile;
  const first = mergeProfiles(initial, initial, [
    update('findings.pleura.pneumothorax_present', 'yes', 'present', 'Left pneumothorax.'),
    update('findings.pleura.effusion_present', 'yes', 'present', 'Pleural effusion.'),
    update('assessment.diagnosis_primary', 'pneumothorax', 'present', 'pneumothorax'),
  ]);
  const second = mergeProfiles(first, initial, [
    update('findings.pleura.pneumothorax_present', 'no', 'absent', 'No pneumothorax.'),
    update('assessment.diagnosis_primary', 'pneumothorax', 'resolved', 'Resolved pneumothorax.'),
  ]);
  assert.equal(second.case_id, first.case_id);
  assert.equal(second.image_id, first.image_id);
  assert.equal(second.findings.pleura.pneumothorax_present, 'no');
  assert.equal(second.assessment.diagnosis_primary, null);
  assert.equal(second.findings.pleura.effusion_present, 'yes');
  assert.deepEqual(second.evidence.filter(e => e.field === 'findings.pleura.pneumothorax_present').map(e => e.status), ['absent']);
});
