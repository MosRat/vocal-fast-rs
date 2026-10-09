import test from 'node:test';
import assert from 'node:assert/strict';
import { observeImport, importActive, suggestIndex } from '../src/model-library-types.ts';
test('import progress ignores old jobs and reordered revisions',()=>{
  const state={jobId:4,revision:7,phase:'downloading'};
  assert.equal(observeImport(state,{jobId:3,revision:90}),state);
  assert.equal(observeImport(state,{jobId:4,revision:6}),state);
  assert.equal(observeImport(state,{jobId:4,revision:7}),state);
  assert.equal(observeImport(state,{jobId:5,revision:1}).jobId,5);
  assert.equal(importActive('cancelling'),true);assert.equal(importActive('failed'),false);
});
test('index recommendation avoids selecting an unrelated index',()=>{
  const indices=['voices/a.index','voices/b.index'].map(path=>({path,kind:'index'}));
  assert.equal(suggestIndex('weights/b.pth',indices),'voices/b.index');
  assert.equal(suggestIndex('weights/c.pth',indices),'');
});
