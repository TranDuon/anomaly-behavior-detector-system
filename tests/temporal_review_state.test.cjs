const test = require('node:test');
const assert = require('node:assert/strict');
const {restoreAnnotations} = require('../frontend/temporal_review_state.js');

function source(video) {
  return {video, sha256:video, split:'train', reviewer:'', review_method:'', review_status:'pending', intervals:[]};
}

function template() {
  return {schema_version:1, source_manifest_sha256:'frozen', sources:[source('one'),source('two')]};
}

function reviewed() {
  return {...source('one'), reviewer:'qa', review_method:'human', review_status:'approved', intervals:[{start_seconds:0,end_seconds:1,action:'hug'}]};
}

test('restoring a partial pack preserves newly added pending sources', () => {
  const draft=template();
  const result=restoreAnnotations(draft,{...draft,sources:[reviewed()]});
  assert.equal(result.sources.length,2);
  assert.equal(result.sources[0].review_status,'approved');
  assert.equal(result.sources[1].review_status,'pending');
  assert.equal(draft.sources[0].review_status,'pending');
});

test('a changed manifest, source digest or split cannot restore approvals', () => {
  const draft=template();
  for(const changed of [{source_manifest_sha256:'other'}, {sources:[{...reviewed(),sha256:'other'}]}, {sources:[{...reviewed(),split:'test'}]}]) {
    const result=restoreAnnotations(draft,{...draft,sources:[reviewed()],...changed});
    assert.equal(result.sources[0].review_status,'pending');
  }
});

test('overlapping, malformed or duplicate saved entries cannot corrupt the pack', () => {
  const draft=template();
  for(const entry of [{...reviewed(),intervals:[{start_seconds:'0',end_seconds:1,action:'hug'}]}, {...reviewed(),intervals:[{start_seconds:0,end_seconds:2,action:'hug'},{start_seconds:1,end_seconds:3,action:'fight'}]}]) {
    assert.deepEqual(restoreAnnotations(draft,{...draft,sources:[entry]}),draft);
  }
  assert.deepEqual(restoreAnnotations(draft,{...draft,sources:[reviewed(),reviewed()]}),draft);
});

test('an approval without reviewer or method becomes pending', () => {
  const draft=template();
  assert.equal(restoreAnnotations(draft,{...draft,sources:[{...reviewed(),reviewer:''}]}).sources[0].review_status,'pending');
});
