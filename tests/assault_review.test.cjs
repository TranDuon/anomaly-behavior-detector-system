const test = require('node:test');
const assert = require('node:assert/strict');
const {validateIntervals, restore} = require('../frontend/assault_review.js');
const row = {video:'one', source_sha256:'hash', split:'train', duration:5};
const entry = {...row, review_status:'pending', reviewer:'', intervals:[]};
const template = {schema_version:1, profile:'assault_extension_v1', manifest_sha256:'manifest', sources:[entry]};

test('general fall is positive without assault evidence', () => {
  const interval = {start_seconds:1, end_seconds:2, action:'fall'};
  assert.doesNotThrow(() => validateIntervals([interval],5));
  assert.throws(() => validateIntervals([{...interval,action:'accidental_fall'}],5));
});
test('invalid or overlapping temporal segments are rejected', () => {
  for(const intervals of [[{start_seconds:-1,end_seconds:2,action:'fight'}], [{start_seconds:0,end_seconds:6,action:'fight'}], [{start_seconds:0,end_seconds:2,action:'wrestle'},{start_seconds:1,end_seconds:3,action:'fight'}]]) assert.throws(() => validateIntervals(intervals,5));
});
test('saved labels cannot change source identity, split or manifest', () => {
  assert.throws(() => restore(template,{...template,manifest_sha256:'other'},[row]));
  assert.throws(() => restore(template,{...template,sources:[{...entry,split:'test'}]},[row]));
  assert.throws(() => restore(template,{...template,sources:[entry,entry]},[row]));
  assert.throws(() => restore(template,{...template,sources:[{...entry,review_status:'reviewed'}]},[row]));
});
test('review restoration preserves precise times without mutating the pending template', () => {
  const reviewed = {...entry,review_status:'reviewed',reviewer:'human',intervals:[{start_seconds:1.2,end_seconds:2.4,action:'wrestle'}]};
  const result=restore(template,{...template,sources:[reviewed]},[row]);
  assert.equal(result.sources[0].intervals[0].start_seconds,1.2);
  assert.equal(template.sources[0].review_status,'pending');
});
