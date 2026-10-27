const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const test = require('node:test');

function harness(saved = null) {
  const elements = new Map();
  function element() {
    return {
      children: [], listeners: {}, checked: false, style: {},
      firstElementChild: {style: {}},
      classList: {toggle() {}},
      append(...items) {this.children.push(...items)},
      replaceChildren(...items) {this.children = items},
      addEventListener(name, callback) {this.listeners[name] = callback},
      setAttribute() {},
    };
  }
  const context = {
    document: {
      documentElement: {dataset: {uiRevision: 'ui-1', serverInstance: 'server-1'}},
      getElementById(id) {if (!elements.has(id)) elements.set(id, element());return elements.get(id)},
      querySelectorAll() {return []}, createElement: element,
    },
    localStorage: {getItem() {return saved}, setItem(key, value) {saved = value}},
    location: {host: '127.0.0.1:8001', reload() {context.reloads++}},
    reloads: 0, setTimeout() {}, clearTimeout() {},
    fetch: async (url, options) => {
      context.lastFetch = {url, options};
      return {ok: true, json: async () => url === '/api/events' ? {events: []} : context.status};
    },
  };
  context.status = {
    server: {ui_revision: 'ui-1', instance_id: 'server-1', process_id: 42, backend_revision: 'backend-1'},
    model: {architecture: 'rgb', message: 'pilot', pilot_default_allowed: true, pilot_required: false},
    model_loaded: true, recording: null, phase: 'idle', source: null, timestamp: 0,
    latency_seconds: 0, event_count: 0, total_frames: 0, running: false,
    prediction: {state: 'warming_up', score: null},
  };
  const source = fs.readFileSync(path.join(__dirname, '../frontend/fight.html'), 'utf8');
  const script = source.match(/<script>([\s\S]*?)<\/script>/)[1].replace(/poll\(\);\s*$/, '');
  vm.runInNewContext(script + ';this.helpers={syncServer,syncPilot,renderEvents,request,poll};', context);
  return {context, elements, helpers: context.helpers};
}

test('UI and backend instance changes reload stale pages', () => {
  const {context, helpers} = harness();
  assert.equal(helpers.syncServer(context.status.server), false);
  assert.equal(context.reloads, 0);
  assert.equal(helpers.syncServer({...context.status.server, ui_revision: 'ui-2'}), true);
  assert.equal(context.reloads, 1);
  assert.equal(helpers.syncServer({...context.status.server, instance_id: 'server-2'}), true);
  assert.equal(context.reloads, 2);
});

test('pilot follows explicit server default but preserves local opt-out', () => {
  const fresh = harness();
  fresh.helpers.syncPilot({pilot_default_allowed: true});
  assert.equal(fresh.elements.get('pilot-mode').checked, true);
  const optedOut = harness('false');
  optedOut.helpers.syncPilot({pilot_default_allowed: true});
  assert.equal(optedOut.elements.get('pilot-mode').checked, false);
});

test('an empty API event list clears old DOM events', () => {
  const {helpers, elements} = harness();
  helpers.renderEvents([{snapshot: '/image.jpg', started_at: 1, ended_at: 2, score: 0.9}]);
  assert.equal(elements.get('events').children[0].className, 'event');
  helpers.renderEvents([]);
  assert.equal(elements.get('events').children.length, 1);
  assert.equal(elements.get('events').children[0].className, 'no-events');
});

test('API fetches do not reuse cached state', async () => {
  const {context, helpers} = harness();
  await helpers.request('/api/status');
  assert.equal(context.lastFetch.options.cache, 'no-store');
});

test('preview and pilot gate are not mislabeled as missing checkpoint', async () => {
  const preview = harness();
  preview.context.status.preview_mode = true;
  await preview.helpers.poll();
  assert.match(preview.elements.get('state').textContent, /Chỉ xem thử/);
  const blocked = harness();
  blocked.context.status.model_loaded = false;
  blocked.context.status.model.pilot_required = true;
  blocked.context.status.prediction.state = 'unavailable';
  await blocked.helpers.poll();
  assert.equal(blocked.elements.get('state').textContent, 'Cần cho phép pilot');
});
