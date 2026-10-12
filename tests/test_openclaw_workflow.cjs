const {test} = require('node:test');
const assert = require('node:assert/strict');
const w = require('../integrations/openclaw/web/workflow.js');
const config = {schemaVersion: 1, repoRoot: '/srv/research project', python: '/opt/python/bin/python',
  profiles: [{id: 'qc', label: 'QC package', contextPath: '/srv/contexts/qc.json'}]};
const opts = {profileId: 'qc', runId: 'fixture'};

test('planning is the default even when the request mentions training', () => {
  const result = w.build(config, 'Train a model on my package.', opts);
  assert(result.argv.includes('--next-version'));
  assert(!result.argv.includes('--execute'));
  assert(!result.argv.includes('--allow-model-execution'));
  assert(!result.argv.includes('--case-id'));
  assert(!result.argv.includes('--reuse-inference'));
});
test('data execution and neural execution are explicit separate choices', () => {
  const data = w.build(config, 'Inspect the package', {...opts, execute: true});
  assert(data.argv.includes('--execute'));
  assert(!data.argv.includes('--allow-model-execution'));
  assert.throws(() => w.build(config, 'Train', {...opts, allowModelExecution: true}));
  const model = w.build(config, 'Train', {...opts, execute: true, allowModelExecution: true});
  assert(model.argv.includes('--allow-model-execution'));
});
test('long multilingual multiline constraints survive unchanged', () => {
  const input = '只审核，不训练。\n' + 'Return all required artifacts. '.repeat(100);
  const result = w.build(config, input, opts);
  assert.equal(result.argv[result.argv.indexOf('--intent') + 1], input);
  assert(w.prompt(result).includes(JSON.stringify(input)));
});
test('coverage is opt in and unknown contexts are rejected', () => {
  assert(!w.build(config, 'Audit', opts).argv.includes('--coverage-check'));
  assert(w.build(config, 'Audit', {...opts, coverageCheck: true}).argv.includes('--coverage-check'));
  assert.throws(() => w.build(config, 'Audit', {...opts, profileId: 'missing'}));
});
test('no implicit deployment path or truncated overlong request', () => {
  assert.throws(() => w.build({}, 'Audit', opts));
  assert.throws(() => w.build({...config, repoRoot: 'relative'}, 'Audit', opts));
  assert.throws(() => w.build(config, 'a'.repeat(8001), opts));
  assert.throws(() => w.build(config, 'Audit', {...opts, runId: '../escape'}));
});
test('command contains only one quoted user argument despite metacharacters', () => {
  const input = "Audit 'x'; $(printf sentinel) `printf sentinel`\n$HOME";
  const result = w.build(config, input, opts);
  assert.equal(result.argv.filter(x => x === input).length, 1);
  assert(result.command.includes(w.quote(input)));
});

test('panel persists Gateway events and never equates final chat with delivery', () => {
  const vm = require('node:vm'), fs = require('node:fs');
  const storage = new Map();
  class Socket {
    constructor() { this.listeners = {}; this.readyState = 1; }
    addEventListener(name, handler) { this.listeners[name] = handler; }
    send() {}
    emit(name, value) { this.listeners[name]?.(value); }
  }
  const document = {readyState: 'loading', addEventListener() {}, getElementById() { return null; },
    createElement() { return {}; }, head: {appendChild() {}}, querySelector() { return null; }};
  const window = {DENTALCLAW_CONFIG: config, WebSocket: Socket,
    localStorage: {getItem(k) { return storage.get(k) || null; }, setItem(k, v) { storage.set(k, v); }},
    requestAnimationFrame(fn) { fn(); }};
  const context = vm.createContext({window, document, console});
  vm.runInContext(fs.readFileSync(require.resolve('../integrations/openclaw/web/agent-workspace.js'), 'utf8'), context);
  const socket = new window.WebSocket('ws://fixture');
  socket.emit('open', {});
  socket.send(JSON.stringify({type: 'req', id: 'hello', method: 'connect', params: {}}));
  socket.emit('message', {data: JSON.stringify({type: 'res', id: 'hello', ok: true, payload: {server: {version: 'fixture'}}})});
  socket.send(JSON.stringify({type: 'req', id: 'submit', method: 'chat.send',
    params: {message: w.prompt(w.build(config, 'Audit only', opts)), idempotencyKey: 'ui-id', sessionKey: 'agent:dentalclaw:main'}}));
  socket.emit('message', {data: JSON.stringify({type: 'res', id: 'submit', ok: true, payload: {runId: 'actual-id'}})});
  socket.emit('message', {data: JSON.stringify({type: 'event', event: 'chat', payload: {state: 'final', runId: 'actual-id'}})});
  const tasks = JSON.parse(storage.get('dentalclaw.control.agentWorkspace.v2:tasks'));
  assert.equal(tasks[0].runId, 'actual-id');
  assert.equal(tasks[0].status, 'todo');
  const events = JSON.parse(storage.get('dentalclaw.control.agentWorkspace.v2:events'));
  assert(events.some(e => e.title === 'Gateway connected'));
  assert(events.some(e => e.title === 'Reply received — review artifacts'));
});
