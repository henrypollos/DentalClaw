(function () {
  "use strict";

  if (window.__openclawAgentWorkspaceInstalled) {
    return;
  }
  window.__openclawAgentWorkspaceInstalled = true;

  const STORAGE_PREFIX = "dentalclaw.control.agentWorkspace.v2";
  const MAIN_SETTINGS_KEY = "openclaw.control.settings.v1";
  const TASKS_KEY = `${STORAGE_PREFIX}:tasks`;
  const EVENTS_KEY = `${STORAGE_PREFIX}:events`;
  const UI_KEY = `${STORAGE_PREFIX}:ui`;
  const DEFAULT_SESSION_KEY = "agent:main";
  const CONFIG = window.DENTALCLAW_CONFIG || {};
  const MAX_EVENTS = 80;
  const MAX_DETAIL = 180;

  const statusOrder = ["todo", "doing", "done", "blocked"];
  const statusLabels = {
    todo: "Todo",
    doing: "In progress",
    done: "Done",
    blocked: "Blocked",
  };

  const eventToneLabels = {
    task: "Task",
    live: "Run",
    tool: "Tool",
    done: "Done",
    warn: "Warning",
    error: "Error",
    system: "System",
  };

  const icons = {
    plus:
      '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v14M5 12h14"/></svg>',
    close:
      '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18 6 6 18M6 6l12 12"/></svg>',
    trash:
      '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6h18M8 6V4h8v2m-1 5v6M9 11v6m-2-11 1 14h8l1-14"/></svg>',
    check:
      '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m20 6-11 11-5-5"/></svg>',
    clock:
      '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8"/><path d="M12 8v5l3 2"/></svg>',
    circle:
      '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="7"/></svg>',
    alert:
      '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 8v5M12 17h.01"/><path d="m10 4-8 14a2 2 0 0 0 2 3h16a2 2 0 0 0 2-3L14 4a2 2 0 0 0-4 0Z"/></svg>',
    route:
      '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="6" cy="6" r="2"/><circle cx="18" cy="18" r="2"/><path d="M8 6h4a4 4 0 0 1 0 8H9a3 3 0 0 0 0 6h7"/></svg>',
    panel:
      '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M14 4v16M7 8h3M7 12h3M7 16h3"/></svg>',
  };


  const state = {
    socketUrl: null,
    connected: false,
    panel: null,
    topbarButton: null,
    socket: null,
    lastSessionKey: null,
    mutationScheduled: false,
    pendingRequests: new Map(),
    seenStreamingRuns: new Set(),
    seenToolUpdates: new Set(),
    tasks: loadJson(TASKS_KEY, []),
    events: loadJson(EVENTS_KEY, []),
    ui: loadJson(UI_KEY, { open: true, filter: "all" }),
    form: {requirement: "", profileId: CONFIG.profiles?.[0]?.id || "", execute: false, allowModelExecution: false, coverageCheck: false},
  };

  injectStyles();
  patchWebSocket();

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", startUi, { once: true });
  } else {
    startUi();
  }

  function loadJson(key, fallback) {
    try {
      const raw = window.localStorage.getItem(key);
      if (!raw) {
        return fallback;
      }
      const parsed = JSON.parse(raw);
      return parsed == null ? fallback : parsed;
    } catch {
      return fallback;
    }
  }

  function saveJson(key, value) {
    try {
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch {
      // Local storage can be unavailable in private browsing modes.
    }
  }

  function uid(prefix) {
    return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (ch) => {
      switch (ch) {
        case "&":
          return "&amp;";
        case "<":
          return "&lt;";
        case ">":
          return "&gt;";
        case '"':
          return "&quot;";
        default:
          return "&#39;";
      }
    });
  }

  function compactText(value, max = MAX_DETAIL) {
    const text = String(value ?? "")
      .replace(/\s+/g, " ")
      .trim();
    if (!text) {
      return "";
    }
    return text.length > max ? `${text.slice(0, max - 1)}…` : text;
  }

  function formatValue(value) {
    if (value == null) {
      return "";
    }
    if (typeof value === "string") {
      return compactText(value);
    }
    try {
      return compactText(JSON.stringify(value));
    } catch {
      return compactText(String(value));
    }
  }

  function getParams(frame) {
    return frame && typeof frame.params === "object" && frame.params
      ? frame.params
      : {};
  }

  function parseFrame(data) {
    if (typeof data !== "string") {
      return null;
    }
    try {
      const parsed = JSON.parse(data);
      return parsed && typeof parsed === "object" ? parsed : null;
    } catch {
      return null;
    }
  }

  function normalizeStatus(value) {
    return statusOrder.includes(value) ? value : "todo";
  }

  function persistTasks() { saveJson(TASKS_KEY, state.tasks); }
  function persistEvents() { saveJson(EVENTS_KEY, state.events); }
  function persistUi() { saveJson(UI_KEY, state.ui); }

  function renderSoon() {
    window.requestAnimationFrame(() => {
      syncShellClass();
      renderPanel();
      syncTopbarButton();
    });
  }

  function addEvent(tone, title, detail, extra) {
    const event = {
      id: uid("evt"),
      tone,
      title: compactText(title, 72),
      detail: compactText(detail, MAX_DETAIL),
      runId: extra && extra.runId ? String(extra.runId) : null,
      sessionKey: extra && extra.sessionKey ? String(extra.sessionKey) : null,
      at: Date.now(),
    };
    state.events = [event, ...state.events].slice(0, MAX_EVENTS);
    persistEvents();
    renderSoon();
  }

  function addTask(title, opts) {
    const cleanTitle = compactText(title, 160);
    if (!cleanTitle) {
      return null;
    }

    const runId = opts && opts.runId ? String(opts.runId) : null;
    if (runId) {
      const existing = state.tasks.find((task) => task.runId === runId);
      if (existing) {
        existing.title = cleanTitle;
        existing.status = normalizeStatus(opts.status);
        existing.updatedAt = Date.now();
        existing.sessionKey = opts.sessionKey || existing.sessionKey || null;
        persistTasks();
        renderSoon();
        return existing;
      }
    }

    const task = {
      id: uid("task"),
      title: cleanTitle,
      status: normalizeStatus(opts && opts.status),
      source: opts && opts.source ? String(opts.source) : "manual",
      runId,
      sessionKey: opts && opts.sessionKey ? String(opts.sessionKey) : null,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    state.tasks = [task, ...state.tasks].slice(0, 120);
    persistTasks();
    renderSoon();
    return task;
  }

  function updateRunTask(runId, status) {
    if (!runId) {
      return;
    }
    const task = state.tasks.find((item) => item.runId === runId);
    if (!task) {
      return;
    }
    task.status = normalizeStatus(status);
    task.updatedAt = Date.now();
    persistTasks();
    renderSoon();
  }

  function cycleTaskStatus(taskId) {
    const task = state.tasks.find((item) => item.id === taskId);
    if (!task) {
      return;
    }
    const current = statusOrder.indexOf(normalizeStatus(task.status));
    task.status = statusOrder[(current + 1) % statusOrder.length];
    task.updatedAt = Date.now();
    persistTasks();
    renderSoon();
  }

  function deleteTask(taskId) {
    const next = state.tasks.filter((item) => item.id !== taskId);
    if (next.length === state.tasks.length) {
      return;
    }
    state.tasks = next;
    persistTasks();
    renderSoon();
  }

  function clearDoneTasks() {
    state.tasks = state.tasks.filter((item) => item.status !== "done");
    persistTasks();
    renderSoon();
  }

  function clearEvents() {
    state.events = [];
    persistEvents();
    renderSoon();
  }

  function setFilter(filter) {
    state.ui.filter = filter === "all" || statusOrder.includes(filter) ? filter : "all";
    persistUi();
    renderSoon();
  }

  function setOpen(open) {
    state.ui.open = Boolean(open);
    persistUi();
    renderSoon();
  }

  function getActiveSessionKey() {
    const app = document.querySelector("openclaw-app");
    if (app && typeof app.sessionKey === "string" && app.sessionKey) return app.sessionKey;
    const settings = loadJson(MAIN_SETTINGS_KEY, {});
    return (
      (settings && typeof settings.lastActiveSessionKey === "string" && settings.lastActiveSessionKey) ||
      (settings && typeof settings.sessionKey === "string" && settings.sessionKey) ||
      state.lastSessionKey || DEFAULT_SESSION_KEY
    );
  }

  function isDentalClawMessage(message) {
    return typeof message === "string" && message.startsWith("[DentalClaw Workflow]");
  }

  function extractDentalClawRequirement(message) {
    if (!isDentalClawMessage(message)) {
      return "";
    }
    const match = /^User requirement JSON:\s*(.+)$/m.exec(message);
    try { return match ? compactText(JSON.parse(match[1]), 128) : ""; } catch { return ""; }
  }

  function deriveTaskTitle(params) {
    const message = typeof params.message === "string" ? params.message.trim() : "";
    const dentalClawRequirement = extractDentalClawRequirement(message);
    if (dentalClawRequirement) {
      return `DentalClaw: ${dentalClawRequirement}`;
    }
    if (message) {
      return message;
    }
    const attachments = Array.isArray(params.attachments) ? params.attachments.length : 0;
    if (attachments > 0) {
      return `Process ${attachments} attachment${attachments === 1 ? "" : "s"}`;
    }
    return "Agent task";
  }

  function sendGatewayRequest(method, params) {
    const socket = state.socket;
    if (!state.connected || !socket || socket.readyState !== 1 || typeof socket.send !== "function") {
      return null;
    }
    const id = uid("req");
    socket.send(JSON.stringify({ type: "req", id, method, params }));
    return id;
  }

  function runDentalClawWorkflow(requirement) {
    let workflow;
    try {
      workflow = window.DentalClawWorkflow.build(CONFIG, requirement, state.form);
    } catch (error) {
      addEvent("error", "Request not sent", error.message, {});
      return false;
    }
    const sessionKey = getActiveSessionKey();
    if (!CONFIG.agentId || !sessionKey.startsWith(`agent:${CONFIG.agentId}:`)) {
      addEvent("warn", "Select the DentalClaw chat", `Select agent ${CONFIG.agentId || "dentalclaw"} before submitting.`, {});
      return false;
    }
    const runId = uid("dentalclaw");
    const requestId = sendGatewayRequest("chat.send", {
      sessionKey, message: window.DentalClawWorkflow.prompt(workflow), idempotencyKey: runId,
    });
    if (!requestId) {
      addEvent("error", "Gateway unavailable", "Connect and authenticate the dashboard before submitting.", {runId, sessionKey});
      return false;
    }
    addEvent("task", "Request delegated", `${workflow.label} · ${workflow.contextLabel}`, {runId, sessionKey});
    addEvent("system", "Awaiting backend decision", workflow.runDir, {runId, sessionKey});
    return true;
  }

  function handleOutgoingFrame(frame) {
    if (!frame || frame.type !== "req" || typeof frame.method !== "string") {
      return;
    }

    state.pendingRequests.set(String(frame.id), {
      method: frame.method,
      params: getParams(frame),
      at: Date.now(),
    });
    if (state.pendingRequests.size > 200) {
      const firstKey = state.pendingRequests.keys().next().value;
      if (firstKey) {
        state.pendingRequests.delete(firstKey);
      }
    }

    const params = getParams(frame);
    if (frame.method === "connect") {
      addEvent("system", "Connecting to gateway", state.socketUrl || "WebSocket", {});
      return;
    }
    if (frame.method === "chat.send") {
      const runId = params.idempotencyKey || frame.id;
      const sessionKey = params.sessionKey;
      if (sessionKey) {
        state.lastSessionKey = String(sessionKey);
      }
      const title = deriveTaskTitle(params);
      const source = isDentalClawMessage(params.message) ? "dentalclaw" : "agent";
      addTask(title, { status: "doing", source, runId, sessionKey });
      addEvent("task", "Task sent", title, { runId, sessionKey });
      return;
    }
    if (frame.method === "chat.abort") {
      addEvent("warn", "Abort requested", params.runId ? `run ${params.runId}` : "current session", {
        runId: params.runId,
        sessionKey: params.sessionKey,
      });
      return;
    }
    if (frame.method === "config.apply") {
      addEvent("system", "Applying config", "config.apply", {});
      return;
    }
    if (frame.method === "cron.run") {
      addEvent("task", "Manual cron run", formatValue(params), {});
    }
  }

  function handleResponseFrame(frame) {
    if (!frame || frame.type !== "res") {
      return;
    }
    const pending = state.pendingRequests.get(String(frame.id));
    if (!pending) {
      return;
    }
    state.pendingRequests.delete(String(frame.id));

    if (pending.method === "connect") {
      if (frame.ok) {
        state.connected = true;
        const payload = frame.payload && typeof frame.payload === "object" ? frame.payload : {};
        const server = payload.server && typeof payload.server === "object" ? payload.server : {};
        const version = typeof server.version === "string" ? `v${server.version}` : "ready";
        addEvent("done", "Gateway connected", version, {});
      } else {
        state.connected = false;
        addEvent("error", "Connection failed", frame.error && frame.error.message, {});
      }
      renderSoon();
      return;
    }

    if (!frame.ok) {
      if (pending.method === "chat.send") updateRunTask(pending.params.idempotencyKey, "blocked");
      addEvent("error", "Request failed", `${pending.method}: ${frame.error && frame.error.message}`, {});
    } else if (pending.method === "chat.send" && frame.payload?.runId) {
      const task = state.tasks.find(item => item.runId === pending.params.idempotencyKey);
      if (task) { task.runId = frame.payload.runId; persistTasks(); }
    }
  }

  function handleEventFrame(frame) {
    if (!frame || frame.type !== "event" || typeof frame.event !== "string") {
      return;
    }
    if (frame.event === "connect.challenge") {
      return;
    }

    const payload = frame.payload && typeof frame.payload === "object" ? frame.payload : {};
    if (frame.event === "chat") {
      handleChatEvent(payload);
      return;
    }
    if (frame.event === "agent") {
      handleAgentEvent(payload);
      return;
    }
    if (frame.event.startsWith("exec.approval.")) {
      const requested = frame.event.endsWith("requested");
      addEvent(
        requested ? "warn" : "done",
        requested ? "Execution approval pending" : "Execution approval resolved",
        formatValue(payload),
        {},
      );
      return;
    }
    if (frame.event.startsWith("cron.")) {
      addEvent("system", "Cron event", frame.event, {});
      return;
    }
    if (frame.event.startsWith("device.pair.")) {
      addEvent("system", "Device pairing event", frame.event, {});
      return;
    }
  }

  function handleChatEvent(payload) {
    const runId = typeof payload.runId === "string" ? payload.runId : null;
    const sessionKey = typeof payload.sessionKey === "string" ? payload.sessionKey : null;
    const stateName = typeof payload.state === "string" ? payload.state : "";

    if (stateName === "delta") {
      if (runId && !state.seenStreamingRuns.has(runId)) {
        state.seenStreamingRuns.add(runId);
        addEvent("live", "Generating reply", "Streaming output", { runId, sessionKey });
      }
      return;
    }

    if (stateName === "final") {
      const task = state.tasks.find(item => item.runId === runId);
      const needsReview = task?.source === "dentalclaw";
      updateRunTask(runId, needsReview ? "todo" : "done");
      addEvent(needsReview ? "warn" : "done", needsReview ? "Reply received — review artifacts" : "Reply completed",
        needsReview ? "Inspect prediction and execution records; chat completion does not verify delivery." : "The agent returned a final reply", { runId, sessionKey });
      if (runId) {
        state.seenStreamingRuns.delete(runId);
      }
      return;
    }

    if (stateName === "aborted") {
      updateRunTask(runId, "blocked");
      addEvent("warn", "Task aborted", "The run was aborted", { runId, sessionKey });
      if (runId) {
        state.seenStreamingRuns.delete(runId);
      }
      return;
    }

    if (stateName === "error") {
      updateRunTask(runId, "blocked");
      addEvent("error", "Task failed", payload.errorMessage || "Agent run failed", { runId, sessionKey });
      if (runId) {
        state.seenStreamingRuns.delete(runId);
      }
    }
  }

  function handleAgentEvent(payload) {
    const runId = typeof payload.runId === "string" ? payload.runId : null;
    const sessionKey = typeof payload.sessionKey === "string" ? payload.sessionKey : null;
    const stream = typeof payload.stream === "string" ? payload.stream : "";
    const data = payload.data && typeof payload.data === "object" ? payload.data : {};

    if (stream === "tool") {
      const phase = typeof data.phase === "string" ? data.phase : "";
      const name = typeof data.name === "string" && data.name ? data.name : "tool";
      const toolCallId = typeof data.toolCallId === "string" ? data.toolCallId : name;

      if (phase === "start") {
        const detail = data.args == null ? "" : formatValue(data.args);
        addEvent("tool", `Tool started: ${name}`, detail, { runId, sessionKey });
        return;
      }

      if (phase === "update") {
        const key = `${toolCallId}:update`;
        if (!state.seenToolUpdates.has(key)) {
          state.seenToolUpdates.add(key);
          addEvent("tool", `Tool update: ${name}`, "Intermediate result received", { runId, sessionKey });
        }
        return;
      }

      if (phase === "result") {
        state.seenToolUpdates.delete(`${toolCallId}:update`);
        const result = data.result == null ? "Completed" : formatValue(data.result);
        addEvent("done", `Tool completed: ${name}`, result, { runId, sessionKey });
        return;
      }

      addEvent("tool", `Tool event: ${name}`, phase || "tool", { runId, sessionKey });
      return;
    }

    if (stream === "compaction") {
      const phase = typeof data.phase === "string" ? data.phase : "";
      addEvent(
        phase === "end" ? "done" : "system",
        phase === "end" ? "Context compaction completed" : "Context compaction started",
        "compaction",
        { runId, sessionKey },
      );
      return;
    }

    if (stream === "fallback" || stream === "lifecycle") {
      const phase = typeof data.phase === "string" ? data.phase : stream;
      const active = data.activeModel || data.toModel || data.selectedModel || "";
      addEvent("warn", "Model routing changed", `${phase} ${formatValue(active)}`, { runId, sessionKey });
      return;
    }

    if (stream) {
      addEvent("system", `Agent event: ${stream}`, formatValue(data), { runId, sessionKey });
    }
  }

  function patchWebSocket() {
    const NativeWebSocket = window.WebSocket;
    if (!NativeWebSocket || NativeWebSocket.__openclawWorkspaceProxy) {
      return;
    }

    const proxy = new Proxy(NativeWebSocket, {
      construct(target, args, newTarget) {
        const socket = Reflect.construct(target, args, newTarget);
        instrumentSocket(socket, args);
        return socket;
      },
      get(target, prop, receiver) {
        return Reflect.get(target, prop, receiver);
      },
    });
    proxy.__openclawWorkspaceProxy = true;
    window.WebSocket = proxy;
  }

  function instrumentSocket(socket, args) {
    state.socketUrl = args && args[0] ? String(args[0]) : null;
    state.socket = socket;

    socket.addEventListener("open", () => {
      state.connected = false;
      state.socket = socket;
      addEvent("system", "WebSocket opened", state.socketUrl || "", {});
      renderSoon();
    });

    socket.addEventListener("message", (event) => {
      const frame = parseFrame(String(event.data || ""));
      if (!frame) {
        return;
      }
      handleResponseFrame(frame);
      handleEventFrame(frame);
    });

    socket.addEventListener("close", (event) => {
      state.connected = false;
      if (state.socket === socket) {
        state.socket = null;
      }
      const detail = event && event.code ? `code ${event.code}` : "";
      if (!event || event.code !== 1000) {
        addEvent("warn", "WebSocket closed", detail, {});
      }
      renderSoon();
    });

    const nativeSend = socket.send.bind(socket);
    socket.send = function patchedSend(data) {
      const frame = parseFrame(typeof data === "string" ? data : "");
      if (frame) {
        handleOutgoingFrame(frame);
      }
      return nativeSend(data);
    };
  }

  function startUi() {
    document.body.classList.add("oc-agent-enhanced");
    ensureMounted();
    const observer = new MutationObserver((mutations) => {
      if (mutations.every(isWorkspaceMutation)) {
        return;
      }
      scheduleMountCheck();
    });
    observer.observe(document.body, { childList: true, subtree: true });
    window.addEventListener("resize", syncShellClass);
  }

  function isWorkspaceMutation(mutation) {
    if (!state.panel || !mutation || !mutation.target) {
      return false;
    }
    return mutation.target === state.panel || state.panel.contains(mutation.target);
  }

  function scheduleMountCheck() {
    if (state.mutationScheduled) {
      return;
    }
    state.mutationScheduled = true;
    window.requestAnimationFrame(() => {
      state.mutationScheduled = false;
      ensureMounted();
    });
  }

  function ensureMounted() {
    const shell = document.querySelector(".shell");
    if (!shell) {
      state.panel = null;
      return;
    }

    let mounted = false;
    if (!state.panel || state.panel.parentElement !== shell) {
      if (state.panel && state.panel.parentElement) {
        state.panel.parentElement.removeChild(state.panel);
      }
      state.panel = document.createElement("aside");
      state.panel.className = "oc-agent-workspace";
      state.panel.setAttribute("aria-label", "Agent workspace");
      shell.appendChild(state.panel);
      state.panel.addEventListener("click", handlePanelClick);
      state.panel.addEventListener("submit", handlePanelSubmit);
      state.panel.addEventListener("input", saveWorkflowDraft);
      state.panel.addEventListener("change", saveWorkflowDraft);
      mounted = true;
    }

    syncShellClass();
    mountTopbarButton();
    if (mounted || !state.panel.innerHTML) {
      renderPanel();
    }
  }

  function syncShellClass() {
    const shell = document.querySelector(".shell");
    if (!shell) {
      return;
    }
    const open = Boolean(state.ui.open);
    shell.classList.toggle("shell--agent-workspace-open", open);
    document.body.classList.toggle("oc-agent-workspace-open", open);
  }

  function mountTopbarButton() {
    const actions = document.querySelector(".topnav-shell__actions");
    if (!actions) {
      return;
    }
    if (!state.topbarButton || state.topbarButton.parentElement !== actions) {
      if (state.topbarButton && state.topbarButton.parentElement) {
        state.topbarButton.parentElement.removeChild(state.topbarButton);
      }
      state.topbarButton = document.createElement("button");
      state.topbarButton.type = "button";
      state.topbarButton.className = "oc-workspace-toggle";
      state.topbarButton.addEventListener("click", () => setOpen(!state.ui.open));
      actions.prepend(state.topbarButton);
    }
    syncTopbarButton();
  }

  function syncTopbarButton() {
    if (!state.topbarButton) {
      return;
    }
    state.topbarButton.setAttribute("aria-pressed", String(Boolean(state.ui.open)));
    state.topbarButton.title = state.ui.open ? "Hide agent workspace" : "Show agent workspace";
    state.topbarButton.innerHTML = `${icons.panel}<span>Workspace</span>`;
  }

  function handlePanelClick(event) {
    const button = event.target.closest("button");
    if (!button) {
      return;
    }
    const action = button.dataset.ocAction;
    if (!action) {
      return;
    }
    if (action === "toggle") {
      setOpen(!state.ui.open);
      return;
    }
    if (action === "filter") {
      setFilter(button.dataset.ocFilter || "all");
      return;
    }
    if (action === "cycle") {
      cycleTaskStatus(button.dataset.ocTaskId);
      return;
    }
    if (action === "delete") {
      deleteTask(button.dataset.ocTaskId);
      return;
    }
    if (action === "clear-done") {
      clearDoneTasks();
      return;
    }
    if (action === "clear-events") {
      clearEvents();
    }
  }

  function saveWorkflowDraft(event) {
    const form = event.target.closest("form[data-oc-dentalclaw-form]");
    if (!form) return;
    state.form.requirement = form.elements.requirement.value;
    state.form.profileId = form.elements.profileId.value;
    state.form.execute = form.elements.execute.checked;
    state.form.allowModelExecution = state.form.execute && form.elements.allowModelExecution.checked;
    state.form.coverageCheck = form.elements.coverageCheck.checked;
    form.elements.allowModelExecution.disabled = !state.form.execute;
    form.elements.allowModelExecution.checked = state.form.allowModelExecution;
  }

  function handlePanelSubmit(event) {
    const workflowForm = event.target.closest("form[data-oc-dentalclaw-form]");
    if (workflowForm) {
      event.preventDefault();
      const input = workflowForm.querySelector("textarea[name='requirement']");
      const requirement = input ? input.value : "";
      if (!requirement.trim()) {
        return;
      }
      saveWorkflowDraft({target: input});
      if (runDentalClawWorkflow(requirement)) {
        state.form.requirement = "";
        input.value = "";
      }
      input.focus();
      return;
    }

    const form = event.target.closest("form[data-oc-task-form]");
    if (!form) {
      return;
    }
    event.preventDefault();
    const input = form.querySelector("input[name='task']");
    const title = input ? input.value.trim() : "";
    if (!title) {
      return;
    }
    addTask(title, { status: "todo", source: "manual" });
    addEvent("task", "Task added", title, {});
    input.value = "";
    input.focus();
  }

  function renderPanel() {
    if (!state.panel) {
      return;
    }
    const active = document.activeElement;
    const draftField = state.panel.contains(active) && active.closest("form[data-oc-dentalclaw-form]") ? active.name : null;
    const selection = draftField === "requirement" ? [active.selectionStart, active.selectionEnd] : null;
    state.panel.innerHTML = renderPanelHtml();
    if (draftField) {
      const field = state.panel.querySelector(`[name="${draftField}"]`);
      if (field) { field.focus(); if (selection) field.setSelectionRange(...selection); }
    }
  }

  function renderPanelHtml() {
    const filter = state.ui.filter || "all";
    const tasks = state.tasks.filter((task) => filter === "all" || task.status === filter);
    const activeCount = state.tasks.filter((task) => task.status === "doing").length;
    const doneCount = state.tasks.filter((task) => task.status === "done").length;
    const blockedCount = state.tasks.filter((task) => task.status === "blocked").length;
    const totalCount = state.tasks.length;

    return `
      <div class="oc-agent-workspace__inner">
        <header class="oc-agent-workspace__header">
          <div>
            <div class="oc-agent-workspace__eyebrow">Agent Workspace</div>
            <h2>Work Trail & Tasks</h2>
          </div>
          <button class="oc-icon-btn" type="button" title="Hide workspace" aria-label="Hide workspace" data-oc-action="toggle">
            ${icons.close}
          </button>
        </header>

        <section class="oc-agent-stats" aria-label="Task overview">
          ${renderStat("In progress", activeCount, "doing")}
          ${renderStat("Done", doneCount, "done")}
          ${renderStat("Blocked", blockedCount, "blocked")}
          ${renderStat("All", totalCount, "all")}
        </section>

        <section class="oc-agent-card oc-agent-card--workflow">
          <div class="oc-agent-card__header">
            <div>
              <h3>DentalClaw Workflow</h3>
              <p>Four registered research workflows</p>
            </div>
            <span class="oc-workflow-pill">Plan first</span>
          </div>

          <form class="oc-workflow-form" data-oc-dentalclaw-form>
            <label>Input context
              <select name="profileId" aria-label="Input context">
                ${(CONFIG.profiles || []).map(p => `<option value="${escapeHtml(p.id)}" ${p.id === state.form.profileId ? "selected" : ""}>${escapeHtml(p.label || p.id)}</option>`).join("")}
              </select>
            </label>
            <textarea name="requirement" aria-label="Research request" autocomplete="off" maxlength="8000"
              placeholder="Describe the requested outputs and constraints">${escapeHtml(state.form.requirement)}</textarea>
            <label><input type="checkbox" name="execute" ${state.form.execute ? "checked" : ""}> Execute if the route is ready</label>
            <label><input type="checkbox" name="allowModelExecution" ${state.form.allowModelExecution ? "checked" : ""} ${!state.form.execute ? "disabled" : ""}> Allow training or inference</label>
            <label><input type="checkbox" name="coverageCheck" ${state.form.coverageCheck ? "checked" : ""}> Check complete deliverables (development)</label>
            <button class="oc-mini-btn oc-mini-btn--accent" type="submit">Submit request</button>
            <p>Use the ${escapeHtml(CONFIG.agentId || "dentalclaw")} chat. Replies and task marks are not delivery verification.</p>
          </form>
        </section>

        <section class="oc-agent-card oc-agent-card--tasks">
          <div class="oc-agent-card__header">
            <div>
              <h3>Task Management</h3>
              <p>${state.connected ? "Gateway online" : "Gateway offline"}</p>
            </div>
            <button class="oc-mini-btn" type="button" data-oc-action="clear-done">Clear done</button>
          </div>

          <form class="oc-task-form" data-oc-task-form>
            <input name="task" type="text" autocomplete="off" maxlength="160" placeholder="Add a task" />
            <button class="oc-icon-btn oc-icon-btn--accent" type="submit" title="Add task" aria-label="Add task">
              ${icons.plus}
            </button>
          </form>

          <div class="oc-filter-row" role="tablist" aria-label="Task filter">
            ${renderFilter("all", "All", filter)}
            ${renderFilter("todo", "Todo", filter)}
            ${renderFilter("doing", "In progress", filter)}
            ${renderFilter("blocked", "Blocked", filter)}
            ${renderFilter("done", "Done", filter)}
          </div>

          <div class="oc-task-list">
            ${tasks.length ? tasks.map(renderTask).join("") : renderEmptyTask(filter)}
          </div>
        </section>

        <section class="oc-agent-card oc-agent-card--timeline">
          <div class="oc-agent-card__header">
            <div>
              <h3>Work Trail</h3>
              <p>${state.events.length} recent event${state.events.length === 1 ? "" : "s"}</p>
            </div>
            <button class="oc-icon-btn" type="button" title="Clear trail" aria-label="Clear trail" data-oc-action="clear-events">
              ${icons.trash}
            </button>
          </div>
          <div class="oc-timeline" role="log" aria-live="polite">
            ${state.events.length ? state.events.map(renderEvent).join("") : renderEmptyEvents()}
          </div>
        </section>
      </div>
    `;
  }

  function renderStat(label, value, tone) {
    return `
      <button class="oc-stat oc-stat--${escapeHtml(tone)}" type="button" data-oc-action="filter" data-oc-filter="${escapeHtml(tone)}">
        <span>${escapeHtml(label)}</span>
        <strong>${escapeHtml(value)}</strong>
      </button>
    `;
  }

  function renderFilter(value, label, filter) {
    const active = value === filter;
    return `
      <button
        class="oc-filter ${active ? "oc-filter--active" : ""}"
        type="button"
        role="tab"
        aria-selected="${active}"
        data-oc-action="filter"
        data-oc-filter="${escapeHtml(value)}"
      >${escapeHtml(label)}</button>
    `;
  }

  function renderTask(task) {
    const status = normalizeStatus(task.status);
    const title = escapeHtml(task.title);
    const sourceLabel =
      task.source === "agent" ? "agent" : task.source === "dentalclaw" ? "DentalClaw" : "manual";
    const meta = [
      sourceLabel,
      task.sessionKey ? compactText(task.sessionKey, 28) : "",
      timeAgo(task.updatedAt || task.createdAt),
    ]
      .filter(Boolean)
      .join(" · ");

    return `
      <article class="oc-task oc-task--${status}">
        <button
          class="oc-task__status"
          type="button"
          title="Change status"
          aria-label="Change task status"
          data-oc-action="cycle"
          data-oc-task-id="${escapeHtml(task.id)}"
        >
          ${statusIcon(status)}
        </button>
        <div class="oc-task__body">
          <div class="oc-task__title">${title}</div>
          <div class="oc-task__meta">
            <span class="oc-task__badge oc-task__badge--${status}">${escapeHtml(statusLabels[status])}</span>
            <span>${escapeHtml(meta)}</span>
          </div>
        </div>
        <button
          class="oc-icon-btn oc-task__delete"
          type="button"
          title="Delete task"
          aria-label="Delete task"
          data-oc-action="delete"
          data-oc-task-id="${escapeHtml(task.id)}"
        >${icons.trash}</button>
      </article>
    `;
  }

  function renderEvent(event) {
    const tone = event.tone in eventToneLabels ? event.tone : "system";
    const detail = event.detail ? `<p>${escapeHtml(event.detail)}</p>` : "";
    const refs = [event.sessionKey ? compactText(event.sessionKey, 28) : "", event.runId ? "run" : ""]
      .filter(Boolean)
      .join(" · ");
    return `
      <article class="oc-event oc-event--${tone}">
        <div class="oc-event__marker">${eventIcon(tone)}</div>
        <div class="oc-event__body">
          <div class="oc-event__topline">
            <strong>${escapeHtml(event.title)}</strong>
            <span>${escapeHtml(timeAgo(event.at))}</span>
          </div>
          ${detail}
          ${
            refs
              ? `<div class="oc-event__meta">${escapeHtml(eventToneLabels[tone])} · ${escapeHtml(refs)}</div>`
              : `<div class="oc-event__meta">${escapeHtml(eventToneLabels[tone])}</div>`
          }
        </div>
      </article>
    `;
  }

  function renderEmptyTask(filter) {
    const text = filter === "all" ? "No tasks yet" : "No tasks match this filter";
    return `<div class="oc-empty">${icons.check}<span>${escapeHtml(text)}</span></div>`;
  }

  function renderEmptyEvents() {
    return `<div class="oc-empty">${icons.route}<span>Waiting for agent events</span></div>`;
  }

  function statusIcon(status) {
    if (status === "done") {
      return icons.check;
    }
    if (status === "doing") {
      return icons.clock;
    }
    if (status === "blocked") {
      return icons.alert;
    }
    return icons.circle;
  }

  function eventIcon(tone) {
    if (tone === "done") {
      return icons.check;
    }
    if (tone === "error" || tone === "warn") {
      return icons.alert;
    }
    if (tone === "tool") {
      return icons.panel;
    }
    return icons.route;
  }

  function timeAgo(timestamp) {
    const then = Number(timestamp || Date.now());
    const diff = Math.max(0, Date.now() - then);
    if (diff < 60_000) {
      return "just now";
    }
    if (diff < 3_600_000) {
      const minutes = Math.floor(diff / 60_000);
      return `${minutes} minute${minutes === 1 ? "" : "s"} ago`;
    }
    if (diff < 86_400_000) {
      const hours = Math.floor(diff / 3_600_000);
      return `${hours} hour${hours === 1 ? "" : "s"} ago`;
    }
    return new Intl.DateTimeFormat("en-US", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    }).format(new Date(then));
  }

  function injectStyles() {
    if (document.getElementById("oc-agent-workspace-style")) {
      return;
    }
    const style = document.createElement("style");
    style.id = "oc-agent-workspace-style";
    style.textContent = `
      .oc-workflow-form label { display: block; font-size: 12px; }
      .oc-workflow-form select { width: 100%; padding: 8px; color: inherit; background: var(--bg, #152033); border: 1px solid var(--border, #40516a); border-radius: 6px; }
      .oc-workflow-form input[type="checkbox"] { width: auto; margin-right: 6px; }
      .oc-workflow-form p { margin: 0; font-size: 11px; line-height: 1.5; }
      body.oc-agent-enhanced {
        background:
          linear-gradient(180deg, color-mix(in srgb, var(--bg) 94%, var(--accent-2) 6%) 0%, var(--bg) 52%),
          var(--bg);
      }

      body.oc-agent-enhanced .topbar {
        border-bottom-color: color-mix(in srgb, var(--border-strong) 76%, transparent);
        box-shadow: 0 12px 30px rgba(0,0,0,.14);
      }

      body.oc-agent-enhanced .content:not(.content--chat) {
        background:
          linear-gradient(180deg, color-mix(in srgb, var(--bg-accent) 72%, transparent), transparent 260px),
          var(--bg);
      }

      body.oc-agent-enhanced .card {
        border-color: color-mix(in srgb, var(--border-strong) 70%, transparent);
        box-shadow: 0 14px 34px rgba(0,0,0,.18), inset 0 1px rgba(255,255,255,.05);
      }

      body.oc-agent-enhanced .nav-item.active,
      body.oc-agent-enhanced .nav-item--active {
        border-color: color-mix(in srgb, var(--accent) 34%, transparent);
        box-shadow: inset 0 1px rgba(255,255,255,.12), 0 12px 28px rgba(0,0,0,.22);
      }

      .shell.shell--agent-workspace-open:not(.shell--onboarding) {
        grid-template-columns: var(--shell-nav-width) minmax(0, 1fr) minmax(342px, 372px);
        grid-template-areas:
          "nav topbar ocworkspace"
          "nav content ocworkspace";
      }

      .shell.shell--agent-workspace-open.shell--nav-collapsed:not(.shell--onboarding) {
        grid-template-columns: var(--shell-nav-rail-width) minmax(0, 1fr) minmax(342px, 372px);
      }

      .shell.shell--agent-workspace-open.shell--chat-focus:not(.shell--onboarding) {
        grid-template-columns: 0 minmax(0, 1fr) minmax(342px, 372px);
      }

      .oc-agent-workspace {
        display: none;
        grid-area: ocworkspace;
        min-width: 0;
        min-height: 0;
        color: var(--text);
        background:
          linear-gradient(180deg, color-mix(in srgb, var(--bg-elevated) 92%, var(--accent-2) 8%), var(--bg) 72%),
          var(--bg);
        border-left: 1px solid color-mix(in srgb, var(--border-strong) 74%, transparent);
        box-shadow: -18px 0 44px rgba(0,0,0,.22);
        overflow: hidden;
        pointer-events: auto;
        z-index: 120;
      }

      .shell--agent-workspace-open .oc-agent-workspace {
        display: flex;
      }

      .oc-agent-workspace__inner {
        overflow-y: auto;
        display: grid;
        grid-template-rows: auto auto auto minmax(190px, auto) minmax(220px, auto);
        gap: 12px;
        width: 100%;
        min-height: 0;
        padding: 14px;
      }

      .oc-agent-workspace__header,
      .oc-agent-card__header,
      .oc-task,
      .oc-stat,
      .oc-task-form,
      .oc-filter-row {
        border-radius: 8px;
      }

      .oc-agent-workspace__header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 2px 2px 4px;
      }

      .oc-agent-workspace__eyebrow {
        color: var(--muted);
        font-size: 10px;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
      }

      .oc-agent-workspace__header h2,
      .oc-agent-card__header h3,
      .oc-agent-card__header p {
        margin: 0;
      }

      .oc-agent-workspace__header h2 {
        color: var(--text-strong);
        font-size: 17px;
        font-weight: 720;
        letter-spacing: 0;
        line-height: 1.2;
      }

      .oc-agent-stats {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 8px;
      }

      .oc-stat {
        display: grid;
        gap: 2px;
        min-height: 58px;
        padding: 10px 8px;
        color: var(--text);
        text-align: left;
        background: color-mix(in srgb, var(--bg-elevated) 82%, transparent);
        border: 1px solid color-mix(in srgb, var(--border) 76%, transparent);
        cursor: pointer;
      }

      .oc-stat:hover,
      .oc-filter:hover,
      .oc-mini-btn:hover,
      .oc-icon-btn:hover {
        border-color: color-mix(in srgb, var(--accent) 34%, var(--border-strong));
        background: color-mix(in srgb, var(--bg-hover) 82%, transparent);
      }

      .oc-stat span {
        color: var(--muted);
        font-size: 11px;
        font-weight: 650;
      }

      .oc-stat strong {
        color: var(--text-strong);
        font-size: 19px;
        line-height: 1.1;
      }

      .oc-stat--doing strong { color: var(--info); }
      .oc-stat--done strong { color: var(--ok); }
      .oc-stat--blocked strong { color: var(--warn); }

      .oc-agent-card {
        display: grid;
        grid-template-rows: auto auto auto minmax(0, 1fr);
        gap: 10px;
        min-height: 0;
        padding: 12px;
        background: color-mix(in srgb, var(--card) 86%, transparent);
        border: 1px solid color-mix(in srgb, var(--border-strong) 68%, transparent);
        border-radius: 8px;
        box-shadow: inset 0 1px rgba(255,255,255,.06);
      }

      .oc-agent-card--timeline {
        grid-template-rows: auto minmax(0, 1fr);
      }

      .oc-agent-card--workflow {
        grid-template-rows: auto auto;
      }

      .oc-agent-card__header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
      }

      .oc-agent-card__header h3 {
        color: var(--text-strong);
        font-size: 14px;
        font-weight: 720;
        letter-spacing: 0;
      }

      .oc-agent-card__header p {
        color: var(--muted);
        font-size: 11px;
      }

      .oc-workspace-toggle,
      .oc-mini-btn,
      .oc-icon-btn,
      .oc-filter,
      .oc-task__status,
      .oc-task__delete {
        -webkit-tap-highlight-color: transparent;
      }

      .oc-workspace-toggle {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        height: 34px;
        padding: 0 12px;
        color: var(--muted);
        background: color-mix(in srgb, var(--bg-elevated) 80%, transparent);
        border: 1px solid color-mix(in srgb, var(--border) 84%, transparent);
        border-radius: 999px;
        cursor: pointer;
        font-size: 12px;
        font-weight: 650;
        transition: background var(--duration-fast) ease, border-color var(--duration-fast) ease, color var(--duration-fast) ease;
      }

      .oc-workspace-toggle[aria-pressed="true"] {
        color: var(--accent);
        border-color: color-mix(in srgb, var(--accent) 34%, transparent);
        background: var(--accent-subtle);
      }

      .oc-workspace-toggle svg,
      .oc-icon-btn svg,
      .oc-task__status svg,
      .oc-event__marker svg {
        width: 16px;
        height: 16px;
        stroke: currentColor;
        stroke-width: 1.8;
        stroke-linecap: round;
        stroke-linejoin: round;
        fill: none;
      }

      .oc-icon-btn {
        display: inline-grid;
        place-items: center;
        flex: 0 0 auto;
        width: 34px;
        height: 34px;
        color: var(--muted);
        background: color-mix(in srgb, var(--bg-elevated) 82%, transparent);
        border: 1px solid color-mix(in srgb, var(--border) 82%, transparent);
        border-radius: 8px;
        cursor: pointer;
      }

      .oc-icon-btn--accent {
        color: var(--accent-foreground);
        background: var(--accent);
        border-color: color-mix(in srgb, var(--accent) 78%, white 12%);
      }

      .oc-mini-btn {
        min-height: 30px;
        padding: 0 10px;
        color: var(--muted);
        background: color-mix(in srgb, var(--bg-elevated) 84%, transparent);
        border: 1px solid color-mix(in srgb, var(--border) 80%, transparent);
        border-radius: 8px;
        cursor: pointer;
        font-size: 11px;
        font-weight: 650;
      }

      .oc-mini-btn--accent {
        color: var(--accent-foreground);
        background: var(--accent);
        border-color: color-mix(in srgb, var(--accent) 78%, white 12%);
      }

      .oc-workflow-pill {
        display: inline-flex;
        align-items: center;
        min-height: 22px;
        padding: 0 8px;
        color: var(--accent-2);
        background: color-mix(in srgb, var(--accent-subtle) 70%, var(--bg-elevated) 30%);
        border: 1px solid color-mix(in srgb, var(--accent-2) 28%, transparent);
        border-radius: 999px;
        font-size: 10.5px;
        font-weight: 680;
        white-space: nowrap;
      }

      .oc-workflow-form {
        display: grid;
        grid-template-columns: minmax(0, 1fr);
        gap: 8px;
      }

      .oc-workflow-form textarea {
        width: 100%;
        min-width: 0;
        min-height: 58px;
        max-height: 128px;
        resize: vertical;
        color: var(--text);
        background: color-mix(in srgb, var(--bg) 74%, transparent);
        border: 1px solid color-mix(in srgb, var(--border) 72%, transparent);
        border-radius: 8px;
        outline: none;
        padding: 9px 10px;
        font: inherit;
        font-size: 12px;
        line-height: 1.38;
      }

      .oc-workflow-form textarea:focus {
        border-color: color-mix(in srgb, var(--accent) 48%, var(--border-strong));
        box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 18%, transparent);
      }

      .oc-workflow-form textarea::placeholder {
        color: color-mix(in srgb, var(--muted) 80%, transparent);
      }

      .oc-task-form {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 8px;
        padding: 8px;
        background: color-mix(in srgb, var(--bg) 74%, transparent);
        border: 1px solid color-mix(in srgb, var(--border) 72%, transparent);
      }

      .oc-task-form input {
        min-width: 0;
        height: 34px;
        color: var(--text);
        background: transparent;
        border: 0;
        outline: none;
        padding: 0 4px;
      }

      .oc-task-form input::placeholder {
        color: color-mix(in srgb, var(--muted) 80%, transparent);
      }

      .oc-filter-row {
        display: flex;
        gap: 4px;
        padding: 4px;
        background: color-mix(in srgb, var(--bg) 70%, transparent);
        border: 1px solid color-mix(in srgb, var(--border) 70%, transparent);
        overflow-x: auto;
      }

      .oc-filter {
        flex: 0 0 auto;
        min-height: 28px;
        padding: 0 9px;
        color: var(--muted);
        background: transparent;
        border: 1px solid transparent;
        border-radius: 6px;
        cursor: pointer;
        font-size: 11px;
        font-weight: 650;
      }

      .oc-filter--active {
        color: var(--text-strong);
        background: color-mix(in srgb, var(--accent-subtle) 84%, var(--bg-elevated) 16%);
        border-color: color-mix(in srgb, var(--accent) 22%, transparent);
      }

      .oc-task-list,
      .oc-timeline {
        display: grid;
        align-content: start;
        gap: 8px;
        min-height: 0;
        overflow: auto;
        padding-right: 2px;
      }

      .oc-task {
        display: grid;
        grid-template-columns: auto minmax(0, 1fr) auto;
        align-items: start;
        gap: 9px;
        padding: 10px;
        background: color-mix(in srgb, var(--bg-elevated) 74%, transparent);
        border: 1px solid color-mix(in srgb, var(--border) 72%, transparent);
      }

      .oc-task--done {
        opacity: .72;
      }

      .oc-task--done .oc-task__title {
        text-decoration: line-through;
        text-decoration-color: color-mix(in srgb, var(--ok) 72%, transparent);
      }

      .oc-task__status {
        display: grid;
        place-items: center;
        width: 28px;
        height: 28px;
        color: var(--muted);
        background: color-mix(in srgb, var(--bg) 68%, transparent);
        border: 1px solid color-mix(in srgb, var(--border) 72%, transparent);
        border-radius: 999px;
        cursor: pointer;
      }

      .oc-task--doing .oc-task__status { color: var(--info); }
      .oc-task--done .oc-task__status { color: var(--ok); }
      .oc-task--blocked .oc-task__status { color: var(--warn); }

      .oc-task__body {
        min-width: 0;
      }

      .oc-task__title {
        color: var(--text-strong);
        font-size: 12.5px;
        font-weight: 650;
        line-height: 1.35;
        overflow-wrap: anywhere;
      }

      .oc-task__meta {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 6px;
        margin-top: 6px;
        color: var(--muted);
        font-size: 10.5px;
      }

      .oc-task__badge {
        display: inline-flex;
        align-items: center;
        min-height: 18px;
        padding: 0 7px;
        border-radius: 999px;
        border: 1px solid color-mix(in srgb, var(--border) 70%, transparent);
      }

      .oc-task__badge--doing {
        color: var(--info);
        border-color: color-mix(in srgb, var(--info) 36%, transparent);
      }

      .oc-task__badge--done {
        color: var(--ok);
        border-color: color-mix(in srgb, var(--ok) 36%, transparent);
      }

      .oc-task__badge--blocked {
        color: var(--warn);
        border-color: color-mix(in srgb, var(--warn) 42%, transparent);
      }

      .oc-task__delete {
        width: 28px;
        height: 28px;
        opacity: .68;
      }

      .oc-task:hover .oc-task__delete {
        opacity: 1;
      }

      .oc-event {
        display: grid;
        grid-template-columns: 26px minmax(0, 1fr);
        gap: 8px;
        position: relative;
      }

      .oc-event:not(:last-child)::before {
        content: "";
        position: absolute;
        left: 12px;
        top: 28px;
        bottom: -10px;
        width: 1px;
        background: color-mix(in srgb, var(--border-strong) 62%, transparent);
      }

      .oc-event__marker {
        display: grid;
        place-items: center;
        width: 26px;
        height: 26px;
        color: var(--muted);
        background: color-mix(in srgb, var(--bg-elevated) 76%, transparent);
        border: 1px solid color-mix(in srgb, var(--border) 76%, transparent);
        border-radius: 999px;
        z-index: 1;
      }

      .oc-event--tool .oc-event__marker { color: var(--accent-2); }
      .oc-event--done .oc-event__marker { color: var(--ok); }
      .oc-event--warn .oc-event__marker { color: var(--warn); }
      .oc-event--error .oc-event__marker { color: var(--danger); }
      .oc-event--live .oc-event__marker { color: var(--info); }

      .oc-event__body {
        min-width: 0;
        padding: 8px 9px;
        background: color-mix(in srgb, var(--bg-elevated) 70%, transparent);
        border: 1px solid color-mix(in srgb, var(--border) 66%, transparent);
        border-radius: 8px;
      }

      .oc-event__topline {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 8px;
      }

      .oc-event__topline strong {
        min-width: 0;
        color: var(--text-strong);
        font-size: 12px;
        font-weight: 680;
        overflow-wrap: anywhere;
      }

      .oc-event__topline span {
        flex: 0 0 auto;
        color: var(--muted);
        font-size: 10.5px;
      }

      .oc-event__body p {
        margin: 5px 0 0;
        color: var(--text);
        font-family: var(--mono);
        font-size: 10.8px;
        line-height: 1.45;
        overflow-wrap: anywhere;
      }

      .oc-event__meta {
        margin-top: 6px;
        color: var(--muted);
        font-size: 10.5px;
      }

      .oc-empty {
        display: grid;
        place-items: center;
        gap: 8px;
        min-height: 92px;
        color: var(--muted);
        border: 1px dashed color-mix(in srgb, var(--border-strong) 58%, transparent);
        border-radius: 8px;
        background: color-mix(in srgb, var(--bg) 58%, transparent);
        font-size: 12px;
      }

      .oc-empty svg {
        width: 22px;
        height: 22px;
        stroke: currentColor;
        stroke-width: 1.7;
        stroke-linecap: round;
        stroke-linejoin: round;
        fill: none;
      }

      @media (width<=1180px) {
        .shell.shell--agent-workspace-open:not(.shell--onboarding),
        .shell.shell--agent-workspace-open.shell--nav-collapsed:not(.shell--onboarding),
        .shell.shell--agent-workspace-open.shell--chat-focus:not(.shell--onboarding) {
          grid-template-columns: var(--shell-nav-width) minmax(0, 1fr);
          grid-template-areas:
            "nav topbar"
            "nav content";
        }

        .shell.shell--agent-workspace-open.shell--nav-collapsed:not(.shell--onboarding) {
          grid-template-columns: var(--shell-nav-rail-width) minmax(0, 1fr);
        }

        .shell.shell--agent-workspace-open.shell--chat-focus:not(.shell--onboarding) {
          grid-template-columns: 0 minmax(0, 1fr);
        }

        .oc-agent-workspace {
          position: fixed;
          top: 0;
          right: 0;
          bottom: 0;
          width: min(92vw, 380px);
          max-width: 380px;
          z-index: 140;
        }
      }

      @media (width<=720px) {
        .oc-workspace-toggle span {
          display: none;
        }

        .oc-workspace-toggle {
          width: 34px;
          padding: 0;
          justify-content: center;
        }

        .oc-agent-workspace__inner {
          padding: 12px;
          grid-template-rows: auto auto auto minmax(190px, .92fr) minmax(220px, 1.08fr);
        }

        .oc-agent-stats {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }
      }

      :root[data-theme-mode=light] .oc-agent-workspace {
        background:
          linear-gradient(180deg, color-mix(in srgb, #fff 88%, var(--accent-2) 12%), var(--bg) 72%),
          var(--bg);
        box-shadow: -18px 0 38px rgba(0,0,0,.12);
      }

      :root[data-theme-mode=light] .oc-agent-card,
      :root[data-theme-mode=light] .oc-task,
      :root[data-theme-mode=light] .oc-event__body,
      :root[data-theme-mode=light] .oc-stat {
        background: rgba(255,255,255,.78);
      }
    `;
    document.head.appendChild(style);
  }
})();
