# OpenClaw browser integration

This directory packages DentalClaw's custom workflow panel and connects it to
`platform_mvp/run_platform_mvp.py --next-version`. OpenClaw supplies the chat UI,
authentication, Gateway and coordinator agent; the DentalClaw backend chooses
among the four registered routes. The panel does not choose a route or default
case, and it sends the complete request without shortening its constraints.

## Prepare an installation

Install and configure OpenClaw and a chat model first. The reference installation
inspected for this adapter is OpenClaw **2026.3.13**. Its Control UI uses the
`.shell` DOM container and the `chat.send`/`chat` Gateway messages. Newer versions
may change these interfaces: preparation records the package version but is not
a compatibility certification. The builder copies your installed assets; it does
not vendor OpenClaw or modify its installation.

From the DentalClaw repository, using Python 3.10 or later:

```sh
python integrations/openclaw/prepare_ui.py \
  --openclaw-package /absolute/path/to/node_modules/openclaw \
  --out-dir artifacts/openclaw-ui \
  --python /absolute/path/to/python \
  --base-config /absolute/path/to/openclaw.json
```

`--openclaw-package` is the package containing `package.json`, `openclaw.mjs` and
`dist/control-ui/`. With npm it is usually below `npm root -g`; follow the
`openclaw` executable's symlink to locate other installation layouts.

The output must be new. The command prepares:

```text
artifacts/openclaw-ui/
  control-ui/                 Served UI assets, including the custom panel
  agent-workspace/            Dedicated coordinator instructions
  context.empty.json         Default unbound input context
  openclaw.fragment.json     UI-root and agent configuration additions
  openclaw.local.json        Merged private config, only with --base-config
  build_manifest.json        Source version and asset hashes
```

The merge preserves existing agents and provider/auth settings, adding a separate
`dentalclaw` agent. An existing agent with that ID is rejected; use `--agent-id`
with a new ID instead of replacing its workspace. `--model provider/model` can
select the coordinator model explicitly; otherwise it inherits the configured
OpenClaw default. The coordinator must have functioning credentials and exec/read
capabilities under your existing host approval and sandbox policy. The builder
does not loosen that policy. Provider auth profiles belonging only to another
agent may need to be configured for this new agent through OpenClaw.

The base config must be JSON; JSON5 users can merge `openclaw.fragment.json`
manually. `openclaw.local.json` is written with mode 0600 outside the served UI
root because it may contain existing credentials. Keep the entire generated
output under ignored `artifacts/`; do not commit it.

## Start the prepared interface

After reviewing the generated config, launch the Gateway with it. Stop or manage
the old Gateway separately if it already occupies the configured port. The
preparation command does not stop, restart or reconfigure a running service.

```sh
OPENCLAW_CONFIG_PATH=/absolute/path/to/artifacts/openclaw-ui/openclaw.local.json \
  openclaw gateway run

# In another terminal, using the same configuration:
OPENCLAW_CONFIG_PATH=/absolute/path/to/artifacts/openclaw-ui/openclaw.local.json \
  openclaw dashboard
```

For remote access, tunnel the configured port (commonly 18789):

```sh
ssh -N -L 18789:127.0.0.1:18789 <user>@<workstation>
```

Open `http://127.0.0.1:18789/` locally, authenticate using your Gateway settings,
and choose the **dentalclaw** agent's chat (session key
`agent:dentalclaw:main`, or another session for that agent). Respect any device
pairing or execution approval requested by OpenClaw. The panel refuses to send to
a different agent. It only enables delegation after the Gateway connect response
succeeds. If the workspace panel is hidden, use the **Workspace** button.

## Bind inputs

Without `--profiles` the panel uses an empty context. It cannot infer local data
or invent model bindings. To use real data, create a server-side context, e.g.:

```json
{"data_root": "/absolute/path/to/binary2d_package"}
```

Create a profiles JSON file listing contexts:

```json
[
  {"id": "audit", "label": "Binary raster package", "contextPath": "/absolute/path/to/qc.context.json"},
  {"id": "saved", "label": "Saved results", "contextPath": "/absolute/path/to/results.context.json"}
]
```

Pass `--profiles /absolute/path/to/profiles.json` when preparing a fresh UI output.
Context schemas and TDD/model requirements are documented in
[the planner guide](../../platform_mvp/next_version/README.md) and
[data services](../../platform_mvp/data_services/README.md). A saved-result context
uses `result_manifest`; a TDD context needs the bound dataset identity and report
case selection. A training context needs a compatible partitioned package.
The browser's public config contains only profile labels and context paths, not
context contents, dataset files or credentials. Do not put secrets in labels.

The Gateway coordinator sees the request and the command's server paths. The
backend planner sends the research request and allowlisted factual metadata to
its separately configured endpoint. Set `DEEPSEEK_API_KEY` in the **Gateway's
process environment** or configure the backend's local credential source. A key
in a different terminal is not automatically inherited by an existing service.
The coordinator's model and backend routing model are separate calls; browser
operation is not a one-call reproduction of the paper's decision evaluation.

## Use the panel

1. Select the **dentalclaw** chat and an **Input context**.
2. Enter the complete research request, including required outputs and limits.
3. Leave both execution checkboxes off to generate a plan only.
4. Enable **Execute if the route is ready** for delivery/QC. Training or inference
   additionally requires **Allow training or inference**. Local preflight still
   applies and can block execution.
5. **Check complete deliverables (development)** adds the optional
   `--coverage-check`; it is not part of the final first-test configuration.
6. Submit and inspect the chat's decision, execution status and evidence paths.

Examples: “Audit pairing, dimensions and binary labels; do not train.”
“Deliver the bound saved overlays and stored metrics; do not run inference.”
“Run the bound TDD model on the full test set and report the selected example.”
“Run a smoke training workflow on this explicitly partitioned binary package.”

Runs are stored in `artifacts/openclaw_runs/` with a new run ID. The coordinator
reads `prediction.json`, `platform_plan.json` and, when present,
`execution_result.json`. A completed chat returns the task to review; it does not
automatically mark workflow delivery successful. Task-board marks are user/agent
tracking state, not scientific execution evidence. HTML reports remain local
artifacts: the panel does not expose arbitrary filesystem paths as HTTP downloads.

This is an agent-mediated integration, not a deterministic Gateway execution
plugin. The coordinator still has to invoke the generated command correctly.
Check tool traces and artifacts; host approval failures or model errors must not
be interpreted as successful backend execution. The CLI remains available for
direct operation without the outer chat agent.

## Checks and upstream references

```sh
python -m pytest tests/test_openclaw_ui.py -q
node --test tests/test_openclaw_workflow.cjs
```

These checks cover configuration isolation, argument preservation, execution
controls and an offline browser-command-to-backend replay. They do not call a
live coordinator or routing model and do not test clinical outcomes.

OpenClaw's [Control UI documentation](https://docs.openclaw.ai/web/control-ui)
and [configuration documentation](https://docs.openclaw.ai/gateway/configuration)
cover the upstream Gateway, authentication and runtime configuration. Upstream
OpenClaw assets retain their own license; only the DentalClaw customization and
preparation code are maintained here.
