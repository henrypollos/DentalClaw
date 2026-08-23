# HEARTBEAT.md

# Supervision heartbeat for the main agent.
#
# On heartbeat:
# 1. Run:
#    $CONDA_HOME/envs/nnunetv2/bin/python $DENTALCLAW_HOME/agents/main/skills/supervision-registry/scripts/refresh_registry.py
# 2. Read:
#    - $DENTALCLAW_HOME/registry/overview.md
#    - $DENTALCLAW_HOME/registry/task_runs.json
#    - $DENTALCLAW_HOME/registry/dataset_qc_reports.json
# 3. If there are running tasks, failed tasks, QC failures, or newly appeared datasets/models, summarize them briefly.
# 4. If nothing changed materially, reply HEARTBEAT_OK.
