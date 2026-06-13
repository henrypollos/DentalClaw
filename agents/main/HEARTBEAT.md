# HEARTBEAT.md

# Supervision heartbeat for the main agent.
#
# On heartbeat:
# 1. Run:
#    /home/yiyang/miniconda3/envs/nnunetv2/bin/python /data/data2/yiyang/DentalClaw/agents/main/skills/supervision-registry/scripts/refresh_registry.py
# 2. Read:
#    - /data/data2/yiyang/DentalClaw/registry/overview.md
#    - /data/data2/yiyang/DentalClaw/registry/task_runs.json
#    - /data/data2/yiyang/DentalClaw/registry/dataset_qc_reports.json
# 3. If there are running tasks, failed tasks, QC failures, or newly appeared datasets/models, summarize them briefly.
# 4. If nothing changed materially, reply HEARTBEAT_OK.
