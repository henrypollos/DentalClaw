# OpenClaw clinical_result_report skill pack

Copy the `clinical_result_report/` folder into a skill-visible directory, or add its parent directory to `skills.load.extraDirs`.

Validate with:
- `openclaw skills list --eligible`
- `openclaw skills info clinical_result_report`
- `openclaw skills check`

OpenClaw can attach local media when the agent prints `MEDIA:<path>` on its own line. Keep the output directory under an allowed OpenClaw workspace path so the media attachment is accepted.
