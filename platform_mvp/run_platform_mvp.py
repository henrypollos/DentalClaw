#!/usr/bin/env python3
"""DentalClaw platform MVP orchestrator.

This file is deliberately small and explicit. It turns one user sentence into a
structured platform plan, performs an offline method-registry lookup, and can
delegate the first executable route to the existing full-flow MVP.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
THIS_DIR = Path(__file__).resolve().parent

# 尽早导入，避免函数内 lazy import 失败
try:
    from benchmark_trace.openclaw_runner import call_openclaw_agent as _call_ocl
except ImportError:
    _call_ocl = None
from schemas.agent_trace import TraceRecorder, create_trace_recorder, AgentTraceEvent
from platform_mvp.i18n import _, detect_language

DEFAULT_REGISTRY = THIS_DIR / "method_registry.json"
DEFAULT_DENTALCLAW_PYTHON = "$CONDA_HOME/envs/nnunetv2/bin/python"

# DeepSeek API 配置（从 OpenClaw auth 配置读取，作为 fallback）
def _load_deepseek_api_key() -> str | None:
    """从 OpenClaw agent auth-profiles.json 读取 DeepSeek API key。"""
    try:
        auth_path = Path.home() / ".openclaw/agents/main/agent/auth-profiles.json"
        if auth_path.exists():
            profiles = json.loads(auth_path.read_text())
            deepseek = profiles.get("profiles", {}).get("deepseek:default", {})
            return deepseek.get("key")
    except Exception:
        pass
    # Fallback: 从环境变量
    return os.environ.get("DEEPSEEK_API_KEY")


def _call_deepseek_api(prompt: str, timeout: int = 60) -> dict[str, Any] | None:
    """直接调用 DeepSeek API (chat/completions)，不依赖 OpenClaw CLI。"""
    import urllib.request
    import urllib.error

    api_key = _load_deepseek_api_key()
    if not api_key:
        print("[platform_mvp] DeepSeek API key not found, direct API call disabled", file=sys.stderr)
        return None

    data = json.dumps({
        "model": "deepseek-v4-pro",
        "messages": [
            {"role": "system", "content": "You are a DentalClaw platform planner. Always reply with valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 2048,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.deepseek.com/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content:
                print(f"[platform_mvp] DeepSeek returned empty content. finish_reason={body.get('choices',[{}])[0].get('finish_reason','?')}", file=sys.stderr)
                return None
            # DeepSeek 可能返回 ```json ... ``` 包裹的 JSON，也可能带前缀说明
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
            # 提取第一个 { 到最后一个 } 之间的内容（处理模型多输出文字的情况）
            brace_start = content.find("{")
            brace_end = content.rfind("}")
            if brace_start >= 0 and brace_end > brace_start:
                content = content[brace_start:brace_end + 1]
            return json.loads(content)
    except json.JSONDecodeError as je:
        print(f"[platform_mvp] DeepSeek returned invalid JSON. Raw (first 300): {content[:300]}", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"[platform_mvp] DeepSeek API call failed: {exc}", file=sys.stderr)
        return None
DEFAULT_SUCCESSFUL_FULLFLOW_RUN = (
    REPO_ROOT / "artifacts/mvp_runs/tdd_binary_fullflow_20260709_085641"
)


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _rel(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _contains_any(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def parse_intent(intent: str) -> dict[str, Any]:
    """Parse a one-sentence medical CV request into a narrow MVP intent."""
    dataset = "unknown"
    if _contains_any(intent, ["tdd", "全景", "panoramic", "dataset501"]):
        dataset = "TDD"
    elif _contains_any(intent, ["toothfairy", "cbct", "锥形束", "三维", "3d"]):
        dataset = "ToothFairy3"
    elif _contains_any(intent, ["私有", "自己的", "我提供的", "我自己的", "private", "本院", "院内"]):
        dataset = "Private2D"

    task_family = "unknown"
    if _contains_any(intent, ["检测框", "检测模型", "bbox", "detector"]):
        # 检测框/检测模型是强检测信号，优先于后文的 segmentation 等词
        # （如 “把 segmentation mask 转成检测框并训练 detector” 应解析为 detection）
        task_family = "detection"
    elif _contains_any(intent, ["超分", "super-resolution", "super resolution", "分辨率", "清晰"]):
        task_family = "super_resolution"
    elif _contains_any(intent, ["异常", "anomaly", "病灶", "可疑"]):
        task_family = "anomaly_detection"
    elif _contains_any(intent, ["分割", "segmentation", "segment", "mask", "掩膜"]):
        task_family = "segmentation"
    elif _contains_any(intent, ["检测", "detection", "detect", "框"]):
        task_family = "detection"
    elif _contains_any(intent, ["分类", "classification", "classify"]):
        task_family = "classification"

    # TTA / ensemble flags
    tta = _contains_any(intent, ["tta", "test-time augmentation", "测试时增强", "test time"])
    ensemble = _contains_any(intent, ["ensemble", "集成", "ensemble学习", "多模型"])
    task_variant = ""
    if tta and ensemble:
        task_variant = "tta_ensemble"
    elif tta:
        task_variant = "tta"
    elif ensemble:
        task_variant = "ensemble"

    # TTA/Ensemble 未指定数据集时默认 TDD
    if task_variant and dataset == "unknown":
        dataset = "TDD"

    modality = "2d"
    if _contains_any(intent, ["3d", "三维", "cbct", "体积", "volume"]):
        modality = "3d"

    # mode 判定：训练优先；显式否定（不做/不进行训练）或“重新训练”否定词 → 推理。
    # 注意：单独出现“报告/评估”不应覆盖“训练”（如“训练检测模型并报告 mAP”应为 train）。
    mode = "inference"
    has_train = _contains_any(intent, ["训练", "train", "微调", "fine-tune", "finetune"])
    train_negated = _contains_any(intent, ["不做监督训练", "不做训练", "不进行训练",
                                          "不要重新训练", "不重新训练"])
    if has_train and not train_negated:
        mode = "private_train" if dataset == "Private2D" else "train"
    elif _contains_any(intent, ["推理", "预测", "评估", "报告"]):
        mode = "inference"

    return {
        "raw_intent": intent,
        "dataset": dataset,
        "task_family": task_family,
        "task_variant": task_variant,
        "modality": modality,
        "mode": mode,
        "tta": tta,
        "ensemble": ensemble,
    }


def load_registry(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if "methods" not in payload or not isinstance(payload["methods"], list):
        raise ValueError(f"Invalid registry: {path}")
    return payload


def _save_registry(path: Path, registry: dict[str, Any]) -> None:
    """原子写入 registry JSON。"""
    import tempfile
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise


_AGENT_DISCOVERIES_PATH = REPO_ROOT / "platform_mvp" / "agent_discoveries.json"
_AUTO_REGISTER_THRESHOLD = 2  # 同一资产被发现 ≥N 次 → 自动注册


def _load_discoveries() -> dict[str, Any]:
    """加载 Agent 发现记录。"""
    if not _AGENT_DISCOVERIES_PATH.exists():
        return {"discoveries": {}}
    return _read_json(_AGENT_DISCOVERIES_PATH)


def _save_discoveries(data: dict[str, Any]) -> None:
    _AGENT_DISCOVERIES_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _track_agent_discovery(entrypoint: str, display_name: str, intent: str, registry_path: Path) -> int:
    """记录 Agent 发现的一个代码库资产，返回该资产的历史发现次数。

    当同一资产被发现 ≥ AUTO_REGISTER_THRESHOLD 次时，自动注册到 registry。
    """
    data = _load_discoveries()
    discoveries = data.setdefault("discoveries", {})

    # 标准化 asset key（用 entrypoint 路径）
    asset_key = entrypoint.strip().lstrip("/")
    if asset_key not in discoveries:
        discoveries[asset_key] = {"count": 0, "display_name": display_name, "intents": []}

    discoveries[asset_key]["count"] += 1
    discoveries[asset_key]["intents"].append({
        "intent": intent[:200],
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
    })
    # 保留最近 5 条触发记录
    discoveries[asset_key]["intents"] = discoveries[asset_key]["intents"][-5:]

    _save_discoveries(data)
    count = discoveries[asset_key]["count"]

    # 达到阈值：自动注册到 registry
    if count >= _AUTO_REGISTER_THRESHOLD:
        _auto_register_to_registry(asset_key, display_name, registry_path)

    return count


def _auto_register_to_registry(asset_key: str, display_name: str, registry_path: Path) -> bool:
    """将 Agent 发现的资产自动注册到 registry（planned_adapter 状态）。"""
    registry = load_registry(registry_path)
    existing_ids = {m["id"] for m in registry.get("methods", [])}

    # 生成 method ID
    safe_id = "agent_" + asset_key.replace("/", "_").replace(".", "_").replace("-", "_").strip("_")[:50]

    if safe_id in existing_ids:
        return False  # 已注册

    new_method = {
        "id": safe_id,
        "display_name": f"Agent discovered: {display_name}",
        "domain": "dental_cv",
        "task_family": "auto",
        "datasets": ["auto"],
        "modality": "auto",
        "allowed_modes": ["auto"],
        "status": "planned_adapter",
        "entrypoint": asset_key,
        "framework": "auto",
        "auto_registered": True,
        "agent_discovered_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "limitations": ["Auto-registered from Agent repeated discoveries. Pending manual review."],
    }

    registry["methods"].append(new_method)
    _save_registry(registry_path, registry)
    print(f"[platform_mvp] Auto-registered Agent discovery in registry: {safe_id} (discovered {_AUTO_REGISTER_THRESHOLD}+ times)",
          file=sys.stderr)
    return True


def _dataset_matches(method: dict[str, Any], dataset: str) -> bool:
    if dataset == "unknown":
        return False
    return dataset in method.get("datasets", [])


def select_method(parsed: dict[str, Any], registry: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    candidates = []
    for method in registry["methods"]:
        if method.get("domain") != "dental_cv":
            continue
        if method.get("task_family") != parsed["task_family"]:
            continue
        if method.get("modality") != parsed["modality"]:
            continue
        if parsed["mode"] not in method.get("allowed_modes", []):
            continue
        if not _dataset_matches(method, parsed["dataset"]):
            continue
        score = 1
        # Boost TTA/Ensemble route when flags are set
        if parsed.get("tta") or parsed.get("ensemble"):
            if "tta_ensemble" in method.get("id", ""):
                score += 10  # 大幅提升，确保优先匹配
        score += 4
        score += 4
        score += 2
        score += 3
        candidates.append((score, method))

    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates:
        reasons.append("No method in the offline registry matched the parsed request.")
        return None, reasons

    _, best_method = candidates[0]
    reasons.append(
        f"Selected {best_method['id']} because dataset, task family, modality, and mode all matched the offline registry."
    )
    if best_method.get("status") != "executable":
        reasons.append(
            "The route is part of the platform scope but does not yet have an executable adapter."
        )
    return best_method, reasons


def _web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """免费 web search，依次尝试 ddgs / duckduckgo_search，均不可用则 Bing HTML 爬取。"""
    # 尝试 1: ddgs (新包名，优先)
    try:
        from ddgs import DDGS  # type: ignore[import-not-found]
        results = list(DDGS().text(query, max_results=max_results))
        if results:
            return [{"title": r.get("title", "?"), "href": r.get("href", "")} for r in results]
    except Exception:
        pass
    # 尝试 2: duckduckgo_search (旧包名，已废弃但兼容)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from duckduckgo_search import DDGS  # type: ignore[import-not-found,no-redef]
        results = list(DDGS().text(query, max_results=max_results))
        if results:
            return [{"title": r.get("title", "?"), "href": r.get("href", "")} for r in results]
    except Exception:
        pass
    # 尝试 3: 简单 curl 百度/Bing（不依赖第三方库）
    try:
        import urllib.request
        import urllib.parse
        q = urllib.parse.quote(query)
        # 用 Bing 搜索（无需 API key，轻量 HTML）
        url = f"https://www.bing.com/search?q={q}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        # 简单提取 <a> 标签中的结果
        import re
        links = re.findall(r'<a[^>]*href="(https?://[^"]+)"[^>]*>\s*<h[23][^>]*>([^<]+)', html)
        if links:
            return [{"title": t.strip(), "href": h} for h, t in links[:max_results]]
    except Exception:
        pass
    return []


def _auto_configure(proposal: dict[str, Any], workspace_dir: Path, dentalclaw_python: str) -> dict[str, Any]:
    """将 Agent 提议的外部方案自动安装依赖、生成 thin adapter。

    包名解析策略（按优先级）：
    1. pip_package 如果是已知映射 → 直接用 PyPI 包名
    2. pip_package 是 git URL → 提取仓库名，尝试 PyPI（如 CLIP→open_clip_torch）
    3. 均失败 → git clone 代理尝试
    """
    entrypoint = proposal.get("proposed_entrypoint", "")
    raw_pkg = proposal.get("pip_package", "")
    result = {"status": "pending", "steps": []}

    # ── 已知映射: GitHub repo → PyPI 包名 ──
    _PKG_MAP = {
        "clip": "open_clip_torch",
        "openai/clip": "open_clip_torch",
        "openai-clip": "open_clip_torch",
        "sam2": "segment_anything",
        "segment-anything": "segment_anything",
        "yolov8": "ultralytics",
        "ultralytics": "ultralytics",
        "monai": "monai",
        "nnunet": "nnunetv2",
    }

    pkg_name = ""
    # Step 0a: 从 pip_package 字段提取（处理空格分隔的多个包名）
    candidates = []
    if raw_pkg:
        raw_lower = raw_pkg.lower()
        # 处理 git URL
        for prefix in ["git+https://github.com/", "https://github.com/", "git+"]:
            raw_lower = raw_lower.replace(prefix, "")
        # 如果有空格，拆分为多个候选
        parts = [p.strip().rstrip("/").split("/")[-1].replace(".git", "").strip() for p in raw_lower.split()]
        for part in parts:
            if part and part not in candidates:
                candidates.append(part)

    for c in candidates:
        if c in _PKG_MAP:
            pkg_name = _PKG_MAP[c]
            break
        elif "/" in c:
            repo = c.split("/")[-1]
            mapped = _PKG_MAP.get(repo.lower(), repo.lower())
            if mapped != repo.lower():
                pkg_name = mapped
                break
    else:
        pkg_name = candidates[0] if candidates else ""

    # Step 0b: 从 entrypoint 推断（fallback）
    if not pkg_name and entrypoint:
        if "github.com" in entrypoint:
            pkg_name = entrypoint.rstrip("/").split("/")[-1].replace(".git", "")
        else:
            pkg_name = Path(entrypoint).stem
        pkg_name = _PKG_MAP.get(pkg_name.lower(), pkg_name)

    if not pkg_name:
        pkg_name = "unknown"
    print(f"[platform_mvp] auto_configure: resolved package name = {pkg_name} (raw={raw_pkg})", file=sys.stderr)

    # Step 1: pip install（逐一尝试所有候选包名）
    pip_candidates = [pkg_name] if pkg_name else []
    for c in candidates:
        if c not in pip_candidates:
            pip_candidates.append(c)
    # 如果 pip_package 是 git URL，也尝试原样安装
    if raw_pkg and raw_pkg.startswith("git+"):
        pip_candidates.append(raw_pkg)

    for candidate in pip_candidates:
        try:
            print(f"[platform_mvp]   trying: pip install {candidate}", file=sys.stderr)
            completed = subprocess.run(
                [dentalclaw_python, "-m", "pip", "install", candidate, "-i", "https://mirrors.aliyun.com/pypi/simple/"],
                capture_output=True, text=True, timeout=120, check=False,
            )
            ok = completed.returncode == 0
            result["steps"].append({
                "step": "pip_install", "ok": ok,
                "package": candidate, "returncode": completed.returncode,
            })
            if ok:
                result["status"] = "configured"
                result["reason"] = f"Successfully installed {candidate} via pip (aliyun mirror)."
                adapter_path = _write_thin_adapter(workspace_dir, pkg_name, entrypoint)
                result["steps"].append({"step": "adapter_generated", "path": str(adapter_path)})
                result["dynamic_registry_entry"] = _make_dynamic_entry(pkg_name, entrypoint, proposal, adapter_path)
                return result
        except Exception as e:
            result["steps"].append({"step": "pip_install", "ok": False, "error": str(e)[:100]})

    # Step 2: git clone 代理（最后尝试）
    repo_dir = workspace_dir / "external_repo"
    for url in [entrypoint,
                f"https://ghproxy.com/{entrypoint}",
                f"https://mirror.ghproxy.com/{entrypoint}"]:
        if not url.startswith("http"):
            continue
        clone_cmd = ["git", "clone", "--depth", "1", url, str(repo_dir)]
        try:
            subprocess.run(clone_cmd, capture_output=True, text=True, timeout=30, check=False)
            if repo_dir.exists() and any(repo_dir.iterdir()):
                result["steps"].append({"step": "git_clone", "ok": True, "url": url})
                break
        except Exception:
            continue
    else:
        result["status"] = "network_error"
        result["reason"] = (
            f"Cannot install {pkg_name}: pip failed. GitHub is not directly accessible from this server. "
            f"Try using a PyPI package name (e.g., 'open_clip_torch' instead of 'CLIP') or provide a Gitee mirror URL."
        )
        return result

    # Step 3: 安装 clone 下来的依赖
    for req_file in ["requirements.txt", "setup.py", "pyproject.toml"]:
        req_path = repo_dir / req_file
        if req_path.exists():
            install_cmd = [dentalclaw_python, "-m", "pip", "install", "-r", str(req_path)]
            if req_file != "requirements.txt":
                install_cmd = [dentalclaw_python, "-m", "pip", "install", "-e", str(repo_dir)]
            completed = subprocess.run(install_cmd, capture_output=True, text=True, timeout=120, check=False)
            result["steps"].append({
                "step": f"pip_install_{req_file}",
                "ok": completed.returncode == 0,
            })
            break

    adapter_path = _write_thin_adapter(workspace_dir, pkg_name, entrypoint)
    result["steps"].append({"step": "adapter_generated", "path": str(adapter_path)})
    result["status"] = "configured"
    result["dynamic_registry_entry"] = _make_dynamic_entry(pkg_name, entrypoint, proposal, adapter_path)
    return result


def _write_thin_adapter(workspace_dir: Path, pkg_name: str, source: str) -> Path:
    adapter_path = workspace_dir / "adapter_runner.py"
    adapter_path.write_text(f'''#!/usr/bin/env python3
"""Auto-generated thin adapter for: {source}
Generated by DentalClaw platform MVP auto-configure at {datetime.now().isoformat()}.
"""
import subprocess, sys, json
PACKAGE = "{pkg_name}"

def main():
    print(json.dumps({{"status": "adapter_ready", "package": PACKAGE, "source": "{source}",
                       "note": "Auto-generated adapter. Import and use {pkg_name} in your pipeline."}}))

if __name__ == "__main__":
    main()
''', encoding="utf-8")
    return adapter_path


def _generate_script(
    proposal: dict[str, Any],
    intent: str,
    parsed: dict[str, Any],
    workspace_dir: Path,
    timeout: int = 60,
) -> Path | None:
    """使用 DeepSeek API 根据 Agent 提议生成完整的可执行 Python 训练/推理脚本。"""
    entrypoint = proposal.get("proposed_entrypoint", "run_pipeline.py")
    pkg = proposal.get("pip_package", "")
    reasoning = proposal.get("reasoning", "")

    # 确定脚本名
    script_name = Path(entrypoint).name if entrypoint else "run_pipeline.py"
    if not script_name.endswith(".py"):
        script_name += ".py"

    prompt = f"""You are a DentalClaw code generator. Write a COMPLETE, RUNNABLE Python script.

=== PROPOSAL ===
Approach: {reasoning}
Required packages: {pkg}
Target filename: {script_name}

=== USER REQUEST ===
{intent}

=== PARSED INTENT ===
dataset={parsed['dataset']}, task={parsed['task_family']}, modality={parsed['modality']}, mode={parsed['mode']}

=== CODEBASE CONTEXT (use these paths and assets) ===
- REPO_ROOT: {REPO_ROOT}
- TDD dataset (2D panoramic X-ray): {REPO_ROOT}/artifacts/datasets/nnUNet/nnUNet_raw/Dataset501_TDDTeethBinary2D
  Images: imagesTr/, Labels: labelsTr/, Test: imagesTs/
- ToothFairy3 (3D CBCT): {REPO_ROOT.parent}/JoD/ToothFairy3_LPS (imagesTr/, labelsTr/)
- Private 2D data: {REPO_ROOT}/data/private01 (images/, masks/)
- Existing checkpoint: {REPO_ROOT}/artifacts/training_runs/trial_501_binary_baseline/best_model/checkpoint_best.pth
- YOLO models: {REPO_ROOT}/yolov8n.pt, {REPO_ROOT}/yolo26n.pt
- nnUNet results: {REPO_ROOT}/artifacts/training_runs/trial_501_binary_baseline/best_inference/
- Output dir convention: use Path(args.output) or sys.argv[2] as output directory

=== REQUIREMENTS ===
1. The script MUST be a complete, self-contained .py file that runs with `python {script_name} <input_image> <output_dir>`
2. Include all imports, model loading, inference/training logic
3. Use the installed packages: {pkg}
4. Handle errors gracefully (print JSON summary at the end)
5. Support both CPU and CUDA
6. For training: use the TDD dataset path above, implement a simple training loop with validation
7. For inference: load the image, run prediction, save mask to output_dir
8. Output a JSON summary file at the end with keys: status, mask_path (or model_path), metrics

=== OUTPUT ===
Return ONLY the Python code. No markdown, no explanation. Start with `#!/usr/bin/env python3`."""

    try:
        import urllib.request
        api_key = _load_deepseek_api_key()
        if not api_key:
            return None

        data = json.dumps({
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "system", "content": "You are an expert Python developer. Write clean, runnable code only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 4096,
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.deepseek.com/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            code = body.get("choices", [{}])[0].get("message", {}).get("content", "")

        if not code:
            print("[platform_mvp] Script generation returned empty content", file=sys.stderr)
            return None

        # Clean up markdown fences
        code = code.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        script_path = workspace_dir / script_name
        script_path.write_text(code, encoding="utf-8")
        print(f"[platform_mvp] Generated script: {script_path} ({len(code)} chars)", file=sys.stderr)
        return script_path
    except Exception as exc:
        print(f"[platform_mvp] Script generation failed: {exc}", file=sys.stderr)
        return None


def _make_dynamic_entry(pkg_name: str, source: str, proposal: dict[str, Any], adapter_path: Path) -> dict[str, Any]:
    return {
        "id": f"external_{pkg_name}",
        "display_name": f"External: {pkg_name}",
        "status": "external_configured",
        "entrypoint": str(adapter_path),
        "source": source,
        "pip_package": pkg_name,
        "confidence": proposal.get("confidence", 0),
        "reasoning": proposal.get("reasoning", ""),
        "risks": proposal.get("risks", []),
    }


def _agent_propose(intent: str, parsed: dict[str, Any], timeout: int = 60) -> dict[str, Any] | None:
    """当静态方法表无法匹配时，调用 OpenClaw Agent + web search 自主提议解决方案。
    
    优先级: OpenClaw Agent → DeepSeek API 直调 (fallback)
    """
    # 1. Web search (改进的 query)
    task = parsed.get("task_family", "dental")
    modality = parsed.get("modality", "2d")
    dataset = parsed.get("dataset", "unknown")
    query = f"{task} {modality} {dataset} model GitHub dental"
    search_results = _web_search(query, max_results=5)
    if search_results:
        search_text = "\n".join(
            f"  [{i+1}] {r.get('title', '?')}\n      {r.get('href', '')}"
            for i, r in enumerate(search_results[:3])
        )
        print(f"[platform_mvp] web search: found {len(search_results)} results", file=sys.stderr)
    else:
        search_text = "(no web results — install duckduckgo-search or ddgs for better results)"
        print("[platform_mvp] web search: no results (try: pip install duckduckgo-search)", file=sys.stderr)

    # 2. 构造 prompt（包含代码库实际资产信息，防止 Agent 凭空编造）
    prompt = (
        f"You are the DentalClaw platform planner. No pre-approved route exists for this request.\n\n"
        f"=== AVAILABLE CODEBASE ASSETS (use these when relevant) ===\n"
        f"- nnU-Net v2 framework (conda env nnunetv2, package: nnunetv2)\n"
        f"- TDD 2D binary segmentation: mvp_fullflow/run_mvp_fullflow.py (inference + report)\n"
        f"- Auto-train: agents/experimentation/skills/tooth_autotrain_nnunet/ (2D/3D nnUNet training)\n"
        f"- Super-resolution: platform_mvp/run_super_resolution_mvp.py\n"
        f"- Anomaly detection: platform_mvp/run_anomaly_detection_mvp.py (ResNet+IsolationForest)\n"
        f"- YOLO detection: run_tdd_detection_traced.py, yolo26n.pt, yolov8n.pt\n"
        f"- TTA + Ensemble: platform_mvp/tta_ensemble.py (test-time augmentation + multi-model majority vote)\n"
        f"- TDD dataset: Dataset501_TDDTeethBinary2D (panoramic X-ray, binary tooth masks)\n"
        f"- ToothFairy3 dataset: 3D CBCT volumes\n"
        f"- Existing models: nnUNet checkpoint for TDD binary segmentation\n\n"
        f"=== WEB SEARCH RESULTS ===\n{search_text}\n\n"
        f"=== USER REQUEST ===\n{intent}\n\n"
        f"=== PARSED INTENT ===\n"
        f"task={task}, dataset={dataset}, modality={modality}, mode={parsed.get('mode')}\n\n"
        f"=== INSTRUCTIONS ===\n"
        f"Propose the MOST SPECIFIC solution using existing codebase assets FIRST.\n"
        f"Only suggest external packages if no internal asset fits.\n"
        f"proposed_entrypoint: an EXISTING script path in the project, or a reasonable new script name.\n"
        f"pip_package: space-separated pip packages if external deps are needed, or empty string.\n"
        f"confidence: 0.0-1.0 based on how well the solution matches.\n"
        f"reasoning: 1-2 sentences explaining the approach.\n\n"
        f"Return ONLY valid JSON (no markdown, no code fences):\n"
        f'{{"proposed_entrypoint": "...", "pip_package": "...", "confidence": 0.0, "reasoning": "..."}}'
    )

    proposal = None

    # 3a. 优先尝试 OpenClaw Agent
    if _call_ocl is not None:
        try:
            print(f"[platform_mvp] calling OpenClaw agent (timeout={timeout}s)...", file=sys.stderr)
            openclaw_result = _call_ocl(
                prompt=prompt, model=None, timeout=min(timeout, 30), dry_run=True, thinking="low",
            )
            ocl_status = openclaw_result.get("status", "unknown") if openclaw_result else "None"
            if openclaw_result and ocl_status not in ("error", "timeout"):
                inner = openclaw_result.get("result", {})
                payloads = inner.get("payloads", [])
                if payloads and isinstance(payloads[0], dict) and payloads[0].get("text"):
                    try:
                        proposal = json.loads(payloads[0]["text"])
                        print(f"[platform_mvp] OpenClaw proposal: confidence={proposal.get('confidence')}, "
                              f"package={proposal.get('pip_package', 'N/A')}", file=sys.stderr)
                    except json.JSONDecodeError:
                        pass
            if proposal is None:
                err = openclaw_result.get("error", "") if openclaw_result else ""
                print(f"[platform_mvp] OpenClaw agent failed (status={ocl_status}), trying DeepSeek API...", file=sys.stderr)
        except Exception as exc:
            print(f"[platform_mvp] OpenClaw agent exception: {exc}, trying DeepSeek API...", file=sys.stderr)
    else:
        print("[platform_mvp] OpenClaw runner not available, using DeepSeek API directly", file=sys.stderr)

    # 3b. Fallback: DeepSeek API 直调
    if proposal is None:
        print(f"[platform_mvp] calling DeepSeek API directly (timeout={timeout}s)...", file=sys.stderr)
        proposal = _call_deepseek_api(prompt, timeout=timeout)
        if proposal:
            print(f"[platform_mvp] DeepSeek API proposal: confidence={proposal.get('confidence')}, "
                  f"package={proposal.get('pip_package', 'N/A')}", file=sys.stderr)
        else:
            print("[platform_mvp] DeepSeek API also failed — no agent proposal available", file=sys.stderr)

    return proposal


def _academic_search(parsed: dict[str, Any], max_results: int = 8) -> list[dict[str, str]]:
    """学术搜索引擎：Europe PMC（医学）→ arXiv（CS/ML），无需 API key。

    返回统一的 [{"title": ..., "href": ..., "source": "arxiv"|"pmc", "year": ...}] 格式。
    """
    task = parsed.get("task_family", "medical")
    dataset = parsed.get("dataset", "dental")
    mode = parsed.get("mode", "inference")

    dataset_en = {"TDD": "panoramic radiograph", "ToothFairy3": "CBCT 3D",
                  "Private2D": "dental x-ray", "unknown": "dental imaging"}
    task_en = {"segmentation": "tooth segmentation", "detection": "tooth detection",
               "classification": "dental classification", "super_resolution": "dental super resolution",
               "anomaly_detection": "dental anomaly detection",
               "unknown": "dental deep learning"}
    ds = dataset_en.get(dataset, "dental")
    ts = task_en.get(task, "dental")
    query = f"{ts} {ds} {mode}"

    results: list[dict[str, str]] = []
    seen_urls = set()

    # ── Source 1: Europe PMC (medical focus, JSON API, free) ──
    try:
        import urllib.request, urllib.parse
        q = urllib.parse.quote(query)
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={q}&resultType=core&pageSize={max_results}&format=json"
        req = urllib.request.Request(url, headers={"User-Agent": "DentalClaw/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        for r in data.get("resultList", {}).get("result", [])[:max_results]:
            title = r.get("title", "").strip()
            doi = r.get("doi", "")
            href = f"https://doi.org/{doi}" if doi else f"https://europepmc.org/article/MED/{r.get('id', '')}"
            if title and href not in seen_urls:
                seen_urls.add(href)
                results.append({"title": title, "href": href, "source": "pmc",
                               "year": str(r.get("pubYear", ""))})
        if results:
            print(f"[platform_mvp] Europe PMC: {len(results)} results", file=sys.stderr)
    except Exception as e:
        print(f"[platform_mvp] Europe PMC unavailable: {e}", file=sys.stderr)

    # ── Source 2: arXiv API (CS/ML focus, XML, free) ──
    if len(results) < max_results:
        try:
            import urllib.request, urllib.parse
            import xml.etree.ElementTree as ET
            q = urllib.parse.quote(query)
            url = f"http://export.arxiv.org/api/query?search_query=all:{q}&max_results={max_results - len(results)}"
            req = urllib.request.Request(url, headers={"User-Agent": "DentalClaw/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                xml_data = resp.read().decode("utf-8")
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(xml_data)
            for entry in root.findall("atom:entry", ns):
                title_el = entry.find("atom:title", ns)
                link_el = entry.find("atom:id", ns)
                if title_el is not None:
                    title = title_el.text.strip().replace("\n", " ")
                    href = link_el.text.strip() if link_el is not None else ""
                    if href not in seen_urls:
                        seen_urls.add(href)
                        results.append({"title": title, "href": href, "source": "arxiv", "year": ""})
            arxiv_count = len([r for r in results if r.get("source") == "arxiv"])
            if arxiv_count > 0:
                print(f"[platform_mvp] arXiv: {arxiv_count} results", file=sys.stderr)
        except Exception as e:
            print(f"[platform_mvp] arXiv unavailable: {e}", file=sys.stderr)

    return results


def _do_web_search(intent: str, parsed: dict[str, Any]) -> list[dict[str, str]]:
    """为当前意图执行搜索：学术引擎（Europe PMC + arXiv）优先，DuckDuckGo fallback。

    返回统一的 [{"title": ..., "href": ..., "source": "pmc"|"arxiv"|"web", "year": ...}] 格式。
    """
    all_results: list[dict[str, str]] = []

    # ── 主搜索: 学术引擎 ──
    academic = _academic_search(parsed, max_results=8)
    all_results.extend(academic)

    # ── Fallback: DuckDuckGo / Bing（补充非学术资源，如 GitHub 仓库） ──
    if len(all_results) < 3:
        task = parsed.get("task_family", "dental")
        dataset = parsed.get("dataset", "unknown")
        mode = parsed.get("mode", "inference")
        web_query = f"{task} {dataset} {mode} medical imaging GitHub"
        web_results = _web_search(web_query, max_results=3)
        for r in web_results:
            if r.get("href"):
                r["source"] = "web"
                all_results.append(r)

    if all_results:
        sources = set(r.get("source", "?") for r in all_results)
        print(f"[platform_mvp] search: {len(all_results)} results from {sources}", file=sys.stderr)
    else:
        print("[platform_mvp] search: no results", file=sys.stderr)

    return all_results


def _agent_decide(
    *,
    intent: str,
    parsed: dict[str, Any],
    search_results: list[dict[str, str]],
    registry_match: dict[str, Any] | None,
    registry_reasons: list[str],
    lang: str = "en",
    timeout: int = 60,
) -> dict[str, Any] | None:
    """Agent 综合决策：web search 结果 + 离线方法表匹配 + 代码库资产 → 最终方案。

    返回 dict:
      - decision: "use_registry" | "external_proposal" | "reject"
      - 如果是 external_proposal: 还包含 proposed_entrypoint, pip_package, confidence, reasoning
    """
    task = parsed.get("task_family", "dental")
    modality = parsed.get("modality", "2d")
    dataset = parsed.get("dataset", "unknown")

    # 构建 web search 结果文本
    if search_results:
        web_text = "\n".join(
            f"  [{i+1}] {r.get('title', '?')}\n      {r.get('href', '')}"
            for i, r in enumerate(search_results[:5])
        )
    else:
        web_text = "(no web search results)"

    # 构建方法表匹配信息
    if registry_match:
        registry_text = (
            f"REGISTRY MATCH FOUND:\n"
            f"  id: {registry_match['id']}\n"
            f"  display: {registry_match['display_name']}\n"
            f"  status: {registry_match.get('status', 'unknown')}\n"
            f"  entrypoint: {registry_match.get('entrypoint', 'N/A')}\n"
            f"  framework: {registry_match.get('framework', 'N/A')}\n"
            f"  allowed_modes: {registry_match.get('allowed_modes', [])}\n"
            f"  limitations: {registry_match.get('limitations', [])}\n"
        )
    else:
        registry_text = "NO REGISTRY MATCH. You must propose a solution from web search + codebase knowledge."

    prompt = _(lang, "agent_user_prompt_full",
        intent=intent,
        task=task, dataset=dataset, modality=modality,
        mode=parsed.get("mode"),
        web_text=web_text,
        registry_text=registry_text,
    )

    proposal = None

    # OpenClaw Agent (--local 模式，DeepSeek V4 Pro，约需 25s)
    if _call_ocl is not None:
        try:
            ocl_timeout = min(timeout, 30)
            print(f"[platform_mvp] Agent deciding (OpenClaw, timeout={ocl_timeout}s)...", file=sys.stderr)
            ocl_result = _call_ocl(prompt=prompt, model=None, timeout=ocl_timeout, dry_run=True, thinking="low")
            ocl_status = ocl_result.get("status", "unknown") if ocl_result else "None"
            if ocl_result and ocl_status not in ("error", "timeout"):
                inner = ocl_result.get("result", {})
                payloads = inner.get("payloads", [])
                if payloads and isinstance(payloads[0], dict) and payloads[0].get("text"):
                    try:
                        proposal = json.loads(payloads[0]["text"])
                    except json.JSONDecodeError:
                        pass
            if proposal is None:
                print(f"[platform_mvp] OpenClaw unavailable, using DeepSeek API...", file=sys.stderr)
        except Exception as exc:
            print(f"[platform_mvp] OpenClaw exception: {exc}, trying DeepSeek API...", file=sys.stderr)

    # Fallback: DeepSeek API direct call
    if proposal is None:
        print(f"[platform_mvp] Agent deciding (DeepSeek API, timeout={timeout}s)...", file=sys.stderr)
        proposal = _call_deepseek_api(prompt, timeout=timeout)

    if proposal:
        decision = proposal.get("decision", "reject")
        print(f"[platform_mvp] Agent decision: {decision}, confidence={proposal.get('confidence', '?')}", file=sys.stderr)
    else:
        print("[platform_mvp] Agent decision failed — no Agent available", file=sys.stderr)

    return proposal


def _strip_ansi(text: str) -> str:
    import re as _re
    return _re.sub(r"\x1b\[[0-9;]*m", "", text)


def build_plan(
    *,
    intent: str,
    registry_path: Path,
    case_id: str,
    run_dir: Path,
    reuse_fullflow_run: Path | None,
    trace: "TraceRecorder | None" = None,
    lang: str = "auto",
) -> dict[str, Any]:
    """构建平台计划。核心流程：web search 驱动 → Agent 决策 → 方法表作为缓存。

    彭老师要求：反对写死逻辑，Agent 自主搜索 + 调用基线模型，举一反三。
    因此 web search 前置，Agent 综合评估 web 结果与离线方法表后做最终决策。
    """
    lang = detect_language(intent, lang)  # 解析 auto → zh/en
    registry = load_registry(registry_path)
    parsed = parse_intent(intent)
    if trace:
        trace.step("planner", "intent.parse",
                   input_summary=intent[:120],
                   output_summary=f"dataset={parsed['dataset']}, task={parsed['task_family']}, mode={parsed['mode']}",
                   detail=dict(parsed))

    # ── Step 0: 关键信息缺失？先判断是否完全超出牙科 CV 范围 ──
    missing = []
    if parsed["dataset"] == "unknown":
        missing.append(_(lang, "clarification_field_dataset"))
    if parsed["task_family"] == "unknown":
        missing.append(_(lang, "clarification_field_task"))

    if missing:
        # 数据集 AND 任务都未知 → 完全超出牙科 CV 范围，直接拒绝
        if parsed["dataset"] == "unknown" and parsed["task_family"] == "unknown":
            return {
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "platform_mvp_version": registry.get("version"),
                "intent": parsed,
                "supported": False, "executable": False,
                "selected_method": None,
                "selection_reasons": [],
                "workflow": ["intent.parse", "platform.out_of_scope"],
                "execution": {
                    "requested_case_id": case_id,
                    "platform_run_dir": _rel(run_dir),
                    "reuse_fullflow_run": _rel(reuse_fullflow_run),
                    "will_execute": False, "delegate_command": None,
                },
                "reason": _(lang, "out_of_scope_reject"),
                "lang": lang,
            }

        # 仅部分信息缺失 → 友好追问
        return {
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "platform_mvp_version": registry.get("version"),
            "intent": parsed,
            "supported": True, "executable": False,
            "selected_method": {
                "id": "platform_clarification",
                "display_name": _(lang, "awaiting_clarification_status"),
                "status": "awaiting_clarification",
                "missing_fields": missing,
                "available_datasets": _(lang, "available_datasets"),
                "available_tasks": _(lang, "available_tasks"),
            },
            "selection_reasons": [],
            "workflow": ["intent.parse", "platform.request_clarification"],
            "execution": {
                "requested_case_id": case_id,
                "platform_run_dir": _rel(run_dir),
                "reuse_fullflow_run": _rel(reuse_fullflow_run),
                "will_execute": False, "delegate_command": None,
            },
            "reason": _(lang, "clarification_missing", fields=', '.join(missing)),
            "lang": lang,
        }

    # ── Step 1: Web search (彭老师要求：以 web search 为主) ──
    web_search_results = _do_web_search(intent, parsed)
    if trace:
        trace.step("planner", "web.search",
                   input_summary=f"task={parsed['task_family']}, dataset={parsed['dataset']}",
                   output_summary=f"{len(web_search_results)} results found")

    # ── Step 2: 离线方法表查询（作为已验证方案的缓存） ──
    registry_match, registry_reasons = select_method(parsed, registry)
    if trace:
        trace.step("planner", "registry.method_lookup",
                   input_summary=f"dataset={parsed['dataset']}, task={parsed['task_family']}",
                   output_summary=f"match={'found' if registry_match else 'none'}, reasons={registry_reasons[0][:80] if registry_reasons else 'none'}")

    # ── Step 3: Agent 综合决策 (web 结果 + 方法表匹配 + 代码库资产) ──
    agent_decision = _agent_decide(
        intent=intent, parsed=parsed,
        search_results=web_search_results,
        registry_match=registry_match,
        registry_reasons=registry_reasons,
        lang=lang,
    )
    if trace and agent_decision:
        trace.step("planner", "agent.decide",
                   input_summary=f"registry_match={'yes' if registry_match else 'no'}",
                   output_summary=f"decision={agent_decision.get('decision')}, confidence={agent_decision.get('confidence')}",
                   decision=agent_decision.get('reasoning', '')[:200],
                   detail=dict(agent_decision))

    if agent_decision is None:
        # Agent 完全不可用 → 回退到纯方法表
        if registry_match:
            selected_method = registry_match
            selection_reasons = registry_reasons
            supported, executable = True, selected_method.get("status") == "executable"
            workflow = selected_method.get("workflow", [])
            selected_payload = {
                "id": selected_method["id"],
                "display_name": selected_method["display_name"],
                "status": selected_method["status"],
                "framework": selected_method.get("framework"),
                "entrypoint": selected_method.get("entrypoint"),
                "expected_outputs": selected_method.get("expected_outputs", []),
                "limitations": selected_method.get("limitations", []),
            }
            reason = "Agent 不可用，回退到离线方法表匹配。"
        else:
            supported, executable = False, False
            workflow = ["intent.parse", "registry.method_lookup", "agent.unavailable", "platform.reject_or_explain"]
            selected_payload = None
            selection_reasons = ["No method in the offline registry matched, and Agent is unavailable."]
            reason = "离线方法表未命中，且 Agent 不可用。请检查 OpenClaw 网关或 DeepSeek API。"
    else:
        # Agent 给了决策
        decision_type = agent_decision.get("decision", "reject")
        if decision_type == "use_registry":
            selected_method = registry_match
            selection_reasons = [f"Agent confirms registry match: {registry_match['id']}"]
            supported, executable = True, selected_method.get("status") == "executable"
            workflow = selected_method.get("workflow", [])
            selected_payload = {
                "id": selected_method["id"],
                "display_name": selected_method["display_name"],
                "status": selected_method["status"],
                "framework": selected_method.get("framework"),
                "entrypoint": selected_method.get("entrypoint"),
                "expected_outputs": selected_method.get("expected_outputs", []),
                "limitations": selected_method.get("limitations", []),
                "agent_note": agent_decision.get("reasoning", ""),
            }
            reason = f"Agent 确认离线方法表方案: {selected_method['display_name']}。{agent_decision.get('reasoning', '')}"
        elif decision_type == "external_proposal":
            supported, executable = True, False
            workflow = ["intent.parse", "web.search", "agent.external_proposal"]
            proposed_entrypoint = agent_decision.get("proposed_entrypoint", "unknown")
            selected_payload = {
                "id": "agent_external_suggestion",
                "display_name": f"Agent proposal: {proposed_entrypoint}",
                "status": "external_suggestion",
                "confidence": agent_decision.get("confidence", 0.5),
                "reasoning": agent_decision.get("reasoning", ""),
                "missing_prerequisites": agent_decision.get("missing_prerequisites", []),
                "risks": agent_decision.get("risks", []),
                "entrypoint": proposed_entrypoint,
                "pip_package": agent_decision.get("pip_package"),
                "web_sources": [r.get("href", "") for r in web_search_results[:3]],
            }
            reason = (
                f"Agent 综合 web search 结果与代码库资产后提议了外部方案"
                f"（置信度 {agent_decision.get('confidence', '?')}）。该方案需人工确认后方可执行。"
            )
            selection_reasons = [f"Agent selected external over registry: {registry_match['id']}"
                                if registry_match else "No registry match; Agent used web search."]

            # ── 自动注册: 记录 Agent 发现的代码库资产 ──
            if proposed_entrypoint and proposed_entrypoint.startswith(("agents/", "run_", "platform_mvp/")):
                discovery_count = _track_agent_discovery(
                    entrypoint=proposed_entrypoint,
                    display_name=agent_decision.get("reasoning", proposed_entrypoint)[:80],
                    intent=intent,
                    registry_path=registry_path,
                )
                if discovery_count >= _AUTO_REGISTER_THRESHOLD:
                    selected_payload["auto_registered"] = True
                    selected_payload["discovery_count"] = discovery_count
        else:  # reject
            supported, executable = False, False
            workflow = ["intent.parse", "web.search", "agent.reject"]
            selected_payload = None
            selection_reasons = ["Agent reviewed web search results and registry, found no viable route."]
            reason = agent_decision.get("reasoning", "Agent 评估后认为当前无可执行方案。")

    final_registry_match = registry_match  # for existing_assets lookup
    fullflow_run = reuse_fullflow_run
    if fullflow_run is None and final_registry_match:
        assets = final_registry_match.get("existing_assets", {})
        if assets.get("successful_fullflow_run"):
            fullflow_run = REPO_ROOT / assets["successful_fullflow_run"]

    plan = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "platform_mvp_version": registry.get("version"),
        "intent": parsed,
        "supported": supported,
        "executable": executable,
        "selected_method": selected_payload,
        "selection_reasons": selection_reasons,
        "workflow": workflow,
        "execution": {
            "requested_case_id": case_id,
            "platform_run_dir": _rel(run_dir),
            "reuse_fullflow_run": _rel(fullflow_run),
            "will_execute": False,
            "delegate_command": None,
        },
        "reason": reason,
        "lang": lang,
        "web_search": {
            "performed": True,
            "query": web_search_results[0].get("query", "") if web_search_results else "",
            "result_count": len(web_search_results),
        },
    }
    return plan


def _format_list(values: list[Any]) -> list[str]:
    if not values:
        return ["- None"]
    return [f"- `{value}`" for value in values]


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_plan_md(plan: dict[str, Any], execution_result: dict[str, Any] | None = None) -> str:
    selected = plan.get("selected_method") or {}
    lines = [
        "# DentalClaw 平台底座 MVP 运行摘要",
        "",
        "## 1. 用户输入",
        "",
        f"> {plan['intent']['raw_intent']}",
        "",
        "## 2. 解析结果",
        "",
        f"- Dataset: `{plan['intent']['dataset']}`",
        f"- Task family: `{plan['intent']['task_family']}`",
        f"- Modality: `{plan['intent']['modality']}`",
        f"- Mode: `{plan['intent']['mode']}`",
        "",
        "## 3. 离线方法表选择",
        "",
        f"- Supported by registry: `{plan['supported']}`",
        f"- Executable now: `{plan['executable']}`",
        f"- Selected method: `{selected.get('id', 'n/a')}`",
        f"- Method name: `{selected.get('display_name', 'n/a')}`",
        f"- Framework: `{selected.get('framework', 'n/a')}`",
        f"- Entrypoint: `{selected.get('entrypoint', 'n/a')}`",
        "",
        "选择依据：",
        "",
    ]
    lines.extend(_format_list(plan.get("selection_reasons", [])))
    lines += [
        "",
        "## 4. 平台工作流",
        "",
    ]
    for idx, step in enumerate(plan.get("workflow", []), start=1):
        lines.append(f"{idx}. `{step}`")

    lines += [
        "",
        "## 5. 预期产物",
        "",
    ]
    lines.extend(_format_list(selected.get("expected_outputs", [])))

    lines += [
        "",
        "## 6. 当前边界",
        "",
    ]
    lines.extend(_format_list(selected.get("limitations", [])))

    if execution_result:
        lines += [
            "",
            "## 7. 实际执行结果",
            "",
            f"- Status: `{execution_result.get('status')}`",
            f"- Delegate run directory: `{execution_result.get('delegate_run_dir')}`",
            f"- Delegate manifest: `{execution_result.get('delegate_manifest')}`",
            f"- Delegate summary: `{execution_result.get('delegate_summary')}`",
        ]
        if execution_result.get("report_html"):
            lines.append(f"- Report HTML: `{execution_result.get('report_html')}`")
        if execution_result.get("report_overlay"):
            lines.append(f"- Overlay: `{execution_result.get('report_overlay')}`")
        lines += [
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ]
        metrics = execution_result.get("metrics") or {}
        for key in metrics:
            lines.append(f"| {key} | {_format_metric(metrics.get(key))} |")
    else:
        lines += [
            "",
            "## 7. 当前执行状态",
            "",
            "- Status: `plan_only`",
            "- Reason: `本次只生成平台计划，未请求执行 adapter。`",
        ]

    lines += [
        "",
        "## 8. 汇报口径",
        "",
        "这个 MVP 不是 benchmark 主线，而是平台底座主线：先把一句话输入、离线方法选择、既有代码入口调用、结果证据收集串起来。当前已接通 TDD 2D 分割推理与报告路线；私有数据训练、3D 分割、异常检测和超分作为同一 registry 机制下的后续 adapter 接入。",
    ]
    return "\n".join(lines) + "\n"


def _run_subprocess(command: list[str], stdout_path: Path, stderr_path: Path) -> subprocess.CompletedProcess:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        return subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            stdout=stdout_file,
            stderr=stderr_file,
            check=False,
        )


def execute_selected_route(
    *,
    plan: dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
    trace: "TraceRecorder | None" = None,
) -> dict[str, Any]:
    selected = plan.get("selected_method") or {}
    if trace:
        trace.step("clinician", "platform.collect_evidence",
                   input_summary=f"method={selected.get('id', 'unknown')}, execute={args.execute}",
                   output_summary="execution started",
                   status="running")

    # ── Agent 外部提议 ──
    # ── TTA + Ensemble ──
    if selected.get("id") == "tta_ensemble_inference" or plan["intent"].get("tta") or plan["intent"].get("ensemble"):
        tta_dir = run_dir / "tta_ensemble"
        tta_dir.mkdir(parents=True, exist_ok=True)
        image_path = str(
            REPO_ROOT / "artifacts/datasets/nnUNet/nnUNet_raw/Dataset501_TDDTeethBinary2D/imagesTs"
            / f"{args.case_id}_0000.png"
        )
        model_folders = selected.get("existing_assets", {}).get("model_folders", [
            "artifacts/training_runs/trial_501_binary_baseline/exp_001/nnUNet_results_snapshot/Dataset501_TDDTeethBinary2D/nnUNetTrainer__nnUNetPlans__2d"
        ])
        use_tta = plan["intent"].get("tta", True)
        checkpoint = selected.get("existing_assets", {}).get("default_checkpoint", "checkpoint_best.pth")

        tta_cmd = [
            args.dentalclaw_python,
            str(REPO_ROOT / "platform_mvp/tta_ensemble.py"),
            "--image", image_path,
            "--models"] + [str(REPO_ROOT / m) for m in model_folders] + [
            "--output", str(tta_dir),
            "--checkpoint", checkpoint,
            "--case-id", args.case_id,
        ]
        if use_tta:
            tta_cmd.append("--tta")

        stdout_path = run_dir / "logs/tta_ensemble_stdout.log"
        stderr_path = run_dir / "logs/tta_ensemble_stderr.log"
        completed = _run_subprocess(tta_cmd, stdout_path, stderr_path)

        summary_path = tta_dir / "summary.json"
        if completed.returncode != 0:
            return {
                "status": "failed",
                "returncode": completed.returncode,
                "command": tta_cmd,
                "stdout": _rel(stdout_path),
                "stderr": _rel(stderr_path),
            }
        summary = _read_json(summary_path) if summary_path.exists() else {}
        return {
            "status": "completed",
            "delegate_run_dir": _rel(tta_dir),
            "delegate_summary": _rel(summary_path),
            "metrics": {
                "mode": summary.get("mode"),
                "num_models": summary.get("num_models"),
                "tta": summary.get("tta"),
                "elapsed_seconds": summary.get("elapsed_seconds"),
            },
        }

    if selected.get("status") == "external_suggestion":
        if not getattr(args, "allow_external", False):
            return {
                "status": "awaiting_confirmation",
                "reason": "Agent proposed an external solution. Review the proposal and re-run with --allow-external to auto-configure.",
                "proposal": {
                    "entrypoint": selected.get("entrypoint"),
                    "confidence": selected.get("confidence"),
                    "reasoning": selected.get("reasoning"),
                    "risks": selected.get("risks", []),
                    "missing_prerequisites": selected.get("missing_prerequisites", []),
                },
            }
        # --allow-external: 自动 clone + install + 生成 adapter
        proposal = {
            "proposed_entrypoint": selected.get("entrypoint"),
            "pip_package": selected.get("pip_package", ""),
            "confidence": selected.get("confidence"),
            "reasoning": selected.get("reasoning"),
            "risks": selected.get("risks", []),
            "missing_prerequisites": selected.get("missing_prerequisites", []),
        }
        workspace = run_dir / "external_setup"
        workspace.mkdir(parents=True, exist_ok=True)
        config_result = _auto_configure(proposal, workspace, args.dentalclaw_python)

        if config_result.get("status") != "configured":
            return {
                "status": config_result.get("status", "failed"),
                **config_result,
                "note": "External dependency installation failed. Cannot proceed.",
            }

        # Step 2: 用 Agent 生成可执行脚本
        print("[platform_mvp] Generating executable script from proposal...", file=sys.stderr)
        script_path = _generate_script(
            proposal=proposal,
            intent=plan["intent"]["raw_intent"],
            parsed=plan["intent"],
            workspace_dir=workspace,
        )

        if script_path is None:
            return {
                "status": "configured_no_script",
                **config_result,
                "note": "Dependencies installed, but script generation failed. Review the proposal and write the script manually.",
            }

        # Step 3: 执行生成的脚本
        if not args.execute:
            return {
                "status": "configured_ready",
                **config_result,
                "script_path": str(script_path),
                "note": f"External solution ready. Script generated at {script_path}. Add --execute to run it.",
            }

        print(f"[platform_mvp] Executing generated script: {script_path}", file=sys.stderr)
        script_stdout = run_dir / "logs/external_script_stdout.log"
        script_stderr = run_dir / "logs/external_script_stderr.log"
        # Use the first test image as default input
        test_image = str(REPO_ROOT / "artifacts/datasets/nnUNet/nnUNet_raw/Dataset501_TDDTeethBinary2D/imagesTs/100_0000.png")
        exec_cmd = [args.dentalclaw_python, str(script_path), test_image, str(workspace / "output")]
        completed = _run_subprocess(exec_cmd, script_stdout, script_stderr)

        return {
            "status": "completed" if completed.returncode == 0 else "script_failed",
            "returncode": completed.returncode,
            "command": exec_cmd,
            "stdout": _rel(script_stdout),
            "stderr": _rel(script_stderr),
            "script_path": str(script_path),
            "config": config_result,
            "note": "External solution: pip install → script generation → execution.",
        }

    
    # ── TTA + Ensemble ──
    if selected.get("id") == "tta_ensemble_inference" or plan["intent"].get("tta") or plan["intent"].get("ensemble"):
        tta_dir = run_dir / "tta_ensemble"
        tta_dir.mkdir(parents=True, exist_ok=True)
        image_path = str(
            REPO_ROOT / "artifacts/datasets/nnUNet/nnUNet_raw/Dataset501_TDDTeethBinary2D/imagesTs"
            / f"{args.case_id}_0000.png"
        )
        model_folders = selected.get("existing_assets", {}).get("model_folders", [
            "artifacts/training_runs/trial_501_binary_baseline/exp_001/nnUNet_results_snapshot/Dataset501_TDDTeethBinary2D/nnUNetTrainer__nnUNetPlans__2d"
        ])
        use_tta = plan["intent"].get("tta", True)
        checkpoint = selected.get("existing_assets", {}).get("default_checkpoint", "checkpoint_best.pth")

        tta_cmd = [
            args.dentalclaw_python,
            str(REPO_ROOT / "platform_mvp/tta_ensemble.py"),
            "--image", image_path,
            "--models"] + [str(REPO_ROOT / m) for m in model_folders] + [
            "--output", str(tta_dir),
            "--checkpoint", checkpoint,
            "--case-id", args.case_id,
        ]
        if use_tta:
            tta_cmd.append("--tta")

        stdout_path = run_dir / "logs/tta_ensemble_stdout.log"
        stderr_path = run_dir / "logs/tta_ensemble_stderr.log"
        completed = _run_subprocess(tta_cmd, stdout_path, stderr_path)

        summary_path = tta_dir / "summary.json"
        if completed.returncode != 0:
            return {
                "status": "failed",
                "returncode": completed.returncode,
                "command": tta_cmd,
                "stdout": _rel(stdout_path),
                "stderr": _rel(stderr_path),
            }
        summary = _read_json(summary_path) if summary_path.exists() else {}
        return {
            "status": "completed",
            "delegate_run_dir": _rel(tta_dir),
            "delegate_summary": _rel(summary_path),
            "metrics": {
                "mode": summary.get("mode"),
                "num_models": summary.get("num_models"),
                "tta": summary.get("tta"),
                "elapsed_seconds": summary.get("elapsed_seconds"),
            },
        }

    if selected.get("id") == "dental_2d_super_resolution":
        sr_dir = run_dir / "super_resolution"
        command = [
            sys.executable,
            str(REPO_ROOT / "platform_mvp/run_super_resolution_mvp.py"),
            "--out-dir",
            str(sr_dir),
            "--case-ids",
            args.case_id,
            "--scale",
            str(args.super_resolution_scale),
        ]
        stdout_path = run_dir / "logs/super_resolution_stdout.log"
        stderr_path = run_dir / "logs/super_resolution_stderr.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_file:
            completed = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                env=env,
                text=True,
                stdout=stdout_file,
                stderr=stderr_file,
                check=False,
            )
        if completed.returncode != 0:
            return {
                "status": "failed",
                "returncode": completed.returncode,
                "command": command,
                "stdout": _rel(stdout_path),
                "stderr": _rel(stderr_path),
            }
        summary_path = sr_dir / "super_resolution_summary.json"
        summary = _read_json(summary_path) if summary_path.exists() else {}
        return {
            "status": "completed",
            "command": command,
            "stdout": _rel(stdout_path),
            "stderr": _rel(stderr_path),
            "delegate_run_dir": _rel(sr_dir),
            "delegate_manifest": _rel(summary_path),
            "delegate_summary": _rel(sr_dir / "super_resolution_summary.md"),
            "metrics": {
                "case_count": summary.get("case_count"),
                "mean_psnr": summary.get("mean_psnr"),
                "mean_ssim": summary.get("mean_ssim"),
            },
        }

    if selected.get("id") == "private_2d_segmentation_train":
        # ── 私有 2D 训练 adapter ──
        private_data_root = Path(args.private_data_root) if args.private_data_root else None

        # 1. 前置检查：有标签才能训练
        validate_cmd = [
            sys.executable,
            str(REPO_ROOT / "platform_mvp/validate_private2d_package.py"),
            "--mode", "private_train",
        ]
        if private_data_root:
            validate_cmd.extend(["--data-root", str(private_data_root)])
        else:
            validate_cmd.extend(["--data-root", str(REPO_ROOT / "data/private01")])
        validate_out = run_dir / "logs" / "private2d_validation_report.json"
        validate_cmd.extend(["--out-dir", str(run_dir / "logs")])
        validate_out.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        _run_subprocess(validate_cmd, run_dir / "logs/private2d_validate_stdout.log",
                        run_dir / "logs/private2d_validate_stderr.log")

        validation = _read_json(validate_out) if validate_out.exists() else {}
        can_train = validation.get("can_execute_requested_mode", False)
        mask_count = validation.get("mask_count", 0)

        if not can_train:
            return {
                "status": "blocked_missing_labels",
                "reason": f"Private 2D training requires masks/labels. Found {mask_count} mask files. "
                          "Please supply images/ and masks/ directories with paired cases.",
                "validation": {k: validation.get(k) for k in
                               ["image_count", "mask_count", "paired_case_count", "can_execute_requested_mode"]
                               if k in validation},
            }

        # 2. 生成训练命令（dry-run 模式只出命令，--execute 才真跑）
        training_workspace = run_dir / "training_workspace"
        training_workspace.mkdir(parents=True, exist_ok=True)
        dataset_spec = _write_json(
            run_dir / "private2d_dataset_spec.json",
            {
                "root": str(private_data_root.resolve()) if private_data_root else str(REPO_ROOT / "data/private01"),
                "imagesTr": "imagesTr",
                "labelsTr": "labelsTr",
                "imagesTs": "imagesTs",
                "labelsTs": "labelsTs",
                "dataset_name": "Private 2D Dental Segmentation",
                "modality": "2d",
                "extra": {"target_backend": "builtin"},
            },
        )
        task_spec = _write_json(
            run_dir / "private2d_task_spec.json",
            {
                "task_id": "Private2DToothBinary",
                "modality": "2d",
                "task_type": "tooth_segmentation",
                "num_classes": 1,
                "class_names": ["teeth"],
                "primary_metric": "mean_dice",
                "extra": {"target_backend": "builtin"},
            },
        )
        budget_spec = _write_json(
            run_dir / "private2d_budget_spec.json",
            {
                "max_trials": 1,
                "max_epochs_per_trial": 1,
                "max_parallel": 1,
            },
        )

        train_cmd = [
            args.dentalclaw_python,
            str(REPO_ROOT / "agents/experimentation/skills/tooth_autotrain_nnunet/scripts/run_training.py"),
            "--dataset-spec", str(dataset_spec),
            "--task-spec", str(task_spec),
            "--budget-spec", str(budget_spec),
            "--workspace", str(training_workspace),
            "--detach",
        ]

        # plan-only: 只出命令不执行
        plan["execution"]["delegate_command"] = " ".join(str(p) for p in train_cmd)

        if not args.execute:
            return {
                "status": "command_generated",
                "command": train_cmd,
                "dataset_spec": _rel(dataset_spec),
                "task_spec": _rel(task_spec),
                "budget_spec": _rel(budget_spec),
                "training_workspace": _rel(training_workspace),
                "validation": {k: validation.get(k) for k in
                               ["image_count", "mask_count", "paired_case_count", "can_execute_requested_mode"]
                               if k in validation},
                "note": "训练命令已生成（plan-only 模式，未实际执行）。加 --execute 启动 smoke training。",
            }

        # --execute: 真正启动训练（1 trial / 1 epoch smoke）
        train_stdout = run_dir / "logs/training_stdout.log"
        train_stderr = run_dir / "logs/training_stderr.log"
        completed = _run_subprocess(train_cmd, train_stdout, train_stderr)

        return {
            "status": "completed" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "command": train_cmd,
            "stdout": _rel(train_stdout),
            "stderr": _rel(train_stderr),
            "dataset_spec": _rel(dataset_spec),
            "task_spec": _rel(task_spec),
            "budget_spec": _rel(budget_spec),
            "training_workspace": _rel(training_workspace),
            "validation": {k: validation.get(k) for k in
                           ["image_count", "mask_count", "paired_case_count", "can_execute_requested_mode"]
                           if k in validation},
            "note": "训练已启动（1 trial / 1 epoch smoke），通过 --detach 后台运行。",
        }

    if selected.get("id") == "toothfairy3_3d_segmentation_infer_or_train":
        # ── TF3 3D CBCT 分割 adapter ──
        DEFAULT_TF3 = REPO_ROOT.parent / "JoD/ToothFairy3_LPS"
        tf3_root = Path(args.tf3_data_root) if getattr(args, 'tf3_data_root', None) else DEFAULT_TF3

        # 推理模式：缺 checkpoint，直接返回 blocked（不跑 QC，避免卡住）
        mode = plan["intent"]["mode"]
        if mode == "inference":
            return {
                "status": "blocked_no_checkpoint",
                "reason": "3D CBCT inference requires a trained checkpoint. No TF3 model exists yet. "
                          "Please train a model first, or switch to train mode.",
            }

        # 训练模式：先跑 CBCT QC，再生成训练命令
        qc_out_dir = run_dir / "cbct_qc"
        qc_out_dir.mkdir(parents=True, exist_ok=True)
        qc_cmd = [
            sys.executable,
            str(REPO_ROOT / "agents/data_curator/skills/core/cbct_qc/scripts/audit_cbct_dataset.py"),
            "--dataset-root", str(tf3_root),
            "--label-policy", "optional",
            "--report-key", "platform_mvp_tf3",
            "--output-root", str(qc_out_dir),
        ]
        _run_subprocess(qc_cmd, run_dir / "logs/cbct_qc_stdout.log",
                        run_dir / "logs/cbct_qc_stderr.log")

        qc_summary_path = qc_out_dir / "cohort_summary.json"
        qc_data = _read_json(qc_summary_path) if qc_summary_path.exists() else {}
        usable = qc_data.get("status_counter", {}).get("usable", 0)
        review = qc_data.get("status_counter", {}).get("needs_manual_review", 0)

        # Training: 生成命令 + 可选执行
        training_workspace = run_dir / "training_workspace"
        training_workspace.mkdir(parents=True, exist_ok=True)
        # auto_train 要求 workspace 必须在 artifacts/ 下
        if str(REPO_ROOT / "artifacts") not in str(training_workspace.resolve()):
            training_workspace = REPO_ROOT / "artifacts/platform_mvp_runs" / run_dir.name / "training_workspace"
        training_workspace.mkdir(parents=True, exist_ok=True)
        dataset_spec = _write_json(
            run_dir / "tf3_dataset_spec.json",
            {
                "root": str(tf3_root.resolve()),
                "imagesTr": "imagesTr",
                "labelsTr": "labelsTr",
                "dataset_name": "ToothFairy3 3D CBCT",
                "modality": "3d",
                "extra": {"target_backend": "builtin"},
            },
        )
        task_spec = _write_json(
            run_dir / "tf3_task_spec.json",
            {
                "task_id": "ToothFairy3_CBCT_Seg",
                "modality": "3d",
                "task_type": "tooth_segmentation",
                "num_classes": 149,
                "class_names": ["multi_class_cbct"],
                "primary_metric": "mean_dice",
                "extra": {"target_backend": "builtin"},
            },
        )
        budget_spec = _write_json(
            run_dir / "tf3_budget_spec.json",
            {"max_trials": 1, "max_epochs_per_trial": 1, "max_parallel": 1},
        )

        train_cmd = [
            args.dentalclaw_python,
            str(REPO_ROOT / "agents/experimentation/skills/tooth_autotrain_nnunet/scripts/run_training.py"),
            "--dataset-spec", str(dataset_spec),
            "--task-spec", str(task_spec),
            "--budget-spec", str(budget_spec),
            "--workspace", str(training_workspace),
            "--detach",
        ]

        plan["execution"]["delegate_command"] = " ".join(str(p) for p in train_cmd)

        if not args.execute:
            return {
                "status": "command_generated",
                "command": train_cmd,
                "dataset_spec": _rel(dataset_spec),
                "task_spec": _rel(task_spec),
                "budget_spec": _rel(budget_spec),
                "training_workspace": _rel(training_workspace),
                "cbct_qc": {"usable": usable, "needs_review": review},
                "note": f"训练命令已生成。QC: {usable} usable + {review} needs_review。加 --execute 启动 smoke training。",
            }

        train_stdout = run_dir / "logs/training_stdout.log"
        train_stderr = run_dir / "logs/training_stderr.log"
        completed = _run_subprocess(train_cmd, train_stdout, train_stderr)
        return {
            "status": "completed" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "command": train_cmd,
            "stdout": _rel(train_stdout),
            "stderr": _rel(train_stderr),
            "dataset_spec": _rel(dataset_spec),
            "task_spec": _rel(task_spec),
            "budget_spec": _rel(budget_spec),
            "training_workspace": _rel(training_workspace),
            "cbct_qc": {"usable": usable, "needs_review": review},
            "note": "训练已启动（1 trial / 1 epoch smoke），通过 --detach 后台运行。",
        }

    if selected.get("id") == "dental_anomaly_detection":
        # ── 异常检测 adapter（无监督 baseline）──
        anomaly_dir = run_dir / "anomaly_detection"
        anomaly_dir.mkdir(parents=True, exist_ok=True)
        command = [
            args.dentalclaw_python,
            str(REPO_ROOT / "platform_mvp/run_anomaly_detection_mvp.py"),
            "--out-dir", str(anomaly_dir),
        ]
        stdout_path = run_dir / "logs/anomaly_detection_stdout.log"
        stderr_path = run_dir / "logs/anomaly_detection_stderr.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)

        if not args.execute:
            return {
                "status": "command_generated",
                "command": command,
                "note": "异常检测命令已生成（plan-only）。加 --execute 运行 ResNet18 + IsolationForest baseline。",
            }

        completed = _run_subprocess(command, stdout_path, stderr_path)
        summary_path = anomaly_dir / "anomaly_detection_summary.json"
        summary = _read_json(summary_path) if summary_path.exists() else {}
        return {
            "status": "completed" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "command": command,
            "stdout": _rel(stdout_path),
            "stderr": _rel(stderr_path),
            "delegate_run_dir": _rel(anomaly_dir),
            "delegate_manifest": _rel(summary_path),
            "delegate_summary": _rel(anomaly_dir / "anomaly_detection_summary.md"),
            "metrics": {
                "total_cases": summary.get("total_cases"),
                "anomaly_count": summary.get("anomaly_count"),
                "normal_count": summary.get("normal_count"),
            },
            "note": "无监督 baseline（ResNet18 + IsolationForest），异常分数基于特征分布而非 ground truth。",
        }

    if selected.get("id") != "tdd_2d_segmentation_infer_report":
        return {
            "status": "not_executed",
            "reason": "The selected route is registered, but no execution adapter is bound in this MVP runner.",
        }

    reuse_fullflow_run = args.reuse_fullflow_run or DEFAULT_SUCCESSFUL_FULLFLOW_RUN
    delegate_run_dir = Path(reuse_fullflow_run).resolve()
    if not delegate_run_dir.exists():
        delegate_run_dir = run_dir / "delegate_tdd_fullflow"

    command = [
        args.dentalclaw_python,
        str(REPO_ROOT / "mvp_fullflow/run_mvp_fullflow.py"),
        "--run-dir",
        str(delegate_run_dir),
        "--case-id",
        args.case_id,
    ]
    if args.reuse_inference:
        command.append("--reuse-inference")

    stdout_path = run_dir / "logs/delegate_stdout.log"
    stderr_path = run_dir / "logs/delegate_stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            stdout=stdout_file,
            stderr=stderr_file,
            check=False,
        )

    if completed.returncode != 0:
        return {
            "status": "failed",
            "returncode": completed.returncode,
            "command": command,
            "stdout": _rel(stdout_path),
            "stderr": _rel(stderr_path),
        }

    manifest_path = delegate_run_dir / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    return {
        "status": "completed",
        "command": command,
        "stdout": _rel(stdout_path),
        "stderr": _rel(stderr_path),
        "delegate_run_dir": _rel(delegate_run_dir),
        "delegate_manifest": _rel(manifest_path),
        "delegate_summary": _rel(delegate_run_dir / "mvp_summary.md"),
        "report_html": manifest.get("report_html"),
        "report_overlay": manifest.get("report_overlay"),
        "metrics": manifest.get("metrics", {}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DentalClaw platform MVP orchestrator.")
    parser.add_argument("--intent", required=True, help="One-sentence user request.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/platform_mvp_runs" / f"platform_mvp_{_now_stamp()}",
        help="Output directory for platform planning and execution evidence.",
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--case-id", default="100")
    parser.add_argument("--super-resolution-scale", type=int, default=2)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the route if the selected registry method has an executable adapter.",
    )
    parser.add_argument(
        "--reuse-inference",
        action="store_true",
        help="Pass --reuse-inference to the delegated full-flow route.",
    )
    parser.add_argument(
        "--reuse-fullflow-run",
        type=Path,
        default=DEFAULT_SUCCESSFUL_FULLFLOW_RUN,
        help="Existing mvp_fullflow run to reuse for quick reporting.",
    )
    parser.add_argument(
        "--dentalclaw-python",
        default=DEFAULT_DENTALCLAW_PYTHON,
        help="Python interpreter used by delegated DentalClaw scripts.",
    )
    parser.add_argument(
        "--private-data-root",
        default=None,
        help="Path to a private 2D dataset root (images/ + masks/) for private training route.",
    )
    parser.add_argument(
        "--tf3-data-root",
        default=None,
        help="Path to ToothFairy3 CBCT dataset root (imagesTr/ + labelsTr/). Default: JoD/ToothFairy3_LPS.",
    )
    parser.add_argument(
        "--allow-external",
        action="store_true",
        help="Allow auto-configuration of Agent-proposed external solutions (git clone + pip install).",
    )
    parser.add_argument(
        "--lang",
        default="auto",
        choices=["auto", "zh", "en"],
        help="Interface language: auto (detect from intent), zh (Chinese), en (English). Default: auto.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    trace = create_trace_recorder(args.run_dir, args.case_id)

    plan = build_plan(
        intent=args.intent,
        registry_path=args.registry,
        case_id=args.case_id,
        run_dir=run_dir,
        reuse_fullflow_run=args.reuse_fullflow_run,
        trace=trace,
        lang=args.lang,
    )

    execution_result = None
    if args.execute:
        plan["execution"]["will_execute"] = bool(plan.get("executable"))
        can_execute = plan.get("executable") or (
            getattr(args, "allow_external", False)
            and plan.get("selected_method", {}).get("status") == "external_suggestion"
        )
        if can_execute:
            execution_result = execute_selected_route(plan=plan, args=args, run_dir=run_dir)
        else:
            execution_result = {
                "status": "not_executed",
                "reason": "Selected method is not executable in the current MVP.",
            }

    _write_json(run_dir / "platform_plan.json", plan)
    if execution_result is not None:
        _write_json(run_dir / "execution_result.json", execution_result)
    (run_dir / "platform_summary.md").write_text(
        build_plan_md(plan, execution_result=execution_result),
        encoding="utf-8",
    )

    print("DentalClaw platform MVP completed.")
    print(f"Run directory: {_rel(run_dir)}")
    print(f"Plan: {_rel(run_dir / 'platform_plan.json')}")
    print(f"Summary: {_rel(run_dir / 'platform_summary.md')}")
    if execution_result is not None:
        print(f"Execution status: {execution_result.get('status')}")
    else:
        print("Execution status: plan_only")
    trace_path = trace.flush()
    print(f"Agent trace: {_rel(trace_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
