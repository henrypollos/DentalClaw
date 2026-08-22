import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_trace.openclaw_runner import (
    build_platform_mvp_prompt,
    normalize_platform_mvp_payload,
    _strip_ansi,
)


def test_build_platform_mvp_prompt_mentions_platform_entrypoint():
    prompt = build_platform_mvp_prompt("用 TDD 全景片做牙齿二值分割推理")
    assert "platform_mvp/run_platform_mvp.py" in prompt
    assert "json" in prompt.lower()


def test_normalize_platform_mvp_payload_handles_json_string_and_defaults():
    payload = normalize_platform_mvp_payload(
        '{"intent": "用 TDD 全景片做牙齿二值分割推理", "execute": true, "case_id": "100"}'
    )
    assert payload["intent"] == "用 TDD 全景片做牙齿二值分割推理"
    assert payload["execute"] is True
    assert payload["case_id"] == "100"
    assert "platform_mvp_runs" in payload["run_dir"]


def test_normalize_extracts_from_openclaw_result_structure():
    """模拟 OpenClaw 的三层嵌套: { result: { payloads: [{text: "{...}"}] } }"""
    payload = normalize_platform_mvp_payload(
        {
            "result": {
                "payloads": [
                    {"text": '{"intent": "超分测试", "execute": false, "case_id": "50"}'}
                ]
            }
        }
    )
    assert payload["intent"] == "超分测试"
    assert payload["execute"] is False
    assert payload["case_id"] == "50"


def test_normalize_falls_back_to_prompt_on_noise():
    """当输入是含有 ANSI 码的噪声文本时，回退到 fallback_intent。"""
    noisy = "\x1b[35m[plugins]\x1b[39m some noise\n\x1b[36m[plugins]\x1b[39m more noise"
    payload = normalize_platform_mvp_payload(noisy, fallback_intent="用 TDD 做分割")
    assert "TDD" in payload["intent"]
    assert "分割" in payload["intent"]


def test_strip_ansi_removes_color_codes():
    text = "\x1b[35m[plugins]\x1b[39m \x1b[36mfeishu_doc: Registered\x1b[39m"
    clean = _strip_ansi(text)
    assert "\x1b" not in clean
    assert "feishu_doc" in clean
    assert "Registered" in clean
