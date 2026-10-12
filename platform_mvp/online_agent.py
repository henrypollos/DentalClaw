"""Single-attempt, stateless planner transport with an explicit audit record."""
from __future__ import annotations
import hashlib
import json
import re
import socket
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SETTINGS = {
    'provider': 'deepseek', 'endpoint': 'https://api.deepseek.com/v1/chat/completions',
    'model': 'deepseek-v4-pro', 'temperature': 0.1, 'max_tokens': 2048,
    'thinking': {'type': 'disabled'}, 'response_format': {'type': 'json_object'},
    'timeout_seconds': 30, 'max_attempts': 1,
    'credential_env': 'DEEPSEEK_API_KEY',
}
SYSTEM_PROMPT = 'You are a DentalClaw platform planner. Return only valid JSON. Do not run tools or execute tasks.'


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def validate_settings(settings):
    if set(settings) != set(DEFAULT_SETTINGS):
        raise ValueError('Settings must use the documented schema, without credentials or extra fields')
    if settings['max_attempts'] != 1 or settings['response_format'] != {'type': 'json_object'}:
        raise ValueError('This transport uses one explicit backend without automatic fallback')
    if settings['provider'] == 'deepseek':
        if (settings['endpoint'] != DEFAULT_SETTINGS['endpoint'] or settings['thinking'] != {'type':'disabled'}
                or settings['credential_env'] != 'DEEPSEEK_API_KEY'):
            raise ValueError('DeepSeek uses its official endpoint and credential source')
    elif settings['provider'] == 'openai_compatible':
        url = urlsplit(settings['endpoint'])
        if (url.scheme != 'https' or not url.hostname or url.username or url.password or url.query or url.fragment
                or not url.path.endswith('/chat/completions') or settings['thinking'] is not None):
            raise ValueError('Custom backend requires an explicit HTTPS Chat Completions endpoint')
        if not re.fullmatch(r'DENTALCLAW_[A-Z0-9_]+_API_KEY', settings['credential_env']):
            raise ValueError('Custom backend requires its own DENTALCLAW_*_API_KEY variable; existing provider keys are not reused')
    else:
        raise ValueError('Unsupported explicit provider; configure deepseek or openai_compatible')
    if not isinstance(settings['model'], str) or not settings['model']:
        raise ValueError('An explicit model is required')
    if not 0 < settings['timeout_seconds'] <= 60 or not 1 <= settings['max_tokens'] <= 4096:
        raise ValueError('Timeout or token budget is outside the development bounds')


def validate_decision(value, registry_available):
    if not isinstance(value, dict) or value.get('decision') not in {'use_registry', 'external_proposal', 'reject'}:
        raise ValueError('Invalid decision schema')
    if value['decision'] == 'use_registry' and not registry_available:
        raise ValueError('use_registry requires an existing registry match')
    confidence = value.get('confidence')
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError('confidence must be a number in [0,1]')
    if not isinstance(value.get('reasoning'), str):
        raise ValueError('reasoning must be a string')
    if value['decision'] == 'external_proposal' and not isinstance(value.get('proposed_entrypoint'), str):
        raise ValueError('External proposals require a string entrypoint')
    return value


def call_planner(prompt, *, settings, api_key, registry_available, audit_dir=None, decision_validator=None):
    """No retries, model switching, reference answers, or execution capabilities."""
    validate_settings(settings)
    directory = Path(audit_dir) if audit_dir else None
    if directory:
        directory.mkdir(parents=True, exist_ok=False)
    payload = {k: settings[k] for k in ['model', 'temperature', 'max_tokens', 'response_format']}
    if settings['provider'] == 'deepseek':
        payload['thinking'] = settings['thinking']
    payload.update(messages=[{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}], stream=False)
    request_text = json.dumps(payload, ensure_ascii=False, indent=2)
    record = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'settings': settings,
              'prompt_sha256': digest(prompt), 'request_sha256': digest(request_text),
              'system_prompt_sha256': digest(SYSTEM_PROMPT), 'attempt_count': 0,
              'status': 'missing_credentials', 'decision': None, 'model_response_observed': False,
              'fallback_performed_by_transport': False}
    if directory:
        (directory / 'prompt.txt').write_text(prompt)
        (directory / 'request.json').write_text(request_text + '\n')
    raw = None
    started = time.monotonic()
    if api_key:
        record['attempt_count'] = 1
        req = urllib.request.Request(settings['endpoint'], data=request_text.encode(),
                                     headers={'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=settings['timeout_seconds']) as response:
                record['http_status'] = response.status
                raw = response.read().decode('utf-8').replace(api_key, '[REDACTED]')
            body = json.loads(raw)
            record.update(returned_model=body.get('model'), response_id=body.get('id'), usage=body.get('usage'))
            choice = body['choices'][0]
            record['finish_reason'] = choice.get('finish_reason')
            record['model_response_observed'] = True
            if choice.get('finish_reason') != 'stop':
                record['status'] = 'incomplete_response'
            else:
                value = json.loads(choice['message']['content'])
                try:
                    record['decision'] = (decision_validator(value) if decision_validator
                                          else validate_decision(value, registry_available))
                    record['status'] = 'ok'
                except ValueError:
                    record['status'] = 'invalid_schema'
        except urllib.error.HTTPError as exc:
            record.update(status='http_error', http_status=exc.code)
            raw = exc.read().decode('utf-8', errors='replace').replace(api_key, '[REDACTED]')
        except (socket.timeout, TimeoutError):
            record['status'] = 'timeout'
        except urllib.error.URLError as exc:
            record.update(status='timeout' if isinstance(exc.reason, (socket.timeout, TimeoutError)) else 'connection_error',
                          error_type=type(exc.reason).__name__)
        except OSError as exc:
            record.update(status='connection_error', error_type=type(exc).__name__)
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            record.update(status='invalid_response', error_type=type(exc).__name__)
    record['elapsed_seconds'] = round(time.monotonic() - started, 3)
    if raw is not None:
        record['raw_response_sha256'] = digest(raw)
        record['raw_response_redacted'] = bool(api_key and '[REDACTED]' in raw)
        if directory:
            (directory / 'raw_response.txt').write_text(raw)
    if directory:
        record['audit_dir'] = str(directory.resolve())
        (directory / 'call_record.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    return record
