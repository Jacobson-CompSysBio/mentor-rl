"""Identify distinct model traces without counting cosmetic acquisition IDs.

Metadata and row IDs are not model inputs. Acquisition IDs preserve equality
and reuse relationships, but their arbitrary spelling must not create extra
examples for production quotas.
"""

import hashlib
import json
import re


def model_trace_sha256(row):
    """Hash model messages/tools after canonical acquisition-ID renaming."""
    messages = []
    for message in row['messages']:
        item = dict(message)
        if item['role'] == 'harness':
            try:
                item['content'] = json.loads(item['content'])
            except (ValueError, TypeError):
                pass
        messages.append(item)
    identities = {}

    def find(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == 'evidence_id' and isinstance(item, str):
                    identities.setdefault(item, f'audit_acquisition_{len(identities)}')
                find(item)
        elif isinstance(value, list):
            for item in value:
                find(item)

    find(messages)
    pattern = re.compile('|'.join(re.escape(item) for item in sorted(identities, key=lambda item: (-len(item), item)))) if identities else None

    def normalize(value):
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, str) and pattern is not None:
            value = pattern.sub(lambda match: identities[match.group(0)], value)
        return value

    payload = normalize({'messages': messages, 'tools': row['tools']})
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()
