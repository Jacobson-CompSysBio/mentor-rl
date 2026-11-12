"""Fingerprint the production wire, including historical conversation context.

The pinned current-turn serializer removes earlier assistant analysis before
rendering context. That private historical text cannot create extra replay
examples. Source records themselves remain untouched.
"""

from .s2_audit_identity import model_trace_sha256 as source_trace_sha256


def model_trace_sha256(row):
    """Ignore masked historical analysis and canonicalize opaque acquisitions."""
    last_user = max(index for index, message in enumerate(row['messages']) if message['role'] == 'user')
    messages = [({key: value for key, value in message.items() if key != 'thinking'}
                 if index < last_user and message['role'] == 'assistant' else message)
                for index, message in enumerate(row['messages'])]
    return source_trace_sha256({**row, 'messages': messages})
