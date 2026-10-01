"""Define the versioned system prompts for the world model.
Additional prompts are added once each stage has passed."""

# imports
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib


# define prompt contract version
PROMPT_CONTRACT_SCHEMA_VERSION = "mentor-rl-world-model-stage-prompts-v1"

# Define the S0 prompt contract.
S0_STAGE = "S0"
S0_ALLOWED_BOOK_MODES = ("tool_call",)
S0_SOURCE_DOCUMENT = (
    "agents/world_model_v2_curriculum/S0-human-gene-identifiers.md"
)

S0_TOOL_TRAJECTORY_SYSTEM_PROMPT = (
    "Use exactly one provided tool to resolve each human Ensembl gene ID "
    "or gene symbol from the pinned Ensembl release 116 registry. Before "
    "the tool call, state why the lookup is required. Copy the input value "
    "exactly into the matching tool argument. Do not answer before the tool "
    "result. After the tool result, answer the task with one short sentence. "
    "Do not use graph facts, network services, or another species."
)

S0_SHARED_TRACE_SYSTEM_PROMPT = (
    "Use exactly one provided tool to resolve each human Ensembl gene ID "
    "or gene symbol with the pinned Ensembl release 116 registry. "
    "Before the call, explain the applicable rule and next action. "
    "Copy the input exactly into the matching tool argument. "
    "After the result, explain the result rule and next action. "
    "Return one JSON object with status, result, continuation_decision, "
    "and summary in that order. Do not use graph facts, network services, "
    "or another species."
)

S1_STAGE = "S1"
S1_ALLOWED_BOOK_MODES = ("tool_call", "closed_book")
S1_SOURCE_DOCUMENT = (
    "agents/world_model_v2_curriculum/S1-coordinate-system.md"
)
S1_SHARED_TRACE_SYSTEM_PROMPT = (
    "Use the provided local tools only when the task needs a fact that the "
    "prompt does not supply. Treat a labeled registry block as authoritative. "
    "Before each call, explain the applicable rule and next action. After the "
    "last call, use its result. In the final action, explain the result rule "
    "and claim limit. Return one JSON object with status, result, "
    "continuation_decision, and summary in that order. Do not invent "
    "identifiers or graph facts."
)

# Define the prompt contract for a stage.
@dataclass(frozen=True)
class StagePromptContract:
    """Contain the prompt contract for a stage.

    Args:
        schema_version: The version of the prompt contract schema.
        stage: The stage of the world model.
        allowed_book_modes: The allowed book modes for the stage.
        source_document: The source document for the system prompt.
        system_prompt: The system prompt for the stage.
        system_prompt_sha256: The SHA256 hash of the system prompt.
    """

    schema_version: str
    stage: str
    allowed_book_modes: tuple[str, ...]
    source_document: str
    system_prompt: str
    system_prompt_sha256: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["allowed_book_modes"] = list(self.allowed_book_modes)
        return payload


def s0_tool_trajectory_prompt_contract() -> StagePromptContract:
    """Return the fixed S0 tool trajectory prompt contract."""

    return StagePromptContract(
        schema_version=PROMPT_CONTRACT_SCHEMA_VERSION,
        stage=S0_STAGE,
        allowed_book_modes=S0_ALLOWED_BOOK_MODES,
        source_document=S0_SOURCE_DOCUMENT,
        system_prompt=S0_TOOL_TRAJECTORY_SYSTEM_PROMPT,
        system_prompt_sha256=_sha256_text(S0_TOOL_TRAJECTORY_SYSTEM_PROMPT),
    )


def s0_shared_trace_prompt_contract() -> StagePromptContract:
    """Return the fixed S0 shared trace prompt contract."""

    return StagePromptContract(
        schema_version=PROMPT_CONTRACT_SCHEMA_VERSION,
        stage=S0_STAGE,
        allowed_book_modes=S0_ALLOWED_BOOK_MODES,
        source_document=S0_SOURCE_DOCUMENT,
        system_prompt=S0_SHARED_TRACE_SYSTEM_PROMPT,
        system_prompt_sha256=_sha256_text(S0_SHARED_TRACE_SYSTEM_PROMPT),
    )


def s1_shared_trace_prompt_contract() -> StagePromptContract:
    """Return the shared S1 prompt contract."""

    return StagePromptContract(
        schema_version=PROMPT_CONTRACT_SCHEMA_VERSION,
        stage=S1_STAGE,
        allowed_book_modes=S1_ALLOWED_BOOK_MODES,
        source_document=S1_SOURCE_DOCUMENT,
        system_prompt=S1_SHARED_TRACE_SYSTEM_PROMPT,
        system_prompt_sha256=_sha256_text(S1_SHARED_TRACE_SYSTEM_PROMPT),
    )


def _sha256_text(value: str) -> str:
    """Return the SHA256 hash of the specified string."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
