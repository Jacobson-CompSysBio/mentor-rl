"""Define the versioned system prompts for the world model.
Additional prompts are added once each stage has passed."""

# imports
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib


# define prompt contract version
PROMPT_CONTRACT_SCHEMA_VERSION = "mentor-rl-world-model-stage-prompts-v1"

# define stage 0 prompt contract specs
S0_STAGE = "S0"
S0_ALLOWED_BOOK_MODES = ("closed_book",)  # closed book only for recall tests
S0_SOURCE_DOCUMENT = (
    "agents/world_model_v2_curriculum/S0-human-gene-identifiers.md"
)

# define the system prompt for stage S0
S0_SYSTEM_PROMPT = (
    "Resolve only human Ensembl gene IDs and gene symbols from the pinned "
    "registry. Return exactly one compact JSON object with no extra text or "
    "fields. For an ambiguous symbol, return the complete candidate set and "
    "the defer action. Do not use graph facts, tools, or another species."
)


# define the prompt contract for a stage
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


def _sha256_text(value: str) -> str:
    """Return the SHA256 hash of the specified string."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stage_prompt_contract(stage: str) -> StagePromptContract:
    """Return the fixed S0 prompt contract."""

    # normalize the stage string to uppercase
    normalized_stage = stage.upper()

    # reject all stages except S0
    if normalized_stage != S0_STAGE:
        raise ValueError(f"Unknown S0 stage: {stage!r}")

    # return the prompt contract for S0
    return StagePromptContract(
        schema_version=PROMPT_CONTRACT_SCHEMA_VERSION,
        stage=S0_STAGE,
        allowed_book_modes=S0_ALLOWED_BOOK_MODES,
        source_document=S0_SOURCE_DOCUMENT,
        system_prompt=S0_SYSTEM_PROMPT,
        system_prompt_sha256=_sha256_text(S0_SYSTEM_PROMPT),
    )
