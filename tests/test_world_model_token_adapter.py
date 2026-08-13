from __future__ import annotations

import torch
from torch import nn

from runtime.world_model_token_adapter import (
    TokenRowDeltaEmbedding,
    TokenRowDeltaOutput,
    _mean_source_row_batch,
)


def test_mean_source_row_batch_uses_each_source_group() -> None:
    """Calculate one mean row for each source group."""

    weight = torch.tensor(
        [
            [0.0, 2.0],
            [2.0, 4.0],
            [8.0, 10.0],
        ]
    )
    tokens = [
        {"base_token_ids": [0, 1]},
        {"base_token_ids": [2]},
    ]

    result = _mean_source_row_batch(
        weight,
        tokens,
        chunk_size=1,
    )

    expected = torch.tensor(
        [
            [1.0, 3.0],
            [8.0, 10.0],
        ]
    )
    assert torch.equal(result, expected)


def test_token_adapter_extends_base_vocabulary() -> None:
    """Check reserved and appended token rows."""

    base_vocab = 8
    token_start = 6
    token_count = 4
    hidden_size = 3

    input_layer = nn.Embedding(
        base_vocab,
        hidden_size,
    )
    output_layer = nn.Linear(
        hidden_size,
        base_vocab,
        bias=False,
    )

    for parameter in input_layer.parameters():
        parameter.requires_grad = False
    for parameter in output_layer.parameters():
        parameter.requires_grad = False

    desired_input = torch.arange(
        token_count * hidden_size,
        dtype=torch.float32,
    ).reshape(token_count, hidden_size)

    desired_output = desired_input + 10.0

    input_delta = desired_input.clone()
    output_delta = desired_output.clone()

    reserved_count = base_vocab - token_start
    input_delta[:reserved_count].sub_(
        input_layer.weight[token_start:base_vocab]
    )
    output_delta[:reserved_count].sub_(
        output_layer.weight[token_start:base_vocab]
    )

    input_adapter = TokenRowDeltaEmbedding(
        input_layer,
        token_start,
        nn.Parameter(input_delta),
    )
    output_adapter = TokenRowDeltaOutput(
        output_layer,
        token_start,
        base_vocab,
        nn.Parameter(output_delta),
    )

    input_ids = torch.tensor(
        [[0, 6, 7, 8, 9]]
    )
    embeddings = input_adapter(input_ids)

    assert input_adapter.final_vocab_size == 10
    assert torch.allclose(
        embeddings[0, 0],
        input_layer.weight[0],
    )
    assert torch.allclose(
        embeddings[0, 1:],
        desired_input,
    )

    hidden_states = torch.ones(
        (1, 2, hidden_size)
    )
    base_logits = output_layer(hidden_states)
    logits = output_adapter(hidden_states)

    assert output_adapter.final_vocab_size == 10
    assert logits.shape == (1, 2, 10)
    assert torch.allclose(
        logits[..., :token_start],
        base_logits[..., :token_start],
    )
    assert torch.allclose(
        logits[..., token_start:],
        nn.functional.linear(
            hidden_states,
            desired_output,
        ),
    )

    loss = (
        embeddings.square().mean()
        + logits.square().mean()
    )
    loss.backward()

    input_gradient = (
        input_adapter.token_input_delta.grad
    )
    output_gradient = (
        output_adapter.token_output_delta.grad
    )

    assert input_gradient is not None
    assert output_gradient is not None
    assert torch.isfinite(input_gradient).all()
    assert torch.isfinite(output_gradient).all()


def test_token_adapter_rejects_out_of_range_id() -> None:
    """Reject a token ID beyond the extended vocabulary."""

    base_layer = nn.Embedding(4, 2)
    for parameter in base_layer.parameters():
        parameter.requires_grad = False

    adapter = TokenRowDeltaEmbedding(
        base_layer,
        token_start=3,
        delta=nn.Parameter(torch.zeros(2, 2)),
    )

    invalid_ids = torch.tensor([[5]])

    try:
        adapter(invalid_ids)
    except RuntimeError as error:
        assert "outside the extended vocabulary" in str(error)
    else:
        raise AssertionError(
            "The invalid token ID did not fail"
        )