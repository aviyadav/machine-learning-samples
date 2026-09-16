"""The toy message-passing GNN from the article.

A single node type and two relation types (``local`` and ``cross``).  Graph
neural networks are notorious for dynamic shapes, which makes this model
unusually prone to recompilation -- exactly what the experiments need.

The only change from the article's listing is that the tensor dimensions come
from :class:`~recompilation_dilemma.config.ModelConfig` instead of module-level
constants, so the same model can be run at article scale on a GPU and at smoke
scale on a laptop.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from recompilation_dilemma.config import ModelConfig


class MessagePassingModel(nn.Module):
    """A minimal GNN whose shapes are driven entirely by its inputs."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()

        feature_dim = self.config.feature_dim
        hidden_dim = self.config.hidden_dim
        layers = self.config.message_passing_layers

        # Encode each node feature vector as a hidden_dim-value hidden state.
        self.input_projection = nn.Linear(feature_dim, hidden_dim)

        # Each layer has one shared message transform and one node update
        # transform.  Local and cross relations use the same learned message
        # transform, but keep separate edges and per-edge scales.
        self.message_projections = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)
        )
        self.update_projections = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)
        )

        # Project pooled feature_dim-value context rows into the hidden-state
        # space, then map the combined graph representation to four outputs.
        self.context_projection = nn.Linear(feature_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, 4)

        # Counts the passes that reached this module without being compiled,
        # i.e. the guard-miss eager fallbacks of the hybrid mode.
        self.eager_passes = 0

    def reset_eager_passes(self) -> None:
        self.eager_passes = 0

    def forward(
        self,
        x: torch.Tensor,
        local_edge_index: torch.Tensor,
        cross_edge_index: torch.Tensor,
        local_edge_scale: torch.Tensor,
        cross_edge_scale: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        # Counting here would be a graph break under Dynamo, so only count the
        # passes that Dynamo did not take over.
        if not torch.compiler.is_compiling():
            self.eager_passes += 1

        # One initial hidden state per node: [num_nodes, hidden_dim].
        state = self.input_projection(x).relu()

        for message_projection, update_projection in zip(
            self.message_projections, self.update_projections, strict=True
        ):
            # edge_index[0] gathers one source-node state per edge.  Each
            # gathered state is transformed and then weighted per edge.
            local_messages = (
                message_projection(state[local_edge_index[0]]) * local_edge_scale
            )
            cross_messages = (
                message_projection(state[cross_edge_index[0]]) * cross_edge_scale
            )

            # edge_index[1] identifies destinations.  index_add sums incoming
            # messages at every destination node.  The relations aggregate
            # independently, so either relation may be empty.
            local_sum = torch.zeros_like(state).index_add(
                0, local_edge_index[1], local_messages
            )
            cross_sum = torch.zeros_like(state).index_add(
                0, cross_edge_index[1], cross_messages
            )

            # A learned self/update term plus both message sums becomes the
            # next node state.  Repeating this propagates information across
            # multiple graph hops.
            state = (
                update_projection(state) + local_sum + cross_sum
            ).relu()

        # Sum node and context rows into two graph-level hidden_dim-value
        # vectors, combine them, and predict four output values.
        return self.output_projection(
            state.sum(0) + self.context_projection(context).sum(0)
        )


def build_model(config: ModelConfig | None = None, *, seed: int = 0) -> MessagePassingModel:
    """Construct a model with reproducible parameters.

    Every experiment rebuilds the model from the same seed so that eager and
    compiled runs, and every mitigation, see identical weights.
    """
    torch.manual_seed(seed)
    return MessagePassingModel(config)
