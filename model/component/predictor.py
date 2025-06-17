from typing import List, Optional, Tuple

import torch
from torch import nn


def apply_padding(input_tensor, padding, pad_value) -> torch.Tensor:
    return padding * pad_value + input_tensor * (1 - padding)


class RNNPredictor(nn.Module):

    def __init__(self,
                 voca_size: int,
                 embed_size: int,
                 output_size: int,
                 embed_dropout: float,
                 hidden_size: int,
                 num_layers: int,
                 bias: bool = True,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.n_layers = num_layers
        self.hidden_size = hidden_size
        self._output_size = output_size

        self.embed = nn.Embedding(voca_size, embed_size)
        self.dropout = nn.Dropout(embed_dropout)

        self.rnn = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.projection = nn.Linear(hidden_size, output_size)

    def output_size(self):
        return self._output_size

    def forward(
        self,
        input_tensor: torch.Tensor,
        cache: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        embed = self.embed(input_tensor)  # [batch, max_time, emb_size]
        embed = self.dropout(embed)

        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if cache is None:
            state = self.init_state(batch_size=input_tensor.size(
                0), device=input_tensor.device)
            states = (state[0], state[1])
        else:
            assert len(cache) == 2
            states = (cache[0], cache[1])

        out, (m, c) = self.rnn(embed, states)
        out = self.projection(out)

        return out

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        method: str = "zero",
    ) -> List[torch.Tensor]:
        assert batch_size > 0
        return [
            torch.zeros(self.n_layers, batch_size,
                        self.hidden_size, device=device),
            torch.zeros(self.n_layers, batch_size,
                        self.hidden_size, device=device)
        ]

    def forward_step(
            self,
            input_tensor: torch.Tensor,
            padding: torch.Tensor,
            cache: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        assert len(cache) == 2
        state_m, state_c = cache[0], cache[1]

        input_tensor = input_tensor.to(self.embed.weight.device)

        embed = self.embed(input_tensor)  # [batch, 1, emb_size]
        embed = self.dropout(embed)
        out, (m, c) = self.rnn(embed, (state_m, state_c))

        out = self.projection(out)
        m = apply_padding(m, padding.unsqueeze(0), state_m)
        c = apply_padding(c, padding.unsqueeze(0), state_c)

        return (out, [m, c])
