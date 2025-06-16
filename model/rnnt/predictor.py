from typing import List, Optional, Tuple

import torch
from torch import nn


def ApplyPadding(input, padding, pad_value) -> torch.Tensor:
    """
    Args:
        input:   [bs, max_time_step, dim]
        padding: [bs, max_time_step]
    """
    return padding * pad_value + input * (1 - padding)


class PredictorBase(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def init_state(self,
                   batch_size: int,
                   device: torch.device,
                   method: str = "zero") -> List[torch.Tensor]:
        _, _, _ = batch_size, method, device
        raise NotImplementedError("this is a base predictor")

    def batch_to_cache(self,
                       cache: List[torch.Tensor]) -> List[List[torch.Tensor]]:
        _ = cache
        raise NotImplementedError("this is a base predictor")

    def cache_to_batch(self,
                       cache: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        _ = cache
        raise NotImplementedError("this is a base predictor")

    def output_size(self):
        raise NotImplementedError("this is a base predictor")

    def forward(
        self,
        input: torch.Tensor,
        cache: Optional[List[torch.Tensor]] = None,
    ):
        _, _, = input, cache
        raise NotImplementedError("this is a base predictor")

    def forward_step(
            self, input: torch.Tensor, padding: torch.Tensor,
            cache: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        _, _, _, = input, padding, cache
        raise NotImplementedError("this is a base predictor")


class RNNPredictor(PredictorBase):
    def __init__(self,
                 voca_size: int,
                 embed_size: int,
                 output_size: int,
                 embed_dropout: float,
                 hidden_size: int,
                 num_layers: int,
                 bias: bool = True,
                 rnn_type: str = "lstm",
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.n_layers = num_layers
        self.hidden_size = hidden_size
        self._output_size = output_size
        self.embed = nn.Embedding(voca_size, embed_size)
        self.dropout = nn.Dropout(embed_dropout)
        
        # 根据rnn_type选择RNN类型
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=embed_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              bias=bias,
                              batch_first=True,
                              dropout=dropout)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(input_size=embed_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             bias=bias,
                             batch_first=True,
                             dropout=dropout)
        else:
            self.rnn = nn.RNN(input_size=embed_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             bias=bias,
                             batch_first=True,
                             dropout=dropout)
        
        self.projection = nn.Linear(hidden_size, output_size)

    def output_size(self):
        return self._output_size

    def forward(
        self,
        input: torch.Tensor,
        cache: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): [batch, max_time).
            cache : rnn predictor cache[0] == state_m
                    cache[1] == state_c
        Returns:
            output: [batch, max_time, output_size]
        """
        embed = self.embed(input)  # [batch, max_time, emb_size]
        embed = self.dropout(embed)
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if cache is None:
            state = self.init_state(batch_size=input.size(0),
                                    device=input.device)
            states = (state[0], state[1])
        else:
            assert len(cache) == 2
            states = (cache[0], cache[1])
        out, (m, c) = self.rnn(embed, states)
        out = self.projection(out)
        
        # NOTE: 虽然在训练forward中不使用state，但我们需要确保padding值正确
        # 所以为推理创建forward_step
        _, _ = m, c
        return out

    def batch_to_cache(self,
                       cache: List[torch.Tensor]) -> List[List[torch.Tensor]]:
        """
        Args:
           cache: [state_m, state_c]
               state_ms: [1*n_layers, bs, ...]
               state_cs: [1*n_layers, bs, ...]
        Returns:
           new_cache: [[state_m_1, state_c_1], [state_m_2, state_c_2]...]
        """
        assert len(cache) == 2
        state_ms = cache[0]
        state_cs = cache[1]

        assert state_ms.size(1) == state_cs.size(1)

        new_cache: List[List[torch.Tensor]] = []
        for state_m, state_c in zip(torch.split(state_ms, 1, dim=1),
                                    torch.split(state_cs, 1, dim=1)):
            new_cache.append([state_m, state_c])
        return new_cache

    def cache_to_batch(self,
                       cache: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """
        Args:
            cache : [[state_m_1, state_c_1], [state_m_1, state_c_1]...]

        Returns:
            new_caceh: [state_ms, state_cs],
                state_ms: [1*n_layers, bs, ...]
                state_cs: [1*n_layers, bs, ...]
        """
        state_ms = torch.cat([states[0] for states in cache], dim=1)
        state_cs = torch.cat([states[1] for states in cache], dim=1)
        return [state_ms, state_cs]

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        method: str = "zero",
    ) -> List[torch.Tensor]:
        assert batch_size > 0
        # TODO: 支持xavier初始化方法
        _ = method
        return [
            torch.zeros(1 * self.n_layers,
                        batch_size,
                        self.hidden_size,
                        device=device),
            torch.zeros(1 * self.n_layers,
                        batch_size,
                        self.hidden_size,
                        device=device)
        ]

    def forward_step(
            self, input: torch.Tensor, padding: torch.Tensor,
            cache: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            input (torch.Tensor): [batch_size, time_step=1]
            padding (torch.Tensor): [batch_size,1], 1 is padding value
            cache : rnn predictor cache[0] == state_m
                    cache[1] == state_c
        """
        assert len(cache) == 2
        state_m, state_c = cache[0], cache[1]
        embed = self.embed(input)  # [batch, 1, emb_size]
        embed = self.dropout(embed)
        out, (m, c) = self.rnn(embed, (state_m, state_c))

        out = self.projection(out)
        m = ApplyPadding(m, padding.unsqueeze(0), state_m)
        c = ApplyPadding(c, padding.unsqueeze(0), state_c)

        return (out, [m, c])
