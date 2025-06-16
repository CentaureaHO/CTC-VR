from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class StreamingEncoder(nn.Module):
    """
    流式编码器，使用LSTM实现，支持分块推理
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # 在流式模式下，建议不使用双向LSTM
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.output_layer = nn.Linear(lstm_output_size, output_size)
    
    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        chunk_size: int = -1,
        num_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): [B, T, D]
            x_lens (torch.Tensor): [B]
            chunk_size (int): 分块大小，-1表示全局
            num_left_chunks (int): 左侧上下文块数量
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 输出和掩码
        """
        B, T, D = x.size()
        if chunk_size == -1:
            # 非流式模式，处理整个序列
            if x_lens is not None:
                # 处理变长序列
                x = nn.utils.rnn.pack_padded_sequence(
                    x, x_lens.cpu(), batch_first=True, enforce_sorted=False
                )
                
            self.lstm.flatten_parameters()
            outputs, _ = self.lstm(x)
            
            if x_lens is not None:
                # 解包
                outputs, _ = nn.utils.rnn.pad_packed_sequence(
                    outputs, batch_first=True, total_length=T
                )
        else:
            # 流式模式，按块处理
            outputs = []
            cache = self.init_state(B, device=x.device)
            
            # 如果num_left_chunks为-1，则使用所有可用的左侧上下文
            # 否则，使用指定数量的左侧块
            num_chunks = (T + chunk_size - 1) // chunk_size
            
            for i in range(num_chunks):
                start = max(0, i * chunk_size - num_left_chunks * chunk_size if num_left_chunks >= 0 else 0)
                end = min(T, (i + 1) * chunk_size)
                
                chunk = x[:, start:end, :]
                out_chunk, cache = self.forward_chunk(chunk, cache)
                
                # 只保留当前块的输出
                current_chunk_start = i * chunk_size
                current_chunk_end = min(T, (i + 1) * chunk_size)
                
                if i == 0:
                    outputs.append(out_chunk)
                else:
                    outputs.append(out_chunk[:, (start - current_chunk_start):, :])
                
            outputs = torch.cat(outputs, dim=1)
        
        # 应用输出层
        outputs = self.output_layer(outputs)
        
        # 创建掩码
        masks = self._create_masks(x_lens, outputs.size(1), outputs.device)
        
        return outputs, masks
    
    def forward_chunk(
        self,
        x: torch.Tensor,
        cache: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """处理单个数据块
        
        Args:
            x: 输入特征 [B, T, D]
            cache: 隐藏状态缓存 [h, c]
            
        Returns:
            输出特征和更新的缓存
        """
        self.lstm.flatten_parameters()
        h_0, c_0 = cache
        outputs, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        outputs = self.output_layer(outputs)
        return outputs, [h_n, c_n]
    
    def init_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> List[torch.Tensor]:
        """初始化LSTM隐藏状态
        
        Args:
            batch_size: 批大小
            device: 设备
            
        Returns:
            初始化的隐藏状态列表 [h_0, c_0]
        """
        direction = 2 if self.bidirectional else 1
        h_0 = torch.zeros(
            self.num_layers * direction, batch_size, self.hidden_size, device=device
        )
        c_0 = torch.zeros(
            self.num_layers * direction, batch_size, self.hidden_size, device=device
        )
        return [h_0, c_0]
    
    def _create_masks(
        self,
        lengths: torch.Tensor,
        max_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """创建掩码张量
        
        Args:
            lengths: 序列长度 [B]
            max_len: 最大长度
            device: 设备
            
        Returns:
            掩码张量 [B, 1, T]，True表示非填充位置
        """
        batch_size = lengths.size(0)
        masks = torch.arange(0, max_len, device=device).expand(batch_size, max_len) < lengths.unsqueeze(1)
        masks = masks.unsqueeze(1)  # [B, 1, T]
        return masks
    
    def get_output_size(self) -> int:
        """
        返回编码器输出大小
        """
        return self.output_size
