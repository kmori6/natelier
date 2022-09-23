import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_attention_heads: int, dropout_rate: float):
        super().__init__()
        self.h = num_attention_heads
        self.d_model = d_model
        self.d_k = self.d_v = int(d_model / num_attention_heads)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            q (torch.Tensor): (B, L1, D)
            k (torch.Tensor): (B, L2, D)
            v (torch.Tensor): (B, L2, D)
            mask (torch.Tensor): (B, L1) or (B, L1, L2)
        """
        q = self.w_q(q).view(q.size(0), -1, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(k.size(0), -1, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(v.size(0), -1, self.h, self.d_v).transpose(1, 2)
        matmul = torch.matmul(q, k.transpose(2, 3))  # (B, H, L1, L2)
        scale = matmul / torch.sqrt(torch.tensor(self.d_k))  # (B, H, L1, L2)
        b, h, l1, l2 = scale.size()
        if len(mask.size()) == 2:
            mask = mask.repeat_interleave(h * l1, dim=0).view(b, h, l1, l2)
        else:
            mask = mask.repeat_interleave(h, dim=0).view(b, h, l1, l2)
        scale[mask == 0] = scale.new_tensor(torch.finfo(torch.float32).min)
        softmax = torch.softmax(scale, dim=-1)  # (B, H, L1, L2)
        matmul = torch.matmul(self.dropout(softmax), v)  # (B, H, L1, D_k)
        concat = matmul.transpose(1, 2).contiguous().view(b, l1, self.d_model)
        return self.w_o(concat)  # (B, L1, D)
