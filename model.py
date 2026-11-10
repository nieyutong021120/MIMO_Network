"""FC-NN-DPD 模型：BOTDNN 共享主干 + 早期跨波束上下文注入。

全局上下文：
    所有波束全延时展平 → Context MLP → 加性注入 RVTBNN 第一层

自波束路径（BOTDNN + 上下文调制）：
    每路波束 → 共享 RVTBNN（h1 += context）→ 幅度非线性特征 → PRB → 2*hidden_2 维

输出：
    逐波束独立 Linear → 2 维
"""

import torch
import torch.nn as nn


def PRB(a, cos, sin):
    """相位恢复模块"""
    p_len = cos.shape[1]
    a_len = a.shape[1]
    d_len = a_len - p_len
    t_cos = cos
    t_sin = sin

    if d_len > 0:
        while int(d_len / p_len):
            cos = torch.cat((cos, t_cos), 1)
            sin = torch.cat((sin, t_sin), 1)
            d_len = d_len - p_len
        cos = torch.cat((cos, t_cos[:, :d_len]), 1)
        sin = torch.cat((sin, t_sin[:, :d_len]), 1)
        return torch.cat((a * cos, a * sin), 1)
    elif d_len < 0:
        cos = t_cos[:, :d_len]
        sin = t_sin[:, :d_len]
        return torch.cat((a * cos, a * sin), 1)
    else:
        return torch.cat((a * cos, a * sin), 1)


class BeamRVTBNN(nn.Module):
    """单波束 RVTBNN：复数线性变换 -> 幅度非线性特征 -> PRB，支持外部上下文注入。"""

    def __init__(self, n_input, n_hidden_1, n_hidden_2):
        super().__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden_1)
        self.hidden2 = nn.Sequential(
            nn.Linear((n_hidden_1 // 2) * 3, n_hidden_2),
            nn.Tanh(),
            # nn.Linear(n_hidden_2, n_hidden_2),
            # nn.Tanh(),
        )
        # self.hidden3 = nn.Sequential(
        #     nn.Linear(n_hidden_1, 2 * n_hidden_2),
        #     nn.Tanh(),
        # )

    def forward(self, x, context):
        h1 = self.hidden1(x)
        # if context is not None:
        #     h1 = h1 + context
        h1_abs = (h1[:, ::2] ** 2 + h1[:, 1::2] ** 2).pow(0.5)
        # x_abs = (x[:, ::2] ** 2 + x[:, 1::2] ** 2).pow(0.5)
        # h1_angle = torch.atan2(h1[:, ::2], h1[:, 1::2])
        h1_abs_pow = torch.cat((h1_abs, h1_abs ** 2, h1_abs ** 3), 1)
        # x_abs_pow = torch.cat((x_abs, x_abs ** 2, x_abs ** 3), 1)
        # f_cos = torch.cos(h1_angle)
        # f_sin = torch.sin(h1_angle)
        h2 = self.hidden2(h1_abs_pow)
        
        return PRB(h2, h1[:, ::2], h1[:, 1::2])

class BeamTBNN(nn.Module):
    """单波束 RVTBNN：复数线性变换 -> 幅度非线性特征 -> PRB，支持外部上下文注入。"""

    def __init__(self, n_input, n_hidden_1, n_hidden_2):
        super().__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden_1)
        self.hidden2 = nn.Sequential(
            nn.Linear((n_hidden_1 // 2) * 3, n_hidden_2),
            nn.Tanh(),
            # nn.Linear(n_hidden_2, n_hidden_2),
            # nn.Tanh(),
        )
        # self.hidden3 = nn.Sequential(
        #     nn.Linear(n_hidden_1, 2 * n_hidden_2),
        #     nn.Tanh(),
        # )

    def forward(self, x, context):
        # 每个波束的信息
        h1 = self.hidden1(x)
        # h1_abs = (h1[:, ::2] ** 2 + h1[:, 1::2] ** 2).pow(0.5)
        context_abs = (context[:, ::2] ** 2 + context[:, 1::2] ** 2).pow(0.5)
        # h1_angle = torch.atan2(h1[:, ::2], h1[:, 1::2])
        # h1_abs_pow = torch.cat((h1_abs, h1_abs ** 2, h1_abs ** 3), 1)
        context_abs_pow = torch.cat((context_abs, context_abs ** 2, context_abs ** 3), 1)
        # f_cos = torch.cos(h1_angle)
        # f_sin = torch.sin(h1_angle)
        context2 = self.hidden2(context_abs_pow)
        return PRB(context2, h1[:, ::2], h1[:, 1::2])

class BeamGRU_RVT(nn.Module):
    """单波束 GRU-RVT：复数线性变换 → 幅度序列 → GRU → PRB。

    结合 RVTBNN 的幅度-相位分离思想和 GRU 的序列建模能力。
    """

    def __init__(self, n_input, n_hidden_1, gru_hidden_size):
        super().__init__()
        self.n_hidden_1 = n_hidden_1
        self.gru_hidden_size = gru_hidden_size

        # 第一层：复数线性变换
        self.hidden1 = nn.Linear(n_input, n_hidden_1)

        # GRU 处理幅度序列
        # 输入：(batch, seq_len=n_hidden_1//2, input_size=2) — [abs, abs²]
        self.gru = nn.GRU(
            input_size=2,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
        )

    def reset_parameters(self):
        # 初始化 hidden1
        nn.init.xavier_uniform_(self.hidden1.weight)
        nn.init.constant_(self.hidden1.bias, 0)

    def forward(self, x, context=None):
        # 第一层线性变换
        h1 = self.hidden1(x)
        if context is not None:
            h1 = h1 + context
        
        # 提取幅度和相位
        # h1: (batch, n_hidden_1) → 视为 n_hidden_1//2 个复数
        h1_real = h1[:, ::2]   # (batch, n_hidden_1//2)
        h1_imag = h1[:, 1::2]  # (batch, n_hidden_1//2)

        h1_abs = (h1_real ** 2 + h1_imag ** 2).pow(0.5)  # (batch, n_hidden_1//2)
        h1_angle = torch.atan2(h1_real, h1_imag)          # (batch, n_hidden_1//2)

        # 构造幅度特征序列：[abs, abs²]
        h1_abs_pow = torch.stack([h1_abs, h1_abs ** 2], dim=-1)  # (batch, n_hidden_1//2, 2)

        # GRU 处理幅度序列
        gru_out, _ = self.gru(h1_abs_pow)  # (batch, n_hidden_1//2, gru_hidden_size)
        gru_last = gru_out[:, -1, :]       # (batch, gru_hidden_size)

        # PRB：结合幅度和相位（直接使用 GRU 输出）
        f_cos = torch.cos(h1_angle)  # (batch, n_hidden_1//2)
        f_sin = torch.sin(h1_angle)  # (batch, n_hidden_1//2)

        return PRB(gru_last, f_cos, f_sin)  # (batch, 2*gru_hidden_size)

class JANETCell(nn.Module):
    """单层 JANET cell，接口模仿 nn.GRUCell。"""

    def __init__(self, input_size, hidden_size, gate_bias_init=-1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate_linear = nn.Linear(input_size + hidden_size, hidden_size)
        self.candidate_linear = nn.Linear(input_size + hidden_size, hidden_size)
        nn.init.constant_(self.gate_linear.bias, gate_bias_init)

    def forward(self, x, hc):
        h, c = hc
        z = torch.cat((h, x), dim=-1)
        i = torch.sigmoid(self.gate_linear(z))
        c = c + i * torch.tanh(self.candidate_linear(z))
        h = i * torch.tanh(c)
        return h, c


class JANET(nn.Module):
    """沿时间步循环的 JANET，输入输出形状与 nn.GRU(batch_first=True) 一致。"""

    def __init__(self, input_size, hidden_size, gate_bias_init=-1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = JANETCell(input_size, hidden_size, gate_bias_init)

    def forward(self, x, hc=None):
        batch_size, seq_len, _ = x.shape
        if hc is None:
            h = x.new_zeros(batch_size, self.hidden_size)
            c = x.new_zeros(batch_size, self.hidden_size)
        else:
            h, c = hc

        outputs = []
        for t in range(seq_len):
            h, c = self.cell(x[:, t, :], (h, c))
            outputs.append(h)
        return torch.stack(outputs, dim=1), (h, c)


class BeamJANET_RVT(nn.Module):
    """单波束 JANET-RVT：复数线性变换 → 幅度序列 → JANET → PRB。"""

    def __init__(self, n_input, n_hidden_1, janet_hidden_size):
        super().__init__()
        self.n_hidden_1 = n_hidden_1
        self.janet_hidden_size = janet_hidden_size

        # 第一层：复数线性变换
        self.hidden1 = nn.Linear(n_input, n_hidden_1)

        # JANET 处理幅度序列
        # 输入：(batch, seq_len=n_hidden_1//2, input_size=2) — [abs, abs²]
        self.janet = JANET(input_size=2, hidden_size=janet_hidden_size)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.hidden1.weight)
        nn.init.constant_(self.hidden1.bias, 0)

    def forward(self, x, context=None):
        # 第一层线性变换
        h1 = self.hidden1(x)
        if context is not None:
            h1 = h1 + context

        # 提取幅度和相位
        h1_real = h1[:, ::2]
        h1_imag = h1[:, 1::2]

        h1_abs = (h1_real ** 2 + h1_imag ** 2).pow(0.5)
        # h1_angle = torch.atan2(h1_real, h1_imag)

        # 构造幅度特征序列：[abs, abs²]
        h1_abs_pow = torch.stack([h1_abs, h1_abs ** 2], dim=-1)

        # JANET 处理幅度序列
        janet_out, _ = self.janet(h1_abs_pow)   # (batch, n_hidden_1//2, janet_hidden_size)
        janet_last = janet_out[:, -1, :]        # (batch, janet_hidden_size)

        # # PRB：结合幅度和相位
        # f_cos = torch.cos(h1_angle)
        # f_sin = torch.sin(h1_angle)

        return PRB(janet_last, h1_real, h1_imag)    # (batch, 2*janet_hidden_size)

class FCNNDPD(nn.Module):
    """BOTDNN 共享主干 + 早期跨波束上下文注入。"""

    def __init__(self, num_beam, memory_depth, hidden_beam, hidden_2):
        super().__init__()
        self.num_beam = num_beam
        delay_count = memory_depth + 1
        context_input_dim = 2 * num_beam * delay_count

        # ——— BOTDNN 共享主干（逐波束独立输入，共享权重）———
        self.beam_rvtbnn1 = BeamRVTBNN(
            n_input=context_input_dim,
            n_hidden_1=2 * hidden_beam,
            n_hidden_2=hidden_2,
        )
        # self.beam_rvtbnn2 = BeamTBNN(
        #     n_input=2 * delay_count,
        #     n_hidden_1=2 * hidden_beam,
        #     n_hidden_2=hidden_2,
        # )
        # self.beam_rvtbnn2 = BeamGRU_RVT(
        #     n_input=2 * delay_count,
        #     n_hidden_1=14,
        #     gru_hidden_size=10,
        # )

        # self.beam_gru_rvt = BeamGRU_RVT(
        #     n_input=2 * delay_count,
        #     n_hidden_1=2 * hidden_beam,
        #     gru_hidden_size=hidden_2,
        # )

        # self.beam_janet_rvt = BeamJANET_RVT(
        #     n_input=2 * delay_count,
        #     n_hidden_1=2 * hidden_beam,
        #     janet_hidden_size=hidden_2,
        # )

        # ——— 逐波束输出 Linear ———
        self.beam_output = nn.ModuleList([
            nn.Linear(2 * hidden_2, 2)
            for _ in range(num_beam)
        ])
        # # self.beam_output = nn.ModuleList([
        #    nn.Linear(4 * hidden_2 + 2 * hidden_beam, 2)
        #    for _ in range(num_beam)
        # ])

        # self.context_mlp2 = nn.Sequential(
        #     nn.Linear(context_input_dim, 2 * delay_count),
        #     # nn.Tanh()
        # )
        # self.context_mlp2 = nn.ModuleList([
        #     nn.Linear(context_input_dim, 2 * delay_count)
        #     for _ in range(num_beam)
        # ])

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, delayed_beam_complex):
        batch_size = delayed_beam_complex.shape[0]

        beam_real = torch.stack(
            [delayed_beam_complex.real, delayed_beam_complex.imag], dim=-1
        )  # (B, Q, T, 2)

        all_beams_flat = beam_real.reshape(batch_size, -1)
        # context2 = self.context_mlp2(all_beams_flat)
        # intermod_out = self.beam_janet_rvt(context2, context=None)
        intermod_out = self.beam_rvtbnn1(all_beams_flat, context=None)
        
        outputs = []
        for q in range(self.num_beam):
            # beam_q = beam_real[:, q, :, :].reshape(batch_size, -1)
            # rvtbnn_out = self.beam_rvtbnn1(beam_q, context=None)
            # rvtbnn_out = self.beam_janet_rvt(beam_q, context=None)
            # intermod_out = self.beam_janet_rvt(context2, context=None)
            # linear_input = torch.cat((rvtbnn_out, intermod_out), 1)
            # linear_input = intermod_out + rvtbnn_out
            # linear_input = torch.cat((rvtbnn_out, intermod_out), 1)
            outputs.append(self.beam_output[q](intermod_out))
        return torch.stack(outputs, dim=1)

