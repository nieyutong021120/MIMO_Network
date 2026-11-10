# -*- coding: utf-8 -*-
"""
复现 fc_nn_1_dpd_best.pth 的真实结构（共享 BeamJANET_RVT + 两支路逐元素相加），
量化 rvtbnn_out(自作用) 与 intermod_out(跨波束) 的输出能量，并复现两种消融 NMSE。
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model import BeamJANET_RVT
from data_val_first import prepare_data

DATA_FOLDER = Path(r"E:\paper\MIMO-DPD\ABA\data\BEAM2_BW_60")
MEMORY_DEPTH = 6
Q, T = 2, MEMORY_DEPTH + 1
JANET_HIDDEN = 14
CKPT = "fc_nn_1_dpd_best.pth"


class LoadedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.beam_janet_rvt = BeamJANET_RVT(
            n_input=2 * T, n_hidden_1=8, janet_hidden_size=JANET_HIDDEN
        )
        self.context_mlp2 = nn.Sequential(nn.Linear(2 * Q * T, 2 * T))
        self.beam_output = nn.ModuleList(
            [nn.Linear(2 * JANET_HIDDEN, 2) for _ in range(Q)]
        )

    def branch_outputs(self, delayed_beam_complex):
        b = delayed_beam_complex.shape[0]
        beam_real = torch.stack(
            [delayed_beam_complex.real, delayed_beam_complex.imag], dim=-1
        )  # (B,Q,T,2)
        all_flat = beam_real.reshape(b, -1)
        context = self.context_mlp2(all_flat)
        intermod = self.beam_janet_rvt(context, context=None)

        per_beam = {}
        for q in range(Q):
            beam_q = beam_real[:, q, :, :].reshape(b, -1)
            rvt = self.beam_janet_rvt(beam_q, context=None)
            per_beam[q] = (rvt, intermod)
        return per_beam


def nmse_db(pred_ri, target_ri):
    p = pred_ri[..., 0] + 1j * pred_ri[..., 1]
    t = target_ri[..., 0] + 1j * target_ri[..., 1]
    return 10 * np.log10(np.sum(np.abs(p - t) ** 2) / np.sum(np.abs(t) ** 2))


def main():
    data = prepare_data(
        DATA_FOLDER, memory_depth=MEMORY_DEPTH,
        train_ratio=0.6, validation_ratio=0.2,
    )
    x = data["delayed_beam_complex"]
    target = data["target_real_imag"]

    model = LoadedModel()
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        branches = model.branch_outputs(x)

        rvt_all, int_all = [], []
        pred_full, pred_no_int, pred_no_rvt = [], [], []
        for q in range(Q):
            rvt, inter = branches[q]
            rvt_all.append(rvt)
            int_all.append(inter)
            pred_full.append(model.beam_output[q](rvt + inter))
            pred_no_int.append(model.beam_output[q](rvt))      # 去掉 intermod
            pred_no_rvt.append(model.beam_output[q](inter))     # 去掉 rvtbnn

        rvt_cat = torch.cat([r.reshape(-1) for r in rvt_all])
        int_cat = torch.cat([r.reshape(-1) for r in int_all])
        e_rvt = rvt_cat.pow(2).mean().item()
        e_int = int_cat.pow(2).mean().item()

        print("=== 支路输出能量（全数据，特征空间）===")
        print(f"rvtbnn(自作用)  RMS = {np.sqrt(e_rvt):.5f}  能量占比 = {e_rvt/(e_rvt+e_int):.3f}")
        print(f"intermod(跨波束) RMS = {np.sqrt(e_int):.5f}  能量占比 = {e_int/(e_rvt+e_int):.3f}")
        print(f"能量比 intermod/rvtbnn = {e_int/e_rvt:.2f}")

        print("\n=== 消融 NMSE 复现（整体，dB，越低越好）===")
        print(f"完整模型            : {nmse_db(torch.stack(pred_full,1).numpy(), target.numpy()):7.2f} dB")
        print(f"去掉 intermod_out   : {nmse_db(torch.stack(pred_no_int,1).numpy(), target.numpy()):7.2f} dB")
        print(f"去掉 rvtbnn_out     : {nmse_db(torch.stack(pred_no_rvt,1).numpy(), target.numpy()):7.2f} dB")


if __name__ == "__main__":
    main()
