"""FC-NN-DPD 数据准备（验证集在前的时间顺序划分）。

与 data.py 的唯一区别是切分顺序：按连续时间划分时，
最前面的 validation_ratio 部分作为验证集，中间 train_ratio 部分
作为训练集，剩余部分作为测试集，即 验证 -> 训练 -> 测试。
函数签名与返回字典的键均与 data.py 保持一致，可直接替换使用。
"""

from pathlib import Path

import numpy as np
import torch


def prepare_data(
    data_folder,
    memory_depth=6,
    train_ratio=0.4,
    validation_ratio=0.3,
):
    """加载数据、构造延迟特征，并按 验证-训练-测试 的时间顺序划分数据集。"""
    data_folder = Path(data_folder)

    # B、U 的形状均为 (样本数, 波束数)，元素为复数。
    X = np.load(data_folder / "X.npy")
    U = np.load(data_folder / "U.npy")
    W_pre = np.load(data_folder / "W_pre.npy")
    W_syn = np.load(data_folder / "W_syn.npy")

    # 归一化：输入按峰值归一化；标签先按自身能量归一化，再对齐到输入的功率。
    # input_peak = np.max(np.abs(X))
    # X = X / input_peak
    # X_norm = np.linalg.norm(X)
    # U = U / np.linalg.norm(U) * X_norm

    sample_count, beam_count = X.shape
    num_ant = W_pre.shape[1]

    valid_count = sample_count - memory_depth

    # # 固定输入层和输出层的权重，不参与梯度更新。
    # input_weight = np.linalg.pinv(W_syn.conj().T)   # (天线数, 波束数)
    # output_weight = np.linalg.pinv(W_pre.conj().T)  # (波束数, 天线数)

    # 直接保存复数形式的波束域延迟输入：
    # [B[n], B[n-1], ..., B[n-memory_depth]]。
    delayed_beam_complex = np.stack(
        [X[memory_depth - delay: sample_count - delay]
         for delay in range(memory_depth + 1)],
        axis=2,
    )

    # 目标 U[n] 与当前输入 B[n] 对齐，因此去除前 memory_depth 个样本。
    target_real_imag = np.stack(
        (U[memory_depth:].real, U[memory_depth:].imag), axis=-1
    ).astype(np.float32)

    # 时间序列按时间连续切分，避免相邻延迟窗口同时进入训练和测试。
    # 切分顺序：验证集在前，训练集居中，测试集在后。
    validation_count = int(validation_ratio * valid_count)
    train_count = int(train_ratio * valid_count)
    all_indices = np.arange(valid_count, dtype=np.int64)
    validation_indices = all_indices[:validation_count]
    train_indices = all_indices[validation_count:validation_count + train_count]
    test_indices = all_indices[validation_count + train_count:]

    return {
        "beam_count": beam_count,
        "num_ant": num_ant,
        "sample_count": sample_count,
        "valid_sample_count": valid_count,
        "delayed_beam_complex": torch.tensor(
            delayed_beam_complex, dtype=torch.complex64
        ),
        "target_real_imag": torch.tensor(target_real_imag),
        # "input_weight": torch.tensor(input_weight, dtype=torch.complex64),
        # "output_weight": torch.tensor(output_weight, dtype=torch.complex64),
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
        "train_sample_count": train_count,
        "validation_sample_count": validation_count,
        "test_sample_count": len(test_indices),
        "single_pa_input_dim": 2 * (memory_depth + 1),
        "memory_depth": memory_depth,
    }
