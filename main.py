"""训练 FC-NN-DPD，并在独立测试集上计算 NMSE。"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# from data import prepare_data
from data import prepare_data
# from data_val_first import prepare_data
from model import FCNNDPD


# ============================ 可修改配置 ============================
RANDOM_SEED = 42
DATA_FOLDER = Path(r"E:\paper\MIMO-DPD\ABA\data\BEAM4_BW_40")

# 数据划分：按时间连续切分，验证集在前、训练集居中、测试集在后。
MEMORY_DEPTH = 7
TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.2

# 波束域模型结构参数。
HIDDEN_BEAM = 9
HIDDEN_2 = 25

# 训练参数。
BATCH_SIZE = 128
LEARNING_RATE = 1e-2
MAXIMUM_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 30
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 10
GRADIENT_CLIP_MAX_NORM = 1.0

MODEL_SAVE_PATH = Path(__file__).resolve().parent / "fc_nn_1_dpd_best.pth"


def count_trainable_parameters(model):
    """统计模型中 requires_grad=True 的参数数量。"""
    return sum(parameter.numel() for parameter in model.parameters())


def calculate_nmse(true_signal, predicted_signal):
    """计算复数信号的 NMSE，单位为 dB。"""
    error_power = np.sum(np.abs(true_signal - predicted_signal) ** 2)
    signal_power = np.sum(np.abs(true_signal) ** 2)
    return 10 * np.log10(error_power / signal_power)


def train_model(
    model,
    train_loader,
    delayed_beam_complex,
    target_real_imag,
    train_indices,
    validation_indices,
    save_path,
):
    """训练模型，并根据验证集损失保存最佳权重。"""
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=LR_REDUCE_FACTOR,
        patience=LR_REDUCE_PATIENCE,
    )

    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(MAXIMUM_EPOCHS):
        model.train()
        total_train_loss = 0.0

        # 只使用训练集更新网络参数。
        for batch_beam, batch_target in train_loader:
            optimizer.zero_grad()
            prediction = model(batch_beam)
            train_loss = loss_function(prediction, batch_target)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM
            )
            optimizer.step()
            total_train_loss += train_loss.item() * batch_beam.size(0)

        average_train_loss = total_train_loss / len(train_indices)

        # 验证集只用于选择模型、调节学习率和早停，不参与参数更新。
        model.eval()
        with torch.no_grad():
            validation_prediction = model(
                delayed_beam_complex[validation_indices],
            )
            validation_loss = loss_function(
                validation_prediction,
                target_real_imag[validation_indices],
            ).item()

        scheduler.step(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch + 1}/{MAXIMUM_EPOCHS}]  "
                f"Train: {average_train_loss:.8f}  "
                f"Validation: {validation_loss:.8f}  "
                f"LR: {current_lr:.6f}"
            )

        # if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        #     print(f"Early stopping after {epoch + 1} epochs")
        #     break

    print(f"Best validation loss: {best_validation_loss:.8f}")


def evaluate_model(
    model,
    delayed_beam_complex,
    target_real_imag,
    test_indices,
    model_path,
    beam_count,
):
    """加载验证集选出的最佳模型，只在测试集上进行最终评估。"""
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        # 对全部有效样本进行推理，用于计算整体 NMSE。
        all_predicted_real_imag = model(delayed_beam_complex).numpy()
        # 保留测试集推理结果，用于计算 Test NMSE。
        test_predicted_real_imag = all_predicted_real_imag[test_indices]

    all_predicted_signal = (
        all_predicted_real_imag[:, :, 0]
        + 1j * all_predicted_real_imag[:, :, 1]
    )
    all_true_signal = (
        target_real_imag[:, :, 0].numpy()
        + 1j * target_real_imag[:, :, 1].numpy()
    )
    test_predicted_signal = (
        test_predicted_real_imag[:, :, 0]
        + 1j * test_predicted_real_imag[:, :, 1]
    )
    test_true_signal = all_true_signal[test_indices]

    print("\n=== All Data NMSE ===")
    for beam in range(beam_count):
        nmse = calculate_nmse(
            all_true_signal[:, beam],
            all_predicted_signal[:, beam],
        )
        print(f"Beam {beam + 1}: {nmse:.2f} dB")
    print(
        f"Overall: {calculate_nmse(all_true_signal, all_predicted_signal):.2f} dB"
    )

    print("\n=== Test NMSE ===")
    for beam in range(beam_count):
        nmse = calculate_nmse(
            test_true_signal[:, beam],
            test_predicted_signal[:, beam],
        )
        print(f"Beam {beam + 1}: {nmse:.2f} dB")
    print(
        f"Overall: {calculate_nmse(test_true_signal, test_predicted_signal):.2f} dB"
    )


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    data = prepare_data(
        data_folder=DATA_FOLDER,
        memory_depth=MEMORY_DEPTH,
        train_ratio=TRAIN_RATIO,
        validation_ratio=VALIDATION_RATIO,
    )

    print("=== FC-NN-DPD (BOTDNN + Early Context Injection) ===")
    print(f"Beam count: {data['beam_count']}")
    print(f"Memory depth: {MEMORY_DEPTH}")
    print(f"Hidden beam: {HIDDEN_BEAM}, Hidden 2: {HIDDEN_2}")
    print(f"Samples: train={data['train_sample_count']}, "
          f"validation={data['validation_sample_count']}, "
          f"test={data['test_sample_count']}")

    model = FCNNDPD(
        num_beam=data["beam_count"],
        memory_depth=MEMORY_DEPTH,
        hidden_beam=HIDDEN_BEAM,
        hidden_2=HIDDEN_2,
    )
    print(f"Trainable parameters: {count_trainable_parameters(model)}")

    delayed_beam_complex = data["delayed_beam_complex"]
    target_real_imag = data["target_real_imag"]
    train_indices = data["train_indices"]
    validation_indices = data["validation_indices"]
    test_indices = data["test_indices"]

    train_dataset = TensorDataset(
        delayed_beam_complex[train_indices],
        target_real_imag[train_indices],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    train_model(
        model=model,
        train_loader=train_loader,
        delayed_beam_complex=delayed_beam_complex,
        target_real_imag=target_real_imag,
        train_indices=train_indices,
        validation_indices=validation_indices,
        save_path=MODEL_SAVE_PATH,
    )

    evaluate_model(
        model=model,
        delayed_beam_complex=delayed_beam_complex,
        target_real_imag=target_real_imag,
        test_indices=test_indices,
        model_path=MODEL_SAVE_PATH,
        beam_count=data["beam_count"],
    )


if __name__ == "__main__":
    main()
