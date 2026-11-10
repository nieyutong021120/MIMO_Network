# MIMO_Network

- 提出了一种适用于大规模 MIMO 发射机的流级 DPD 方法，该方法的线性化对象是由阵列发送的波束信号，有效解决 DPD 中由于 RF 链数目增长带来的高计算复杂度问题。
- 结合对 MIMO 系统非线性失真的分析，设计了一种双分支“共享主干+轻量微调”的网络结构，通过跨波束特征提取层构造交调特征，分别建模了目标波束自作用分量和跨波束干扰分量，能在较低的复杂度下实现更高的建模性能。

## 目录结构与文件说明

| 文件 / 目录 | 功能说明 |
|---|---|
| `model.py` | 网络结构定义：包含 PRB 相位恢复、单波束 RVTBNN 及 GRU/JANET 等可选主干；主类 `FCNNDPD` 实现“共享主干 + 双分支”，分别输出目标波束自作用分量 `rvtbnn_out` 与跨波束干扰分量 `intermod_out`，并逐波束线性读出 |
| `data.py` | 数据准备：加载 `X/U/W_pre/W_syn`，构造复数波束域延迟特征与实虚部标签，按“训练 → 验证 → 测试”的连续时间顺序划分数据集 |
| `data_val_first.py` | 与 `data.py` 函数接口完全一致，仅切分顺序改为“验证 → 训练 → 测试”（验证集在前），可直接替换使用 |
| `main.py` | 训练与评估入口：配置数据路径、记忆深度、网络宽度及训练超参，训练网络并按验证损失保存最佳权重，最终输出全数据/测试集的逐波束及整体 NMSE |
| `plot_u_amplitude.py` | 读取 `U` 绘制各波束逐样本幅度 `|U|` 散点图，支持 `-d`（数据目录）、`-o`（输出路径）命令行参数 |
| `remove_u_outliers.py` | 按 `|U|` 幅度阈值剔除离群样本，生成仅包含 `X/U` 的新数据文件夹 |
| `fc_nn_1_dpd_best.pth` | 训练过程中按验证集损失保存的最佳模型权重 |
| `requirements.txt` | 运行所需的 Python 依赖清单 |

## 环境准备

1. 准备 Python 3.10 环境（推荐使用 conda 或 venv 隔离）：

   ```bash
   conda create -n mimo-dpd python=3.10
   conda activate mimo-dpd
   ```

2. 在项目根目录下运行 `requirements.txt` 完成依赖安装：

   ```bash
   pip install -r requirements.txt
   ```

   该清单包含 `torch`、`numpy`、`matplotlib`，默认安装 CPU 版本，已在 Windows 环境验证可用。若需 GPU 加速，请先按 PyTorch 官网指引安装与本机 CUDA 匹配的 torch，再安装其余依赖。

## 数据准备

`main.py` 中 `DATA_FOLDER` 指向的数据集目录需包含以下文件：

- `X.npy`、`U.npy`：形状均为 `(样本数, 波束数)` 的复数信号，分别为模型输入与目标输出；
- `W_pre.npy`、`W_syn.npy`：发射预编码与空间合成矩阵。

## 运行方式

1. **训练与评估**：先在 `main.py` 顶部修改 `DATA_FOLDER`、`MEMORY_DEPTH`、网络宽度等配置，再执行：

   ```bash
   python main.py
   ```

   训练结束后会自动加载最佳权重，打印逐波束及整体的全数据/测试集 NMSE。

2. **绘制幅度散点图**：

   ```bash
   python plot_u_amplitude.py -d <数据集目录> -o <输出图片路径>
   ```

3. **剔除离群样本**：先在脚本内设置数据路径与幅度阈值，再执行：

   ```bash
   python remove_u_outliers.py
   ```
