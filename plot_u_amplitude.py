"""读取原始数据集中的 U，计算每个样本点的幅度并绘制散点图。

U 的形状为 (样本数, 波束数)，每个样本点在每个波束上各有一个复数值，
这里画每个波束的幅度 |U|，横轴为样本序号。

用法：
    python plot_u_amplitude.py                        # 默认 BEAM2_BW_60
    python plot_u_amplitude.py -d <数据集目录>
    python plot_u_amplitude.py -o <输出png路径>
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_DATA_FOLDER = Path(r"E:\paper\MIMO-DPD\ABA\data\BEAM4_BW_60")


def main():
    parser = argparse.ArgumentParser(description="绘制 U 的逐样本幅度散点图")
    parser.add_argument("-d", "--data-folder", type=Path,
                        default=DEFAULT_DATA_FOLDER, help="数据集目录")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="输出图片路径，默认放在数据集目录下")
    args = parser.parse_args()

    target = np.load(args.data_folder / "U.npy")
    amplitude = np.abs(target)               # (样本数, 波束数)
    sample_index = np.arange(len(amplitude))
    beam_count = amplitude.shape[1]

    output = args.output or (
        Path(__file__).resolve().parent
        / f"U_amplitude_scatter_{args.data_folder.name}.png"
    )

    colors = plt.get_cmap("tab10").colors
    figure, axes = plt.subplots(figsize=(14, 5))
    for beam in range(beam_count):
        axes.scatter(
            sample_index, amplitude[:, beam],
            s=1.5, alpha=0.35, color=colors[beam % len(colors)],
            label=f"beam {beam + 1}",
            rasterized=True,
        )

    axes.set_xlabel("sample index")
    axes.set_ylabel("amplitude  |U|")
    axes.set_title(f"Per-sample amplitude of U   ({args.data_folder.name})")
    axes.set_xlim(0, len(amplitude) - 1)
    axes.grid(alpha=0.3)
    axes.legend(markerscale=8, loc="upper right")
    figure.tight_layout()
    figure.savefig(output, dpi=160)

    print(f"U: shape={target.shape}, dtype={target.dtype}")
    for beam in range(beam_count):
        value = amplitude[:, beam]
        print(f"  beam {beam + 1}: mean={value.mean():.4f}  "
              f"std={value.std():.4f}  max={value.max():.4f}  "
              f"p99.9={np.percentile(value, 99.9):.4f}")
    print(f"已保存图像: {output}")


if __name__ == "__main__":
    main()
