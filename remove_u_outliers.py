"""按 |U| 幅度阈值剔除离群样本，生成新的数据文件夹（只保存 X 和 U）。"""

from pathlib import Path

import numpy as np

DATA_FOLDER = Path(r"E:\paper\MIMO-DPD\ABA\data\BEAM3_BW_40")
OUTPUT_FOLDER = DATA_FOLDER.parent / f"{DATA_FOLDER.name}_u3"
AMPLITUDE_THRESHOLD = 3.0


def main():
    X = np.load(DATA_FOLDER / "X.npy")
    U = np.load(DATA_FOLDER / "U.npy")

    # 任一波束的 |U| >= 3 即视为离群点。
    keep_mask = np.abs(U).max(axis=1) < AMPLITUDE_THRESHOLD
    removed_indices = np.flatnonzero(~keep_mask)

    print(f"样本总数: {len(U)}")
    print(f"剔除样本: {len(removed_indices)}")
    print(f"保留样本: {int(keep_mask.sum())}")
    print(f"剔除索引: {removed_indices}")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_FOLDER / "X.npy", X[keep_mask])
    np.save(OUTPUT_FOLDER / "U.npy", U[keep_mask])
    print(f"已保存: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
