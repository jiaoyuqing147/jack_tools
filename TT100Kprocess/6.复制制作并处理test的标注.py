# -*- coding: utf-8 -*-
import os
import pandas as pd

# ===== 路径配置（按需修改）=====
stats_csv      = r"class_distribution_AfterDivied.csv"  # 含列: class_id, train_count, val_count
src_test_dir   = r"E:\Datasets\tt100k_2021\labels_all\test"
dst_test_dir   = r"E:\Datasets\tt100k_2021\yolojack\labels\test"
# =================================

def load_old2new(stats_csv_path):
    df = pd.read_csv(stats_csv_path)
    need_cols = {"class_id", "train_count", "val_count"}
    if not need_cols.issubset(df.columns):
        raise ValueError(f"{stats_csv_path} 必须包含列：{need_cols}")
    keep_ids = (
        df[(df["train_count"] > 0) & (df["val_count"] > 0)]
        ["class_id"].astype(int).sort_values().tolist()
    )
    old2new = {old: new for new, old in enumerate(keep_ids)}
    return old2new, keep_ids

def remap_one_file(src_path, old2new_map):
    """读取一个txt，返回重映射后的有效行列表（可能为空）"""
    out_lines = []
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            # 解析首列为旧类ID
            try:
                old_id = int(float(parts[0]))
            except:
                continue
            # 丢弃不在保留集合的类别
            if old_id not in old2new_map:
                continue
            # 替换为新ID
            parts[0] = str(old2new_map[old_id])
            out_lines.append(" ".join(parts))
    return out_lines

def main():
    if not os.path.isdir(src_test_dir):
        raise FileNotFoundError(f"未找到测试标签目录：{src_test_dir}")
    os.makedirs(dst_test_dir, exist_ok=True)

    old2new, keep_ids = load_old2new(stats_csv)
    print(f"保留类别数（新 nc）：{len(keep_ids)}")
    if len(keep_ids) == 0:
        print("⚠️ 保留集合为空，停止处理。请检查统计CSV。")
        return

    total_files = 0
    written_files = 0
    skipped_empty = 0
    orig_lines_sum = 0
    kept_lines_sum = 0

    for fn in os.listdir(src_test_dir):
        if not fn.endswith(".txt"):
            continue
        total_files += 1
        src_fp = os.path.join(src_test_dir, fn)
        out_lines = remap_one_file(src_fp, old2new)
        # 统计原始行数（用于摘要）
        with open(src_fp, "r", encoding="utf-8") as f:
            o_lines = [ln for ln in (x.strip() for x in f) if ln]
            orig_lines_sum += len(o_lines)

        if out_lines:
            dst_fp = os.path.join(dst_test_dir, fn)
            with open(dst_fp, "w", encoding="utf-8") as f:
                f.write("\n".join(out_lines) + "\n")
            kept_lines_sum += len(out_lines)
            written_files += 1
        else:
            skipped_empty += 1
            # 若目标处已有同名旧文件，可选择删除（确保 yolojack\test 中仅保留有效文件）
            dst_fp = os.path.join(dst_test_dir, fn)
            if os.path.exists(dst_fp):
                try:
                    os.remove(dst_fp)
                except:
                    pass

    print("\n===== Test → yolojack/test 重映射完成 =====")
    print(f"总测试标签文件：{total_files}")
    print(f"写入到 yolojack/test 的文件：{written_files}")
    print(f"因无保留标注而跳过：{skipped_empty}")
    print(f"原始总标注框数：{orig_lines_sum}")
    print(f"保留总标注框数：{kept_lines_sum}")
    print(f"输出目录：{dst_test_dir}")

if __name__ == "__main__":
    main()
