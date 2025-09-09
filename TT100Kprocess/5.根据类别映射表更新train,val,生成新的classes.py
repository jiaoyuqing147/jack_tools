# -*- coding: utf-8 -*-
"""
将标签中的旧ID重映射为新ID，并删除不在保留集合中的标注行；
若文件被删空则删除该txt；同时生成新的 classes.txt（无引号）。

保留规则：仅保留在统计表中 train_count>0 且 val_count>0 的类别。
"""

import os
import json
import pandas as pd

# ========== 路径配置（按需修改） ==========
train_label_dir = r"E:\Datasets\tt100k_2021\yolojack\labels\train"
val_label_dir   = r"E:\Datasets\tt100k_2021\yolojack\labels\val"
stats_csv       = r"class_distribution_AfterDivied.csv"  # 你的三列表统计CSV
classes_txt_out = r"E:\Datasets\tt100k_2021\yolojack\labels\classes.txt"
# =======================================

# 原始 232 类 names（index 对应 old_id）
names = [
  "pl80","w9","p6","ph4.2","i8","w14","w33","pa13","im","w58","pl90","il70","p5","pm55","pl60","ip",
  "p11","pdd","wc","i2r","w30","pmr","p23","pl15","pm10","pss","w1","p4","w38","w50","w34","pw3.5","iz",
  "w39","w11","p1n","pr70","pd","pnl","pg","ph5.3","w66","il80","pb","pbm","pm5","w24","w67","w49","pm40",
  "ph4","w45","i4","w37","ph2.6","pl70","ph5.5","i14","i11","p7","p29","pne","pr60","pm13","ph4.5","p12",
  "p3","w40","pl5","w13","pr10","p14","i4l","pr30","pw4.2","w16","p17","ph3","i9","w15","w35","pa8","pt",
  "pr45","w17","pl30","pcs","pctl","pr50","ph4.4","pm46","pm35","i15","pa12","pclr","i1","pcd","pbp",
  "pcr","w28","ps","pm8","w18","w2","w52","ph2.9","ph1.8","pe","p20","w36","p10","pn","pa14","w54","ph3.2",
  "p2","ph2.5","w62","w55","pw3","pw4.5","i12","ph4.3","phclr","i10","pr5","i13","w10","p26","w26","p8",
  "w5","w42","il50","p13","pr40","p25","w41","pl20","ph4.8","pnlc","ph3.3","w29","ph2.1","w53","pm30",
  "p24","p21","pl40","w27","pmb","pc","i6","pr20","p18","ph3.8","pm50","pm25","i2","w22","w47","w56",
  "pl120","ph2.8","i7","w12","pm1.5","pm2.5","w32","pm15","ph5","w19","pw3.2","pw2.5","pl10","il60",
  "w57","w48","w60","pl100","pr80","p16","pl110","w59","w64","w20","ph2","p9","il100","w31","w65","ph2.4",
  "pr100","p19","ph3.5","pa10","pcl","pl35","p15","w7","pa6","phcs","w43","p28","w6","w3","w25","pl25",
  "il110","p1","w46","pn-2","w51","w44","w63","w23","pm20","w8","pmblr","w4","i5","il90","w21","p27",
  "pl50","pl65","w61","ph2.2","pm2","i3","pa18","pw4"
]

def load_valid_classes(stats_csv_path):
    df = pd.read_csv(stats_csv_path)
    if not {"class_id","train_count","val_count"}.issubset(df.columns):
        raise ValueError("统计CSV需要包含列：class_id, train_count, val_count")
    valid = df[(df["train_count"] > 0) & (df["val_count"] > 0)]["class_id"].astype(int).tolist()
    return sorted(valid), df

def build_maps(valid_ids, names_all):
    new_names = [names_all[i] for i in valid_ids]
    old_to_new = {old: new for new, old in enumerate(valid_ids)}
    return new_names, old_to_new

def remap_one_file(txt_path, old_to_new):
    """返回(保留下来的行列表, 原行数, 新行数)。若新行数=0，调用处删除文件。"""
    kept = []
    orig, newc = 0, 0
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            orig += 1
            parts = s.split()
            try:
                old_id = int(float(parts[0]))
            except:
                continue
            if old_id not in old_to_new:
                continue
            parts[0] = str(old_to_new[old_id])
            kept.append(" ".join(parts))
            newc += 1
    return kept, orig, newc

def process_dir(label_dir, old_to_new):
    n_files = 0
    n_deleted_files = 0
    n_lines_before = 0
    n_lines_after  = 0
    for fn in os.listdir(label_dir):
        if not fn.endswith(".txt"):
            continue
        n_files += 1
        fp = os.path.join(label_dir, fn)
        kept, orig, newc = remap_one_file(fp, old_to_new)
        n_lines_before += orig
        n_lines_after  += newc
        if newc == 0:
            os.remove(fp)
            n_deleted_files += 1
        else:
            with open(fp, "w", encoding="utf-8") as f:
                f.write("\n".join(kept) + "\n")
    return {
        "files_seen": n_files,
        "files_deleted": n_deleted_files,
        "lines_before": n_lines_before,
        "lines_after": n_lines_after,
        "lines_removed": n_lines_before - n_lines_after
    }

def main():
    # 1) 确定保留类与映射
    valid_ids, df_stats = load_valid_classes(stats_csv)
    new_names, old_to_new = build_maps(valid_ids, names)

    print("===== 重映射计划 =====")
    print(f"保留类别数：{len(valid_ids)} / 232")
    print(f"示例 old->new（前20）：{list(old_to_new.items())[:20]}")
    print("\n===== 新 names（无引号） =====")
    for n in new_names:
        print(n)

    # 2) 执行重写与清理
    print("\n===== 处理 train 目录 =====")
    r_train = process_dir(train_label_dir, old_to_new)
    print(r_train)

    print("\n===== 处理 val 目录 =====")
    r_val = process_dir(val_label_dir, old_to_new)
    print(r_val)

    # 3) 写 classes.txt（无引号，一行一个）
    os.makedirs(os.path.dirname(classes_txt_out), exist_ok=True)
    with open(classes_txt_out, "w", encoding="utf-8") as f:
        for n in new_names:
            f.write(f"{n}\n")
    print(f"\n[OK] 新 classes.txt 已生成：{classes_txt_out}")
    print(f"[INFO] 新 nc = {len(new_names)}")

    # 4)（可选）打印摘要
    total_removed = r_train["lines_removed"] + r_val["lines_removed"]
    total_files_deleted = r_train["files_deleted"] + r_val["files_deleted"]
    print("\n===== 摘要 =====")
    print(f"总删掉的无效标注行：{total_removed}")
    print(f"被删除的空txt文件：{total_files_deleted}")

    # 5)（可选）若需要追溯，打印/保存映射
    remap_preview = {k: old_to_new[k] for k in list(old_to_new)[:20]}
    print("\n映射预览（前20项）：", remap_preview)
    # 如果你想保存映射，取消下面注释：
    # with open(os.path.join(os.path.dirname(classes_txt_out), "old_to_new.json"), "w", encoding="utf-8") as f:
    #     json.dump(old_to_new, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
