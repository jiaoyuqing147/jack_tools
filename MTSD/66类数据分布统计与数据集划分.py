import pandas as pd
import os

# ==== 配置路径 ====
gt_file = 'GT.txt'
gt_cols = ['File Name', 'X', 'Y', 'Width', 'Height', 'Sign Type', 'Sign Group', 'Sign Class',
           'TS Class', 'TS ID', 'TS Color', 'Shape', 'Shape ID', 'Lightning', 'Image Source']

# ==== 读取标注 ====
gt_df = pd.read_csv(gt_file, delimiter=';', encoding='latin1', header=None, names=gt_cols)
gt_df['File Name'] = gt_df['File Name'].str.strip().str.strip("'").str.lower()

# ==== 统计66类分布 ====
ts_id_counts = gt_df['TS ID'].value_counts().sort_index()

print("📊 66类细粒度类别的分布：")
for ts_id, count in ts_id_counts.items():
    print(f"类别 {ts_id}: {count} 个")

# 可选：保存统计结果
output_csv = 'ts_id_distribution.csv'
ts_id_counts.to_csv(output_csv, header=['Count'])
print(f"\n✅ 66类类别分布已保存至 {output_csv}")
