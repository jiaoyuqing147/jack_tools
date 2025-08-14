import os
import pandas as pd

# ==== 路径配置 ====
work_dir = r'F:/jack_dataset/MTSD'
mapping_csv = os.path.join(work_dir, 'class_id_mapping.csv')   # columns: old_id,new_id (0-based)
classes66_txt = r'classes66.txt'     # 66类全称，每行一个，对应 ClassID=1..66
out_txt = os.path.join(work_dir, 'yolo54/labels/classes.txt')              # 目标：纯txt，每行一个类别名

# ==== 读取映射 ====
df = pd.read_csv(mapping_csv)
# 兼容可能的列名
cols = {c.lower(): c for c in df.columns}
old_col = cols.get('old_id', 'old_id')
new_col = cols.get('new_id', 'new_id')
old2new = dict(zip(df[old_col].astype(int), df[new_col].astype(int)))

# ==== 读取66类名称（index: 0..65 对应 ClassID 1..66）====
with open(classes66_txt, 'r', encoding='utf-8') as f:
    all_names = [ln.strip() for ln in f if ln.strip()]

# ==== 生成按 new_id 排序的54类名称 ====
pairs = [(int(new_id), all_names[int(old_id)]) for old_id, new_id in old2new.items()]
pairs.sort(key=lambda x: x[0])
names54 = [name for _, name in pairs]

# ==== 写出纯txt ====
with open(out_txt, 'w', encoding='utf-8') as f:
    f.write('\n'.join(names54))
    f.write('\n')

print(f'✅ 已写出: {out_txt}')
print(f'类别数(应=54): {len(names54)}')
