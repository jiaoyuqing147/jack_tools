import os
import pandas as pd

# ======= 配置区 =======
class_counts_csv = 'class_id_counts.csv'                      # 你统计得到的CSV（1-based Class ID, 列名为 count）
labels_dir_in    = r'F:/jack_dataset/MTSD/labels_yolo66'      # 原始 YOLO 标签目录（0-based id, 0..65）
work_dir         = r'F:/jack_dataset/MTSD'                    # 输出的上级目录
names_66_txt     = None  # 可选：66类名称txt（每行一个名字，顺序与Class ID 1..66一致）。没有就保持None

# 输出目录与文件
labels_dir_filtered = os.path.join(work_dir, 'labels_pruned_filtered')   # 仅删除≤3类后的标签
labels_dir_remap    = os.path.join(work_dir, 'labels_pruned_remapped')   # 进一步重映射后的标签
mapping_csv         = os.path.join(work_dir, 'class_id_mapping.csv')     # old_id -> new_id
kept_ids_txt        = os.path.join(work_dir, 'kept_class_ids_old.txt')   # 保留的旧ID（0-based，排序）
removed_ids_txt     = os.path.join(work_dir, 'removed_class_ids_old.txt')# 删除的旧ID（0-based，排序）
new_names_yaml      = os.path.join(work_dir, 'names_pruned.yaml')        # 可选产出名字列表（若提供 names_66_txt）

os.makedirs(labels_dir_filtered, exist_ok=True)
os.makedirs(labels_dir_remap, exist_ok=True)

# ======= 读取 class_id_counts.csv，确定低频类 =======
# 你的保存方式：class_counts.to_csv('class_id_counts.csv', header=['count'])
# -> 第一列是 index=Class ID(1..66)，第二列是 count
counts = pd.read_csv(class_counts_csv, index_col=0)
if 'count' not in counts.columns:
    # 兼容用户不同导出格式
    counts.columns = ['count']
counts.index = counts.index.astype(int)  # 1..66

low_freq_1based = counts[counts['count'] <= 3].index.tolist()
low_freq_0based = sorted([i - 1 for i in low_freq_1based])    # 转YOLO 0-based
print(f'将删除的类别（YOLO 0-based）：{low_freq_0based}')

# ======= 第一步：删除低频类（仅过滤，不改ID）=======
total_removed = 0
files_changed = 0

for fname in os.listdir(labels_dir_in):
    if not fname.endswith('.txt'):
        continue
    src = os.path.join(labels_dir_in, fname)
    with open(src, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    new_lines = []
    removed = 0
    for ln in lines:
        parts = ln.split()
        try:
            cls_id = int(parts[0])
        except Exception:
            # 非法行，跳过
            continue
        if cls_id in low_freq_0based:
            removed += 1
        else:
            new_lines.append(ln)

    dst = os.path.join(labels_dir_filtered, fname)
    # 保留空文件（YOLO允许空标签文件）
    with open(dst, 'w') as f:
        f.write('\n'.join(new_lines))
        if new_lines:
            f.write('\n')

    if removed > 0:
        files_changed += 1
        total_removed += removed

print(f'✅ 删除完毕：修改 {files_changed} 个文件，共删除 {total_removed} 条标注')

# ======= 第二步：统计保留的旧ID，并构建 old->new 连续映射 =======
kept_old_ids = set()
for fname in os.listdir(labels_dir_filtered):
    if not fname.endswith('.txt'):
        continue
    with open(os.path.join(labels_dir_filtered, fname), 'r') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            cls_id = int(ln.split()[0])
            kept_old_ids.add(cls_id)

kept_old_ids = sorted(kept_old_ids)
removed_old_ids = sorted([i for i in range(66) if i not in kept_old_ids])

old_to_new = {old_id: new_id for new_id, old_id in enumerate(kept_old_ids)}

# 保存映射与列表
pd.DataFrame(
    [(k, v) for k, v in old_to_new.items()],
    columns=['old_id', 'new_id']
).to_csv(mapping_csv, index=False)

with open(kept_ids_txt, 'w') as f:
    f.write('\n'.join(map(str, kept_old_ids)))
with open(removed_ids_txt, 'w') as f:
    f.write('\n'.join(map(str, removed_old_ids)))

print(f'✅ 重映射表已保存：{mapping_csv}')
print(f'保留旧ID（0-based）：{kept_old_ids}')
print(f'删除旧ID（0-based）：{removed_old_ids}')
print(f'还剩类别数：{len(kept_old_ids)}')

# ======= 第三步：按映射重写标签 =======
files_changed_remap = 0
for fname in os.listdir(labels_dir_filtered):
    if not fname.endswith('.txt'):
        continue
    src = os.path.join(labels_dir_filtered, fname)
    with open(src, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    remapped_lines = []
    for ln in lines:
        parts = ln.split()
        try:
            old_id = int(parts[0])
        except Exception:
            continue
        if old_id not in old_to_new:
            # 理论上不会发生，因为已过滤
            continue
        parts[0] = str(old_to_new[old_id])
        remapped_lines.append(' '.join(parts))

    dst = os.path.join(labels_dir_remap, fname)
    with open(dst, 'w') as f:
        f.write('\n'.join(remapped_lines))
        if remapped_lines:
            f.write('\n')
    if remapped_lines != lines:
        files_changed_remap += 1

print(f'✅ 重映射完成：重写 {files_changed_remap} 个文件')
print(f'重映射后的标签目录：{labels_dir_remap}')

# ======= 可选：根据 66 类名字表，导出新的 names 列表（与 new_id 顺序一致）=======
if names_66_txt and os.path.isfile(names_66_txt):
    with open(names_66_txt, 'r', encoding='utf-8') as f:
        names_all = [ln.strip() for ln in f if ln.strip()]
    # names_all 假定顺序与 Class ID 1..66 一致；YOLO旧ID=ClassID-1
    new_names = [names_all[old_id] for old_id in kept_old_ids]
    with open(new_names_yaml, 'w', encoding='utf-8') as f:
        f.write('names:\n')
        for n in new_names:
            f.write(f'  - {n}\n')
    print(f'✅ 新的 names 列表已写入：{new_names_yaml}')
