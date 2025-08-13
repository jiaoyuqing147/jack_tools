import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ✅ 参数配置（你只需要修改这里）
# ----------------------------------------
# 子集列表（支持任意个目录，路径必须成对给出）
# ✅ 只需改这一行路径即可
base_dir = r'E:\tt100k_2021\yolo110'

# ✅ 子集设置（无需再每次重复写路径）
sets = [
    {
        'name': 'train',
        'image_dir': os.path.join(base_dir, 'images', 'train'),
        'label_dir': os.path.join(base_dir, 'labels', 'train')
    },
    {
        'name': 'other',
        'image_dir': os.path.join(base_dir, 'images', 'other'),
        'label_dir': os.path.join(base_dir, 'labels', 'other')
    },
    {
        'name': 'test',
        'image_dir': os.path.join(base_dir, 'images', 'test'),
        'label_dir': os.path.join(base_dir, 'labels', 'test')
    }
]


max_cls_id = 110           # 类别 ID 上限（从0开始）
delete_bad_files = False    # 是否执行删除，False 表示只检测不删
# ----------------------------------------

img_suffixes = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# ✅ 主逻辑循环
for subset in sets:
    name = subset['name']
    image_dir = Path(subset['image_dir']).resolve()
    label_dir = Path(subset['label_dir']).resolve()

    print(f"\n📂 正在检查子集：{name}")
    print(f" - 图像路径：{image_dir}")
    print(f" - 标签路径：{label_dir}")

    bad_images = []
    bad_labels = []

    # 检查图像是否损坏
    image_files = [f for f in image_dir.rglob('*') if f.suffix.lower() in img_suffixes]
    print(f'\n🔍 正在检查 {len(image_files)} 张图像...')
    for img_path in tqdm(image_files):
        try:
            with Image.open(img_path) as im:
                im.verify()
        except Exception as e:
            print(f'❌ 损坏图像: {img_path}，原因: {e}')
            bad_images.append(img_path)

    # 检查标签是否合法
    label_files = list(label_dir.rglob('*.txt'))
    print(f'\n🔍 正在检查 {len(label_files)} 个标签...')
    for label_path in tqdm(label_files):
        try:
            if os.path.getsize(label_path) == 0:
                print(f'⚠️ 空标签: {label_path}')
                bad_labels.append(label_path)
                continue

            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 1:
                    print(f'❌ 标签格式错误: {label_path}')
                    bad_labels.append(label_path)
                    break
                cls_id = int(float(parts[0]))
                if cls_id < 0 or cls_id >= max_cls_id:
                    print(f'❌ 标签越界: {label_path}，类别 ID={cls_id}')
                    bad_labels.append(label_path)
                    break
        except Exception as e:
            print(f'❌ 标签文件异常: {label_path}, {e}')
            bad_labels.append(label_path)

    # 删除异常文件（可选）
    if delete_bad_files:
        for bad in bad_images:
            try:
                os.remove(bad)
                print(f'🗑️ 已删除图像: {bad}')
                label_path = label_dir / (bad.stem + '.txt')
                if label_path.exists():
                    os.remove(label_path)
                    print(f'🗑️ 已删除对应标签: {label_path}')
            except Exception as e:
                print(f'⚠️ 删除失败: {bad}, {e}')

        for bad in bad_labels:
            try:
                os.remove(bad)
                print(f'🗑️ 已删除标签: {bad}')
                img_candidates = [image_dir / (bad.stem + ext) for ext in img_suffixes]
                for img_path in img_candidates:
                    if img_path.exists():
                        os.remove(img_path)
                        print(f'🗑️ 已删除对应图像: {img_path}')
                        break
            except Exception as e:
                print(f'⚠️ 删除失败: {bad}, {e}')

    # ✅ 报告结果
    print(f'\n📊 {name} 子集检查完成：')
    print(f'🚫 损坏图像数量：{len(bad_images)}')
    print(f'🚫 异常标签数量：{len(bad_labels)}')

    if not delete_bad_files:
        if bad_images:
            print('\n🧾 损坏图像列表：')
            for path in bad_images:
                print(f' - {path}')
        if bad_labels:
            print('\n🧾 异常标签文件列表：')
            for path in bad_labels:
                print(f' - {path}')
        print('📌 如需删除，请将 delete_bad_files = True 后再次运行。')
