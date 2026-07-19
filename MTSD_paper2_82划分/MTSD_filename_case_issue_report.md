# MTSD 文件名大小写问题排查报告

## 结论

根因已由当前代码和 Git 中未修改版本的实际差异确认，不是推测。

最早丢失原始大小写的位置是旧版 `1_get_image_sizes.py`：

```python
writer.writerow([image_path.name.lower(), width, height])
```

它把磁盘真实名称 `DSC-0101.JPG` 写成了 `dsc-0101.jpg`。旧版
`2_merge_detection_recognition_to_yolo66.py` 随后又通过
`normalize_name(...).lower()` 将 detection、recognition 和 image_sizes 三个来源统一小写，最后使用
`Path(scene).stem` 生成标签。因此最早的损失在第 1 步，第 2 步又独立重复了损失并把小写 lookup key 错当成输出名。

当前工作区中的第 1、2 步已经被手工改动：第 1 步现在写 `image_path.name`；第 2 步现在从
`image_sizes.csv` 保存的真实图片名生成标签，并在运行前删除旧 YOLO66 标签。这两处改动尚未放在
`*_fixed.py` 中，且是 Git 未提交修改。本报告保留旧文件不动，另行提供可审查的 fixed 版本。

## 名称传播链

1. 原始图片目录 `E:\DataSets\MTSD\Detection` 保存真实磁盘名称，例如 `DSC-0101.JPG`。
2. 旧第 1 步对 `image_path.name.lower()`，从这里开始 CSV 名称成为小写。
3. 旧第 2 步把三个输入名称全部小写，以小写 `scene` 分组，并用该 `scene` 的 stem 写出 YOLO66 标签，小写被固定到标签文件名。
4. 第 4 步用 `src.name` 写 filtered 和 final 标签。它没有主动转小写，但原样继承第 2 步的小写名称。
5. 第 6 步用 `label_paths[idx].name` 复制 train/val 标签。它也没有主动转小写，只继承第 4 步名称。
6. 第 7 步用 `label_path.stem.lower()` 查找图片，却以 `src_image.name` 复制图片。因此标签沿旧链保持小写，图片重新取得磁盘真实大小写，最终形成严格不匹配、忽略大小写匹配。

`3_count_66_classes.py` 的 `.lower()` 仅作用于 recognition DataFrame 的文件名列，而该脚本只统计 Class ID，
不生成标签文件名；它对本问题无影响，无需替换。`5_generate_classes54.py` 只根据映射生成类别名称，
不处理图片/标签名称，也无需替换。

## 为什么重新运行后仍可能存在

- 旧第 4 步不清理 `labels_pruned_filtered` 和 `labels` 根目录，旧标签可继续残留。若新一轮少生成某些文件，旧文件不会消失。
- 旧第 2 步原本也不清理 YOLO66；当前工作区的手工改动才加入清理。
- 仅重跑第 6、7 步只会继续复制已经小写的 final 标签，无法恢复名称。
- 若运行的是另一个目录中的同名脚本，或 Python 当前工作目录/导入路径加载了另一份
  `mtsd_paths_config.py`，实际读写目录会与预期不同。
- Windows 默认大小写不敏感。仅改变字符大小写的覆盖/重命名不能可靠证明目录已干净；必须先按脚本责任范围清空，再生成并核对精确文件集合。
- `LABELS_FINAL_DIR` 是 `...\labels`，而 train/val 标签是其子目录。这不是错误或重复路径，但意味着第 4 步只能清理根层 `.txt`，不能删除 `train`/`val`；第 6 步分别负责重建这两个子目录。

本次实测配置全部指向 `E:\DataSets\MTSD\yolo54_paper2`，未发现配置指向旧 `yolo54` 目录或重复目录。
当前磁盘时间戳显示：image_sizes 在 00:43 重建，YOLO66 在 00:50 重建，下游在 00:52 重建。
当前实测结果已经是 train 785/785 严格匹配、val 198/198 严格匹配，且两个示例均为
`DSC-0101.JPG`/`DSC-0101.txt`、`P1830924.JPG`/`P1830924.txt`。这说明用户此前的 123/30
结果来自手工修补和重跑之前的产物。

## 重名和隐藏问题检查

实际扫描结果：原始 1000 张图片、YOLO66 991 个标签、filtered 991 个标签、final 根层 991 个标签，
各目录均未发现忽略大小写后 stem 重名。train/val 各自也未发现此类重名，且 train 与 val 当前没有交集。

仍需防范的隐藏问题包括：CSV 名称与真实图片名大小写不同；同一 stem 使用不同图片扩展名；输出目录残留非预期文件；
标签格式不是 5 列；类别越界；非数值或超出 `[0,1]` 的坐标；空标签被错误纳入划分；不同脚本加载不同配置模块。
fixed 脚本均对此进行前置或输出验证。

## 替换关系和修改内容

| 新脚本 | 对应旧脚本 | 主要修改 |
|---|---|---|
| `1_get_image_sizes_fixed.py` | `1_get_image_sizes.py` | CSV 使用真实 `image_path.name`；检查图片 stem 忽略大小写重名；原子替换 CSV；回读验证名称和数量。 |
| `2_merge_detection_recognition_to_yolo66_fixed.py` | `2_merge_detection_recognition_to_yolo66.py` | lookup key 与真实名称分离；CSV 必须严格保留磁盘名称；从真实图片 stem 输出；仅清理 YOLO66；验证重名、来源、数量、66 类范围、YOLO 格式和坐标。 |
| `4_prune_le3_and_remap_labels_fixed.py` | `4_prune_le3_and_remap_labels.py` | 从真实图片索引恢复每个输出 stem；分别清理 filtered 和 final 根层标签；保留 `<=3` 删除和原映射逻辑；验证来源、输出集合、格式和类别范围。 |
| `6_split_train_val_8_2_fixed.py` | `6_split_train_val_8_2.py` | 完整保留 54 类、0.20、300 trials、seed 42 和原评分/提前停止逻辑；复制时使用真实图片 stem；重建 train/val 标签目录；验证缺图、格式、数量和划分互斥完备。 |
| `7_copy_images_by_label_split_fixed.py` | `7_copy_images_by_label_split.py` | 查找仍可忽略大小写，但要求复制前 label stem 与真实 image stem 严格相同；检查重名和缺图；重建自己负责的图片目录；验证复制集合和 train/val 无重复。 |
| `8_validate_final_dataset_fixed.py` | 新增最终检查 | 严格检查图片/标签 stem、忽略大小写重名、54 类范围、YOLO 五列格式、坐标范围、空标签和 train/val 重复。 |

`mtsd_case_utils.py` 是 fixed 脚本共用的安全检查模块。旧脚本和配置均未被修改。

## 推荐运行顺序

在本脚本目录中依次执行：

```text
python 1_get_image_sizes_fixed.py
python 2_merge_detection_recognition_to_yolo66_fixed.py
python 3_count_66_classes.py
python 4_prune_le3_and_remap_labels_fixed.py
python 5_generate_classes54.py
python 6_split_train_val_8_2_fixed.py
python 7_copy_images_by_label_split_fixed.py
python 8_validate_final_dataset_fixed.py
```

无需手工删除原始图片或原始标注，也不建议手工删除输出。每个 fixed 生成脚本会在所有输入预检通过后，
只清理自己明确负责的输出：第 2 步清理 YOLO66 `.txt`；第 4 步清理 filtered `.txt` 和 final 根层 `.txt`；
第 6 步重建 labels/train、labels/val；第 7 步重建 images/train、images/val。这样也覆盖旧文件残留检查。

## 每步检查点

1. 第 1 步：CSV 行数应等于原始图片数；示例名称应保持 `DSC-0101.JPG`、`P1830924.JPG`。
2. 第 2 步：YOLO66 标签名应使用真实图片 stem；缺 recognition、缺尺寸和框数不一致报告应符合历史统计。
3. 第 3 步：类别计数和原统计一致；其文件名小写不影响结果。
4. 第 4 步：仍删除出现次数 `<=3` 的类别；最终映射仍为 54 类；文件和框统计应与旧逻辑一致。
5. 第 5 步：`classes.txt`、`names.yaml` 均应有 54 类。
6. 第 6 步：保持 8:2、多标签分层、seed 搜索和原统计；当前数据应为 train 785、val 198（8 个空标签仍按原逻辑丢弃）。
7. 第 7 步：复制数必须等于对应标签数，缺图为 0。
8. 最终检查：必须严格通过，不能只看忽略大小写匹配。

期望输出为：

```text
TRAIN
图片数：785
标签数：785
严格同名匹配：785
有图片但无同名标签：0
有标签但无同名图片：0

VAL
图片数：198
标签数：198
严格同名匹配：198
有图片但无同名标签：0
有标签但无同名图片：0
```

如果仍失败，首先打印实际导入的 `mtsd_paths_config.__file__` 和所有解析后的路径；确认运行的是本目录 fixed 文件；
检查 `image_sizes.csv` 的修改时间和示例名称；逐级比较 YOLO66、filtered、final、split label 的真实目录；
检查是否有同步软件、另一个进程或旧脚本在 fixed 步骤后重新写回目录；最后检查报错中给出的大小写折叠重名、缺源文件或格式异常。
