# GTSDB 8:2 科学划分脚本

本流程具有以下特点：

- 以图片为单位划分，避免一张图片跨越 train/val；
- 根据一张图片包含的全部类别进行多标签分层；
- 只出现于一张图片的类别强制保留在 train；
- 保留无交通标志的背景图片，并按比例参与划分；
- 全部图片统一转换为无损 PNG；
- 自动生成空背景标签、逐类别统计和最终验收报告；
- 固定随机种子，保证划分可以复现。

## 1. 安装依赖

在当前Python环境执行：

```bash
pip install -r requirements.txt
```

## 2. 修改路径

打开 `config.py`，至少修改：

```python
SOURCE_IMAGE_DIR = Path(r"你的GTSDB原始图片目录")
GT_TXT_PATH = SOURCE_IMAGE_DIR / "gt.txt"
WORK_DIR = Path(r"中间结果目录")
OUTPUT_DATASET_DIR = Path(r"最终YOLO数据集目录")
```

如果 `gt.txt` 不在图片目录中，请直接填写其完整路径。

## 3. 按顺序运行

```bash
python 01_prepare_gtsdb.py
python 02_split_gtsdb_multilabel.py
python 03_build_and_verify_yolo_dataset.py
```

每一步成功后再运行下一步。第二步输出的关键文件包括：

```text
prepared_8_2/
├── splits/
│   ├── train.txt
│   └── val.txt
└── reports/
    ├── split_summary.txt
    └── split_class_statistics.csv
```

先查看 `split_summary.txt` 和 `split_class_statistics.csv`，确认数量及类别覆盖符合预期，再执行第三步。

## 4. 最终目录

```text
GTSDB_YOLO_8_2/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── splits/
├── reports/
└── gtsdb.yaml
```

训练时使用生成的 `gtsdb.yaml`。

## 5. 注意事项

1. `names` 默认写为字符串 `0` 到 `42`，不影响按类别编号训练。如需显示真实交通标志名称，可在 `03_build_and_verify_yolo_dataset.py` 生成YAML的位置替换名称。
2. 验证集不一定覆盖全部43类。极低频类别优先留在训练集，这是正常且合理的。
3. 不要手工向最终目录增加旧文件，否则可能污染固定划分。
4. 如果输出目录已经有内容，第三步默认停止运行。建议更换一个新的输出目录；只有确认目录内容可重新生成时，才把 `CLEAN_OUTPUT=True`。
5. 论文实验中应保存并公开 `train.txt`、`val.txt`、随机种子及逐类别统计表。
