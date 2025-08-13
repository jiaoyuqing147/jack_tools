import os

def collect_class_ids(label_dir):
    class_ids = set()
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(label_dir, filename), "r") as f:
                for line in f:
                    if line.strip():
                        cls_id = line.strip().split()[0]
                        class_ids.add(cls_id)
    return class_ids

# 设置路径
train_dir = r"D:\Jiao\dataset\TsingHua100K\tt100k_2021\labels_all\train"
test_dir = r"D:\Jiao\dataset\TsingHua100K\tt100k_2021\labels_all\test"
other_dir = r"D:\Jiao\dataset\TsingHua100K\tt100k_2021\labels_all\other"  # ✅ 新增 other 文件夹

# 获取类别 ID 集合
train_classes = collect_class_ids(train_dir)
test_classes = collect_class_ids(test_dir)
other_classes = collect_class_ids(other_dir)

# 打印基本统计
print(f"📊 训练集中类别数量: {len(train_classes)}")
print(f"类别 ID 列表: {sorted(list(train_classes))}\n")

print(f"📊 测试集中类别数量: {len(test_classes)}")
print(f"类别 ID 列表: {sorted(list(test_classes))}\n")

print(f"📊 other 文件夹类别数量: {len(other_classes)}")
print(f"类别 ID 列表: {sorted(list(other_classes))}\n")

# 差异分析（train vs test）
missing_in_test = train_classes - test_classes
extra_in_test = test_classes - train_classes
common_train_test = train_classes & test_classes

# 差异分析（train vs other）
missing_in_other = train_classes - other_classes
extra_in_other = other_classes - train_classes
common_train_other = train_classes & other_classes

# ✅ 新增：差异分析（test vs other）
missing_in_other_from_test = test_classes - other_classes
extra_in_other_from_test = other_classes - test_classes
common_test_other = test_classes & other_classes

# ✅ 新增：三方交集
common_all_three = train_classes & test_classes & other_classes

# 输出交集和差异
print(f"🔁 训练和测试共有的类别数量: {len(common_train_test)}")
print(f"共有类别 ID: {sorted(list(common_train_test))}\n")

print(f"🔁 训练和 other 共有的类别数量: {len(common_train_other)}")
print(f"共有类别 ID: {sorted(list(common_train_other))}\n")

print(f"🔁 测试和 other 共有的类别数量: {len(common_test_other)}")
print(f"共有类别 ID: {sorted(list(common_test_other))}\n")

print(f"🔁🔁🔁 train / test / other 三者共有的类别数量: {len(common_all_three)}")
print(f"共有类别 ID: {sorted(list(common_all_three))}\n")

# 输出缺失类别
print(f"✅ 测试集中缺失的类别数（train 有但 test 没有）: {len(missing_in_test)}")
print(f"缺失类别 ID: {sorted(list(missing_in_test))}\n")

print(f"✅ other 中缺失的类别数（train 有但 other 没有）: {len(missing_in_other)}")
print(f"缺失类别 ID: {sorted(list(missing_in_other))}\n")

print(f"✅ other 相比 test 缺失的类别数（test 有但 other 没有）: {len(missing_in_other_from_test)}")
print(f"缺失类别 ID: {sorted(list(missing_in_other_from_test))}\n")

# 输出多余类别
print(f"⚠️ 测试集中多余的类别数（test 有但 train 没有）: {len(extra_in_test)}")
print(f"多余类别 ID: {sorted(list(extra_in_test))}\n")

print(f"⚠️ other 中多余的类别数（other 有但 train 没有）: {len(extra_in_other)}")
print(f"多余类别 ID: {sorted(list(extra_in_other))}\n")

print(f"⚠️ other 相比 test 多余的类别数（other 有但 test 没有）: {len(extra_in_other_from_test)}")
print(f"多余类别 ID: {sorted(list(extra_in_other_from_test))}")
