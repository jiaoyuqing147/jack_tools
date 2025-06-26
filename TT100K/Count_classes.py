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
train_dir = r"F:\jack_dataset\tt100k_2021\yolo\labels\train"
test_dir = r"F:\jack_dataset\tt100k_2021\yolo\labels\test"

# 获取类别ID集合
train_classes = collect_class_ids(train_dir)
test_classes = collect_class_ids(test_dir)

# 打印基本统计
print(f"训练集中类别数量: {len(train_classes)}")
print(f"训练集类别 ID 列表: {sorted(list(train_classes))}\n")

print(f"测试集中类别数量: {len(test_classes)}")
print(f"测试集类别 ID 列表: {sorted(list(test_classes))}\n")

# 差异分析
missing_in_test = train_classes - test_classes      # 只在训练集中出现的类别（测试集中缺失）
extra_in_test = test_classes - train_classes        # 只在测试集中出现的类别（训练集中未学过）
common_classes = train_classes & test_classes       # 同时出现在训练和测试集中的类别（交集）

# 输出交集
print(f"🔁 训练和测试共有的类别数量: {len(common_classes)}")
print(f"共有类别 ID: {sorted(list(common_classes))}\n")

# 输出测试集中缺失的类（训练有但测试没有）
print(f"✅ 测试集中缺失的类别数（训练集中有但测试集中没有）: {len(missing_in_test)}")
print(f"缺失的类别 ID: {sorted(list(missing_in_test))}\n")

# 输出测试集多余的类（训练没有但测试出现）
print(f"⚠️ 测试集中多余的类别数（测试集中有但训练集中没有）: {len(extra_in_test)}")
print(f"多余的类别 ID: {sorted(list(extra_in_test))}")


'''
🔁 训练和测试共有的类别数量: 143
共有类别 ID: ['0', '10', '100', '101', '102', '11', '110', '111', '112', '114', '115', '116', '118', '119', '12', '121', '122', '123', '124', '126', '128', '13', '130', '132', '133', '134', '135', '136', '137', '138', '139', '14', '140', '143', '145', '148', '15', '150', '153', '154', '156', '158', '159', '16', '160', '162', '165', '168', '169', '17', '170', '172', '174', '175', '176', '179', '18', '180', '181', '182', '183', '185', '186', '187', '188', '19', '191', '193', '194', '196', '197', '198', '2', '20', '201', '203', '205', '207', '208', '209', '21', '210', '211', '214', '216', '218', '22', '220', '221', '222', '223', '224', '227', '228', '229', '23', '231', '24', '25', '27', '3', '30', '31', '32', '35', '36', '39', '42', '43', '44', '45', '49', '50', '51', '52', '55', '57', '58', '60', '61', '62', '64', '65', '66', '68', '69', '7', '71', '72', '73', '76', '77', '8', '80', '85', '87', '88', '9', '91', '95', '96', '97', '98']
'''