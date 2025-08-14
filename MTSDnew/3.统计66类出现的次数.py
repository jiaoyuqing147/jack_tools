import pandas as pd

# 读取 recognition 文件
recognition_file = 'GT_Recognition.txt'
recognition_df = pd.read_csv(recognition_file, sep=';', engine='python')

# 去除文件名引号、标准化
recognition_df['File Name'] = recognition_df['File Name'].str.strip("'").str.lower()

# 统计 Class ID 出现次数
class_counts = recognition_df['Class ID'].value_counts().sort_index()

# 打印每个类别的出现次数
for class_id, count in class_counts.items():
    print(f'Class ID {class_id}: {count} 次')

# 也可以保存到 CSV
class_counts.to_csv('class_id_counts.csv', header=['count'])

print('\n✅ 统计完成，已保存至 class_id_counts.csv')
