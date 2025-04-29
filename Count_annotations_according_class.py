# import os
# import xml.etree.ElementTree as ET
#
# # Path to the folder containing XML files
# xml_folder_path = r'W:\Jack_datasets\S2TLD\S2TLD\S2TLD720x1280\normal_2\Annotations'
#
# # Initialize counts for each label
# label_counts = {
#     "red": 0,
#     "yellow": 0,
#     "green": 0,
#     "off": 0
# }
#
# # Iterate through all files in the folder
# for filename in os.listdir(xml_folder_path):
#     if filename.endswith('.xml'):
#         file_path = os.path.join(xml_folder_path, filename)
#         try:
#             # Parse the XML file
#             tree = ET.parse(file_path)
#             root = tree.getroot()
#
#             # Check for objects and count specific labels
#             for obj in root.findall('object'):
#                 name = obj.find('name').text
#                 if name in label_counts:
#                     label_counts[name] += 1
#         except ET.ParseError as e:
#             print(f"Error parsing {filename}: {e}")
#
# # Print the counts
# for label, count in label_counts.items():
#     print(f'Total number of "{label}" labels: {count}')




#找到yellow或off出现在哪些图像中
import os
import xml.etree.ElementTree as ET

# XML 文件夹路径
xml_folder = r"W:\Jack_datasets\S2TLD\S2TLD\S2TLD720x1280\normal_2\Annotations"

# 统计出现次数的字典
count_dict = {"yellow": 0, "off": 0}
file_dict = {"yellow": [], "off": []}

# 遍历 XML 文件
for filename in os.listdir(xml_folder):
    if filename.endswith(".xml"):
        file_path = os.path.join(xml_folder, filename)
        tree = ET.parse(file_path)
        root = tree.getroot()

        # 统计 yellow 和 off
        yellow_count = sum(1 for obj in root.findall("object") if obj.find("name").text == "yellow")
        off_count = sum(1 for obj in root.findall("object") if obj.find("name").text == "off")

        if yellow_count > 0:
            count_dict["yellow"] += yellow_count
            file_dict["yellow"].append((filename, yellow_count))

        if off_count > 0:
            count_dict["off"] += off_count
            file_dict["off"].append((filename, off_count))

# 输出统计结果
print("🔍 统计结果：")
print(f"Yellow 总共出现次数: {count_dict['yellow']}")
print(f"Off 总共出现次数: {count_dict['off']}")
print("\n📂 Yellow 出现的文件：")
for file, count in file_dict["yellow"]:
    print(f"{file}: {count} 次")

print("\n📂 Off 出现的文件：")
for file, count in file_dict["off"]:
    print(f"{file}: {count} 次")
