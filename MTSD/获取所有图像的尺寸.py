from PIL import Image
import os
import csv

image_folder = r'F:\jack_dataset\MTSD\MTSD\Detection'
output_csv = 'image_sizes.csv'

with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'width', 'height'])

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(image_folder, filename)
            with Image.open(img_path) as img:
                width, height = img.size
                # 将文件名统一小写写入
                writer.writerow([filename.lower(), width, height])

print(f'Saved image sizes to {output_csv}')
