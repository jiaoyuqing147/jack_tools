from pathlib import Path
import pandas as pd

CSV_FILE = "train_box_statistics.csv"

df = pd.read_csv(CSV_FILE)

weather_names = [
    "normal",
    "rain",
    "fog",
    "lowlight"
]

groups = {
    k: []
    for k in weather_names
}

for idx, row in df.iterrows():

    weather = weather_names[idx % 4]

    groups[weather].append(
        row["image_name"]
    )

# 保存
for weather in weather_names:

    save_file = f"{weather}.txt"

    with open(save_file, "w", encoding="utf-8") as f:

        for name in groups[weather]:
            f.write(f"{name}\n")

# 输出统计
print("=" * 60)

for weather in weather_names:

    subset = df.iloc[
        [i for i in range(len(df))
         if weather_names[i % 4] == weather]
    ]

    print(f"\n{weather.upper()}")

    print(
        f"Images : {len(subset)}"
    )

    print(
        f"Boxes  : {subset['num_boxes'].sum()}"
    )

    print(
        f"AvgBox : "
        f"{subset['num_boxes'].mean():.4f}"
    )

print("\nDone.")