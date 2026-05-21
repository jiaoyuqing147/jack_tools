import matplotlib.pyplot as plt

# =========================
# ✅ 你的统计结果（改这里）
# =========================
labels = ['Tiny', 'Small', 'Medium', 'Large']
sizes = [5757, 1936, 516, 8]  # ← 换成你的统计结果

# =========================
# ✅ 高亮颜色（适合论文）
# =========================
colors = [
    '#FF4D4F',  # 红（tiny）
    '#FFA500',  # 橙（small）
    '#1E90FF',  # 蓝（medium）
    '#32CD32',  # 绿（large）
]

plt.figure(figsize=(6, 6))

wedges, texts, autotexts = plt.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',
    startangle=140,
    textprops={'fontsize': 12}
)

plt.title('Object Size Distribution (TT100K)', fontsize=14)

# 让字体更清晰
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(11)

plt.tight_layout()

# 保存（论文用）
plt.savefig("size_distribution.png", dpi=300)

plt.show()