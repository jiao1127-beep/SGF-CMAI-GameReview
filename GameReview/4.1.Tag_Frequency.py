import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. 配置区域 ---
INPUT_FILE = "./data/game_tags_analysis.xlsx"
# 确保标签名称与你脚本中输出的一致
TARGET_TAGS = ["焦虑表达", "放松舒缓", "对学习有帮助", "经典数学游戏", "游戏玩法难度大", "知识难度大"]

# 设置中文字体（解决绘图中文乱码问题）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows常用黑体
plt.rcParams['axes.unicode_minus'] = False


def plot_cooccurrence_heatmap():
    print(f"正在读取分析数据: {INPUT_FILE}")
    df = pd.read_excel(INPUT_FILE)

    # 过滤掉无效数据
    valid_df = df[~df['标签'].isin(['跳过', '处理失败', None])].copy()

    # --- 2. 构建多标签矩阵 (One-hot Encoding) ---
    # 创建一个全为0的矩阵，行数为评论数，列数为标签数
    tag_matrix = pd.DataFrame(0, index=valid_df.index, columns=TARGET_TAGS)

    for idx, row in valid_df.iterrows():
        tags_in_row = str(row['标签']).split('，')  # 注意这里用的是你脚本里的中文逗号
        for tag in tags_in_row:
            tag = tag.strip()
            if tag in TARGET_TAGS:
                tag_matrix.loc[idx, tag] = 1

    # --- 3. 计算共现矩阵 ---
    # 共现矩阵 = 矩阵转置 * 矩阵
    cooccurrence_matrix = tag_matrix.T.dot(tag_matrix)

    # --- 4. 绘图 ---
    plt.figure(figsize=(10, 8))

    # 使用蓝红色调或绿色调，颜色深浅代表频次
    sns.heatmap(
        cooccurrence_matrix,
        annot=True,  # 在格子里显示数值
        fmt="d",  # 格式为整数
        cmap="YlGnBu",  # 颜色方案：黄-绿-蓝
        square=True,  # 设置为正方形
        cbar_kws={"label": "共现次数"}
    )

    plt.title("游戏评论标签共现相关性图", fontsize=15)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # 保存结果
    plt.savefig("./data/tag_cooccurrence_heatmap.png", dpi=300)
    print("相关性热力图已生成并保存至 ./data/tag_cooccurrence_heatmap.png")
    plt.show()


if __name__ == "__main__":
    plot_cooccurrence_heatmap()