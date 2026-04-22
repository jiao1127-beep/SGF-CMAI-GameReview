import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 1. 环境配置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
sns.set_theme(style="whitegrid", font='SimHei')


def draw_topic_boxplot():
    # --- 配置文件路径 ---
    input_path = r"D:\LDA-gamereviews\data\game_tags_analysis.xlsx"
    output_path = r"D:\LDA-gamereviews\data\topic_playtime_boxplot.pdf"

    if not os.path.exists(input_path):
        print("错误：未找到数据文件")
        return

    # 读取数据
    df = pd.read_excel(input_path)

    # --- 2. 数据清洗与预处理 ---
    # 过滤掉时长为 0 的数据
    df_clean = df[df['游戏时长'] > 0].copy()

    # 主题名称映射
    topic_names = {
        0: '主题1：关卡设计',
        1: '主题2：视觉交互',
        2: '主题3：休闲属性',
        3: '主题4：智力挑战'
    }
    df_clean['主题名称'] = df_clean['游戏评论_主题编号'].map(topic_names)

    # 计算对数时长以缓解长尾分布，增加箱线图的可读性 [cite: 6, 25]
    df_clean['log_playtime'] = np.log1p(df_clean['游戏时长'].astype(float))

    # --- 3. 绘制箱线图 ---
    # 创建子图：左侧为原始时长，右侧为对数时长
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 左图：原始时长分布
    sns.boxplot(x='主题名称', y='游戏时长', data=df_clean, ax=ax1,
                palette="Set3", showfliers=False)  # showfliers=False 隐藏极端异常值以看清箱体
    ax1.set_title('不同主题游戏时长分布 (剔除异常值显示)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('时长 (分钟)')
    ax1.set_xlabel('主题分类')

    # 右图：对数时长分布 (学术论文推荐)
    sns.boxplot(x='主题名称', y='log_playtime', data=df_clean, ax=ax2,
                palette="Set3")
    ax2.set_title('不同主题游戏时长分布 (对数处理)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('对数时长 $\ln(1+Playtime)$')
    ax2.set_xlabel('主题分类')

    # 优化布局
    plt.xticks(rotation=15)
    plt.tight_layout()

    # 保存图片
    plt.savefig(output_path, dpi=300)
    print(f"箱线图已保存至: {output_path}")
    plt.show()


if __name__ == "__main__":
    draw_topic_boxplot()