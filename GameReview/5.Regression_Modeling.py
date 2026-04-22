import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import OLSInfluence
import scipy.stats as stats
import os

# 环境配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='SimHei')


def run_labeled_triple_regression():
    input_path = r"D:\LDA-gamereviews\data\game_tags_analysis.xlsx"
    if not os.path.exists(input_path):
        print("错误：未找到数据文件")
        return

    df = pd.read_excel(input_path)

    # --- 1. 数据清洗与变量转换 ---
    df['游戏时长'] = df['游戏时长'].replace(0, np.nan)
    cols_to_check = ['游戏时长', 'score', '情感标签', '标签', '游戏评论_主题编号', 'contents']
    df = df.dropna(subset=cols_to_check).reset_index(drop=True)

    # A. 因变量：对数转换
    df['log_playtime'] = np.log1p(df['游戏时长'].astype(float))

    # B. 机制标签 (多标签 0/1)
    m_tags = ["焦虑表达", "放松舒缓", "对学习有帮助", "经典数学游戏", "游戏玩法难度大", "知识难度大"]
    for tag in m_tags:
        # 重命名标签列名，避免公式解析错误（空格/特殊字符）
        clean_name = tag.replace(' ', '_')
        df[clean_name] = df['标签'].apply(lambda x: 1 if tag in str(x) else 0)

    clean_m_tags = [t.replace(' ', '_') for t in m_tags]

    # C. 情感标签：确保为字符串（保存原始名称）
    df['Emotion'] = df['情感标签'].astype(str)

    # D. 主题编号：转换为 Topic0, Topic1 等字符串标签
    df['Topic'] = df['游戏评论_主题编号'].apply(lambda x: f"Topic{int(x)}")

    # E. 控制变量
    df['contents_length'] = df['contents'].str.len()

    # --- 2. 定义模型公式 (使用 SMF 自动处理分类变量) ---
    # C(Variable, Treatment(reference='xxx')) 明确指定基准组
    formula_base = " + ".join(clean_m_tags) + " + C(Emotion, Treatment('中性'))"
    formula_controls = " + score + contents_length"
    formula_topics = " + C(Topic, Treatment('Topic2'))"  # 假设以主题2为基准

    models_config = {
        "Model_1_Full": f"log_playtime ~ {formula_base} + {formula_topics} + {formula_controls}",
        "Model_2_Base": f"log_playtime ~ {formula_base}",
        "Model_3_WithControls": f"log_playtime ~ {formula_base} + {formula_controls}"
    }

    # --- 3. 循环运行分析与绘图 ---
    for name, formula in models_config.items():
        print(f"正在分析 {name}...")

        # 估计模型 (使用 HC3 稳健标准误)
        model = smf.ols(formula, data=df)
        results = model.fit(cov_type='HC3')

        # A. 导出 LaTeX
        tex_path = rf"D:\LDA-gamereviews\data\{name}_labeled_results.tex"
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(results.summary().as_latex())

        # B. 绘制四象限诊断图
        influence = OLSInfluence(results)
        std_resid = np.array(influence.resid_studentized_internal)
        fitted = np.array(results.fittedvalues)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{name} 诊断证图 (保留分类标签)', fontsize=20, fontweight='bold', y=0.95)

        # 1. 残差 vs 拟合值
        sns.residplot(x=fitted, y=results.resid, lowess=True, ax=axes[0, 0], line_kws={'color': 'red'})
        axes[0, 0].set_title('Residuals vs Fitted (线性度)')

        # 2. 正态 Q-Q 图
        stats.probplot(std_resid, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Normal Q-Q (正态性)')

        # 3. 比例-位置图
        sns.scatterplot(x=fitted, y=np.sqrt(np.abs(std_resid)), ax=axes[1, 0], alpha=0.4)
        sns.regplot(x=fitted, y=np.sqrt(np.abs(std_resid)), scatter=False, lowess=True, ax=axes[1, 0],
                    line_kws={'color': 'red'})
        axes[1, 0].set_title('Scale-Location (方差齐性)')

        # 4. 库克距离
        axes[1, 1].stem(np.arange(len(std_resid)), influence.cooks_distance[0], markerfmt=",")
        axes[1, 1].set_title("Cook's Distance (影响点分析)")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = rf"D:\LDA-gamereviews\data\{name}_quadrant.pdf"
        plt.savefig(plot_path, dpi=300)
        plt.close()

        print(f"  - 完成：{tex_path}")


if __name__ == "__main__":
    run_labeled_triple_regression()