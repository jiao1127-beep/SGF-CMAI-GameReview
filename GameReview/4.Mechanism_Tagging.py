import pandas as pd
import json
import os
import time
from openai import OpenAI
from tqdm import tqdm

# --- 1. 配置区域 ---
client = OpenAI(
    api_key="sk-R2QX0RbbhY6dNo5w6587785cA0E14aFeAb91D0A9CeEdEc26",
    base_url="https://api.apiyi.com/v1"
)

INPUT_FILE = "./data/LDA_comments_t.xlsx"
OUTPUT_FILE = "./data/game_tags_analysis.xlsx"
STAT_OUTPUT_FILE = "./data/tag_statistics_report.xlsx"  # 统计报表文件名
SAVE_INTERVAL = 50
MODEL_NAME = "gemini-2.5-flash"

# 预设标签列表，用于最后统计
TARGET_TAGS = ["焦虑表达", "放松舒缓", "对学习有帮助", "经典数学游戏", "游戏玩法难度大", "知识难度大"]

# 提示词模板
PROMPT_TEMPLATE = """
请作为一名资深的游戏化学习研究员，结合【游戏描述】与【玩家评论】，对数据进行多标签标注。
标签定义：
1. 焦虑表达：压力、疲倦、紧张或消极状态。
2. 放松舒缓：平静、快乐、舒适或感到放松。
3. 对学习有帮助：掌握知识、考试、升学有帮助。
4. 经典数学游戏：数独、珠算、21点、棋牌等。
5. 游戏玩法难度大：玩法、操作、机制有难度。
6. 知识难度大：数学知识有难度，让人不愉悦。

输出格式（严格 JSON）：
{{
  "标签": ["标签1", "标签2"],
  "判定依据": "简要说明判定逻辑"
}}

---
【游戏描述】：{description}
【玩家评论】：{content}
"""


# --- 2. 核心分析函数 ---
def analyze_multilabel(description, content):
    if not content or pd.isna(content) or str(content).strip() == "":
        return {"标签": "跳过", "判定依据": "评论内容为空"}

    desc_text = description if pd.notna(description) else "暂无描述"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个专业的教育游戏研究助手。"},
                {"role": "user", "content": PROMPT_TEMPLATE.format(description=desc_text, content=content)}
            ],
            response_format={"type": "json_object"}
        )
        res_data = json.loads(response.choices[0].message.content)
        tags = res_data.get("标签", [])
        tags_str = "，".join(tags) if isinstance(tags, list) else str(tags)
        return {"标签": tags_str, "判定依据": res_data.get("判定依据", "")}
    except Exception as e:
        time.sleep(2)
        return {"标签": "处理失败", "判定依据": str(e)}


# --- 3. 统计函数 ---
def save_statistics(df):
    print("\n正在生成统计报表...")
    # 过滤掉非有效行
    valid_df = df[~df['标签'].isin(['跳过', '处理失败', None])]
    total_valid = len(valid_df)

    stats_list = []
    for tag in TARGET_TAGS:
        count = valid_df['标签'].str.contains(tag, na=False).sum()
        percentage = (count / total_valid) * 100 if total_valid > 0 else 0
        stats_list.append({"标签名称": tag, "出现频次": count, "占比 (%)": round(percentage, 2)})

    stats_df = pd.DataFrame(stats_list)
    stats_df.to_excel(STAT_OUTPUT_FILE, index=False)
    print(f"统计报表已保存至: {STAT_OUTPUT_FILE}")


# --- 4. 主执行流程 ---
def main():
    # 读取/续传逻辑
    if os.path.exists(OUTPUT_FILE):
        df = pd.read_excel(OUTPUT_FILE)
    else:
        df = pd.read_excel(INPUT_FILE)

    if '标签' not in df.columns: df['标签'] = None
    if '判定依据' not in df.columns: df['判定依据'] = None

    total_rows = len(df)
    for i in tqdm(range(total_rows), desc="标注进度"):
        if pd.notna(df.loc[i, '标签']) and df.loc[i, '标签'] != "处理失败":
            continue

        result = analyze_multilabel(df.loc[i, 'description'], df.loc[i, 'contents'])
        df.loc[i, '标签'] = result.get("标签")
        df.loc[i, '判定依据'] = result.get("判定依据")

        # 分批保存并实时更新统计（可选）
        if (i + 1) % SAVE_INTERVAL == 0:
            df.to_excel(OUTPUT_FILE, index=False)
            print(f"\n[系统] 已保存前 {i + 1} 行数据。")

    # 最终保存详细结果
    df.to_excel(OUTPUT_FILE, index=False)

    # 调用统计功能
    save_statistics(df)

    print("\n" + "=" * 30)
    print("所有分析与统计任务完成！")
    print("=" * 30)


if __name__ == "__main__":
    main()