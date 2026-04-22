import pandas as pd
from openai import OpenAI
import json
import os
from tqdm import tqdm
import time

# --- 1. 配置区域 ---
client = OpenAI(
    api_key="",
    base_url=""
)

INPUT_FILE = "./data/LDA_comments_t.xlsx"
OUTPUT_FILE = "./data/game_comments_analysis.xlsx"
SAVE_INTERVAL = 50
MODEL_NAME = "gemini-2.5-flash"

PROMPT_TEMPLATE = """
请将以下游戏评论归类为：焦虑表达、情绪缓解 或 中性。
输出格式（严格 JSON）：
{{
  "情绪状态": "类别名称",
  "解释": "简要说明判断依据"
}}
内容："{content}"
"""


def analyze_text(text):
    if not text or pd.isna(text):
        return {"情绪状态": "中性", "解释": "空内容"}
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个专业的情感分析助手。"},
                {"role": "user", "content": PROMPT_TEMPLATE.format(content=text)}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"\n[错误] 处理出错: {e}")
        time.sleep(2)
        return {"情绪状态": "处理失败", "解释": str(e)}


# --- 2. 执行逻辑 ---
def main():
    # --- 核心修改：检查并加载已有结果 ---
    if os.path.exists(OUTPUT_FILE):
        print(f"检测到已存在的输出文件: {OUTPUT_FILE}，正在加载进度...")
        df = pd.read_excel(OUTPUT_FILE)
    else:
        print(f"未检测到现有结果，读取原始文件: {INPUT_FILE}")
        df = pd.read_excel(INPUT_FILE)

    # 确保必要的列存在
    if '情绪状态' not in df.columns:
        df['情绪状态'] = None
    if '解释' not in df.columns:
        df['解释'] = None

    # 计算待处理的任务数量
    todo_df = df[df['情绪状态'].isna()]
    total_todo = len(todo_df)

    if total_todo == 0:
        print("所有数据已分析完成，无需重复执行。")
        return

    print(f"共有 {len(df)} 条数据，其中 {total_todo} 条待处理。")

    # 开始遍历
    for i in tqdm(df.index):
        # 核心逻辑：如果已经有结果，直接跳过
        if pd.notna(df.loc[i, '情绪状态']):
            continue

        result = analyze_text(df.loc[i, 'contents'])

        # 如果 result 是列表，取第一个元素；如果是字典，直接使用
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
        elif not isinstance(result, dict):
            result = {"情绪状态": "格式错误", "解释": "API返回非预期格式"}
        # ------------------
        df.loc[i, '情绪状态'] = result.get("情绪状态")
        df.loc[i, '解释'] = result.get("解释")

        # 定期保存
        if (i + 1) % SAVE_INTERVAL == 0:
            df.to_excel(OUTPUT_FILE, index=False)
            print(f"\n[系统提示] 进度已同步至第 {i + 1} 行并保存。")

    # 最终保存
    df.to_excel(OUTPUT_FILE, index=False)
    print("-" * 30)
    print(f"处理完成！最终结果保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
