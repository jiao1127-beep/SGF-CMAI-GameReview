import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
import pandas as pd
import re
import jieba
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from loguru import logger
import os
from datetime import datetime
from umap import UMAP
import pyLDAvis
import pyLDAvis.sklearn
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import math
import matplotlib.pyplot as plt

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# TODO:
# 1. 可视化文件加时间（x）
# 2. 困惑度 + 一致性，按照一致性最高保留主题数，绘制折线图(x)
# 3. 保存输出日志到txt（x）
# 4. 优化可视化主题分布图(x)
# 5. 还需要把主题词写入文件(x)

title_keyword_dir="./fig_0118"
# 确保保存结果路径存在
os.makedirs(title_keyword_dir, exist_ok=True)

# 定义日志保存的目录
log_dir = "./app_logs"  # 你可以修改为任意绝对路径，比如 "/var/log/my_app"

# 确保日志目录存在，如果不存在则创建
os.makedirs(log_dir, exist_ok=True)

# 移除 loguru 默认的控制台输出（可选，如果你想保留控制台输出可以删掉这行）
logger.remove()

# 添加日志文件处理器，配置日志保存规则
logger.add(
    # 日志文件路径：指定目录 + 日志文件名
    os.path.join(log_dir, "app_{time:YYYY-MM-DD HH_mm_ss}.log"),
    # 日志级别：只记录 INFO 及以上级别的日志（DEBUG < INFO < WARNING < ERROR < CRITICAL）
    level="INFO",
    # 按文件大小分割：超过 500MB 就新建一个文件
    rotation="500 MB",
    # 按时间滚动：每天 0 点新建一个文件（和 size 二选一或结合使用）
    # rotation="00:00",
    # 保留旧日志的时间：最多保留 7 天的日志
    # retention="7 days",
    # 日志文件压缩格式：旧日志自动压缩为 zip
    compression="zip",
    # 编码格式：避免中文乱码
    encoding="utf-8",
    # 序列化方式：结构化输出（可选，方便后续解析）
    serialize=False,
    # 日志格式：自定义日志内容的展示格式
    format="{time:YYYY-MM-DD HH} | {level: <8} | {name}:{function}:{line} - {message}"
)


# 直接使用不同级别的日志
# logger.debug("这是 DEBUG 级别的调试日志")
# logger.info("这是 INFO 级别的普通运行日志")
# logger.warning("这是 WARNING 级别的警告日志")
# logger.error("这是 ERROR 级别的错误日志")




custom_stopwords = {
    "还可以", "不错", "一般", "挺好", "支持", "感觉",
    "觉得", "比较", "有点", "真的", "就是", "这个",
    "那个", "一个", "没有", "不是", "可以", "游戏","番茄"
}
logger.info("加载完成指定停用词")


def load_stopwords(path):
    with open(path, encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def load_whitelist(path):
    with open(path, encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

# 在 if __name__ == '__main__': 中加载
math_whitelist = load_whitelist("./data/LDA_topicwords.csv")
print(f"成功加载主题白名单，共包含 {len(math_whitelist)} 个核心词")
logger.info(f"成功加载主题白名单，共包含 {len(math_whitelist)} 个核心词")

stopwords = load_stopwords(".\data\hit_stopwords.txt")
stopwords |= custom_stopwords
logger.info("加载哈工大指定停用词")

def clean_text(text):
    text = re.sub(r'[\r\n\t]', ' ', text)
    text = re.sub(r'[^\u4e00-\u9fa5]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    words = jieba.lcut(text)

    words = [
        w for w in words
        if w not in stopwords  # 去停用词
           and len(w) > 1  # 去单字
           and w in math_whitelist  # 只保留白名单内的词
    ]
    return " ".join(words)

# ----------------------  计算主题一致性 (Coherence) ----------------------
def calculate_topic_coherence(topic_model, docs, coherence_type="c_v"):
    """
    计算主题一致性
    :param topic_model: 训练好的BERTopic模型
    :param docs: 原始文本列表
    :param coherence_type: 一致性指标类型，可选 "c_v", "c_npmi", "u_mass"
    :return: 平均主题一致性得分
    """
    # 1. 预处理文本，构建gensim词典
    processed_docs = [doc.split() for doc in docs]  # 简单分词（实际场景建议用更精细的预处理）
    dictionary = Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    # 2. 提取每个主题的关键词（取前10个）
    topic_words = []
    for topic_id in topic_model.get_topics():
        if topic_id == -1:  # 跳过噪声主题
            continue
        words = [word for word, _ in topic_model.get_topic(topic_id)[:10]]
        topic_words.append(words)

    # 3. 计算一致性
    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=processed_docs,
        dictionary=dictionary,
        coherence=coherence_type,
        processes=10
    )
    coherence_scores = coherence_model.get_coherence_per_topic()
    avg_coherence = np.mean(coherence_scores)

    return avg_coherence, coherence_scores




# ----------------------  计算困惑度 (Perplexity) ----------------------
def calculate_topic_perplexity(topic_model, docs, topics, probs):
    """
    计算主题模型的困惑度
    :param topic_model: 训练好的BERTopic模型
    :param docs: 原始文本列表
    :param topics: 每个文档的主题分配结果
    :param probs: 每个文档的主题概率分布
    :return: 困惑度值
    """
    # 1. 预处理：将文本转为词袋模型（用于计算文档长度）
    processed_docs = [doc.split() for doc in docs]
    doc_lengths = [len(doc) for doc in processed_docs]

    # 2. 计算对数似然（所有文档的对数概率之和）
    log_likelihood = 0
    for i, doc in enumerate(docs):
        # 跳过噪声主题（topic_id=-1）
        if topics[i] == -1:
            continue
        # 获取文档在对应主题上的概率（取对数避免下溢）
        topic_prob = probs[i][topics[i]] if len(probs[i]) > topics[i] else 1e-10
        log_prob = math.log(max(topic_prob, 1e-10))  # 防止log(0)
        log_likelihood += log_prob * doc_lengths[i]  # 乘以文档长度（困惑度考虑长度）

    # 3. 计算困惑度：perplexity = exp(-log_likelihood / 总词数)
    total_words = sum(doc_lengths)
    perplexity = math.exp(-log_likelihood / total_words)

    return perplexity



if __name__ == '__main__':

    # 获取当前时间并格式化（年-月-日_时-分-秒，避免冒号等非法字符）
    # 时间格式说明：%Y=年 %m=月 %d=日 %H=时(24小时制) %M=分 %S=秒
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 读取中英文词向量模型
    embedding_model = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )
    logger.info("加载双语种embedding模型")

    # 读取数据
  
    df = pd.read_csv("./data/game_comments_t.csv")
    # print(df.head())
    logger.info("读取数据完成")

    # 数据分类
    # edu_df_bak = df[df["labels"].str.contains("教育", na=False)]
    math_df = df[df["labels"].str.contains(r"数学|算数", regex=True, na=False)]
    # other_df = df[~df["labels"].str.contains("教育", na=False)&~df["labels"].str.contains("数学", na=False)]
    # # print(edu_df.__len__())
    # print(math_df.__len__())
    # # print(other_df.__len__())
    logger.info(f"筛选出对应标签的数据，查看关于数学的数据长度{math_df.__len__()}")

    # 数据清洗
    # edu_df_bak = df[df["labels"].str.contains("教育", na=False)].copy()
    math_df_bak = df[df["labels"].str.contains(r"数学|算数", regex=True, na=False)].copy()
    # edu_docs = edu_df_bak["contents"].dropna().astype(str).tolist()
    math_docs = math_df_bak["contents"].dropna().astype(str).tolist()
    # edu_docs = [clean_text(t) for t in edu_docs if clean_text(t)]
    math_docs = [clean_text(t) for t in math_docs if clean_text(t)]
    logger.info(f"中文文本数据清理完成")


    # 自定义中文分词
    vectorizer_model = CountVectorizer(
        tokenizer=jieba.lcut,
        max_df=0.9,
        min_df=2
    )
    umap_model = UMAP()
    tf = vectorizer_model.fit_transform(math_docs)
    feature_names = vectorizer_model.get_feature_names_out()
    logger.info(f"词汇表大小: {len(feature_names)}")
    logger.info(f"创建自定义词汇模型")

    # 定义主题数量范围（3到10）
    nr_topics_list = list(range(5, 11))  # [5,6,7,8,9,10]
    #nr_topics_list = list(range(5, 20))  # [5,6,7,8,9,10]
    nr_topics_best = 5
    temp_perplexity = math.inf
    temp_avg_coherence = 0
    coherence_scores = []  # 存储每个主题数对应的一致性
    perplexity_scores = []  # 存储每个主题数对应的困惑度o
    BERTopic_list = {}

    # 批量训练模型并计算指标
    for nr_topic in nr_topics_list:
        print(f"正在训练主题数为 {nr_topic} 的BERTopic模型...")

        # 初始化 BERTopic
        # edu_topic_model = BERTopic(
        #     nr_topics=10,
        #     embedding_model=embedding_model,
        #     vectorizer_model=vectorizer_model,
        #     language="chinese",
        #     verbose=True
        # )
        math_topic_model = BERTopic(
            nr_topics=nr_topic,
            embedding_model=embedding_model,
            vectorizer_model=vectorizer_model,
            umap_model=umap_model,
            calculate_probabilities=True,
            language="chinese",
            verbose=True
        )
        BERTopic_list[nr_topic] = math_topic_model
        logger.info(f"初始化 主题数为：{nr_topic} BERTopic ")

        # 训练主题模型
        # edu_topics, _ = edu_topic_model.fit_transform(edu_docs)
        math_topics, math_probs = math_topic_model.fit_transform(math_docs)
        logger.info(f"训练 主题数为：{nr_topic} BERTopic主题模型")

        # 计算困惑度和一致性
        # 计算c_v一致性（最常用）
        avg_coherence, per_topic_coherence = calculate_topic_coherence(math_topic_model, math_docs, "c_v")
        logger.info(f"主题数为：{nr_topic} 平均主题一致性 (c_v): {avg_coherence:.4f}")
        logger.info(f"主题数为：{nr_topic} 各主题一致性: {[round(s, 4) for s in per_topic_coherence][:5]}...")  # 打印前5个主题
        # 计算困惑度
        perplexity = calculate_topic_perplexity(math_topic_model, math_docs, math_topics, math_probs)
        logger.info(f"主题数为：{nr_topic} 主题模型困惑度: {perplexity:.4f}")

        coherence_scores.append((avg_coherence))
        perplexity_scores.append(perplexity)


        if perplexity < temp_perplexity and avg_coherence > temp_avg_coherence:
            temp_avg_coherence = avg_coherence
            temp_perplexity = perplexity
            nr_topics_best = nr_topic

        # 获取主题结果信息
        math_info = math_topic_model.get_topic_info()
        logger.info(f"查看 BERTopic主题模型结果")
        # 拼接完整的文件路径
        file_name = f"math_tile_keyword_{current_time}_{nr_topic}.txt"  # 文件名示例：file_2026-01-08_14-35-20.txt
        file_path = os.path.join(title_keyword_dir, file_name)

        with open(f"{file_path}", "w", encoding="utf8") as f:
            for topic_id in math_info["Topic"]:
                f.write(f"{topic_id}\n")
                if topic_id != -1:
                    # print("数学主题", topic_id, math_topic_model.get_topic(topic_id))
                    # logger.info(f"{math_topic_model.get_topic(topic_id)}")
                    for kw_msg in math_topic_model.get_topic(topic_id):
                        f.write(f"_{nr_topic}关键词：{kw_msg[0]}      频率：{kw_msg[1]}\n")
                        logger.info(f"_{nr_topic}关键词：{kw_msg[0]}      频率：{kw_msg[1]}\n")
            logger.info(f"查看 BERTopic主题模型的主题结果")

        # 查看某主题的代表评论
        # edu_topic_model.get_representative_docs(0)
        math_topic_model.get_representative_docs(0)
        logger.info(f"查看 BERTopic主题模型的某主题的代表评论")

        # 主题分布图
        # edu_fig = edu_topic_model.visualize_topics()

        # edu_fig.write_image("./fig_0118/edu_topics.png", scale=2)
        # edu_fig.write_html("./fig_0118/edu_topics.html")

        math_fig = math_topic_model.visualize_topics()

        math_fig.write_image(f"./fig_0118/math_topics_{current_time}_{nr_topic}.png", scale=1)
        math_fig.write_html(f"./fig_0118/math_topics_{current_time}_{nr_topic}.html")
        logger.info(f"保存BERTopic主题模型的主题分布图")

        # 关键词对比条形图
        # edu_fig = edu_topic_model.visualize_barchart(
        #     top_n_topics=10,
        #     n_words=10
        # )
        #
        # edu_fig.write_image("./fig_0118/edu_keywords.png", scale=2)
        # edu_fig.write_html("./fig_0118/edu_keywords.html")
        math_fig = math_topic_model.visualize_barchart(
            top_n_topics=10,
            n_words=20
        )

        math_fig.write_image(f"./fig_0118/math_keywords_{current_time}_{nr_topic}.png", scale=2)
        math_fig.write_html(f"./fig_0118/math_keywords_{current_time}_{nr_topic}.html")
        logger.info(f"保存BERTopic主题模型的关键词对比条形图")

        # 3 文档散点图可视化
        embeddings = embedding_model.encode(math_docs)
        reduced_embeddings = UMAP().fit_transform(embeddings)
        fig3 = math_topic_model.visualize_documents(math_docs, reduced_embeddings=reduced_embeddings)
        fig3.write_html(f"./fig_0118/docs_visual_{current_time}_{nr_topic}.html")
        # exit()






        # 主题层级结构对比
        # edu_fig = edu_topic_model.visualize_hierarchy()
        # edu_fig.write_image("./fig_0118/edu_hierarchy.png", scale=2)

        math_fig = math_topic_model.visualize_hierarchy()
        math_fig.write_image(f"./fig_0118/math_hierarchy_{current_time}_{nr_topic}.png", scale=2)
        logger.info(f"保存BERTopic主题模型的主题层级结构对比")

        # 主题相似度热力图
        # edu_fig = edu_topic_model.visualize_heatmap()
        # edu_fig.write_image("./fig_0118/edu_heatmap.png", scale=2)

        math_fig = math_topic_model.visualize_heatmap()
        math_fig.write_image(f"./fig_0118/math_heatmap_{current_time}_{nr_topic}.png", scale=2)
        logger.info(f"保存BERTopic主题模型的主题相似度热力图")


    # ----------------------  绘制双轴折线图 ----------------------
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建画布和双轴
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 左轴：一致性（蓝色，越高越好）
    color1 = 'tab:blue'
    ax1.set_xlabel('主题数量 (nr_topics)', fontsize=12)
    ax1.set_ylabel('主题一致性 (c_v)', color=color1, fontsize=12)

    logger.info(f"nr_topics_list:{nr_topics_list}")
    logger.info(f"coherence_scores:{coherence_scores}")


    ax1.plot(nr_topics_list, coherence_scores, 'o-', color=color1, linewidth=2, markersize=8, label='一致性')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(axis='y', alpha=0.3)

    # 右轴：困惑度（红色，越低越好）
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('困惑度', color=color2, fontsize=12)
    logger.info(f"nr_topics_list:{nr_topics_list}")
    logger.info(f"perplexity_scores:{perplexity_scores}")
    ax2.plot(nr_topics_list, perplexity_scores, 's-', color=color2, linewidth=2, markersize=8, label='困惑度')
    ax2.tick_params(axis='y', labelcolor=color2)

    # 添加标题和图例
    fig.suptitle('不同主题数量（3~10）的BERTopic模型性能对比', fontsize=14, fontweight='bold')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    # 标注每个点的数值（可选，增强可读性）
    for x, y in zip(nr_topics_list, coherence_scores):
        ax1.text(x, y + 0.1, f'{y:.3f}', ha='center', va='bottom', fontsize=9, color=color1)
    for x, y in zip(nr_topics_list, perplexity_scores):
        ax2.text(x, y + 5, f'{y:.1f}', ha='center', va='bottom', fontsize=9, color=color2)

    # 调整布局并保存/显示
    fig.tight_layout()
    plt.savefig(f'./{title_keyword_dir}/bertopic_nr_topics_metrics_{current_time}.png', dpi=150, bbox_inches=None,facecolor='white',edgecolor='none')  # 保存高清图
    logger.info(f"保存不同主题数的bertopic指标评价折线变化图")

    # 获取主题数最佳模型
    math_topic_model = BERTopic(
        nr_topics=nr_topics_best,
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        language="chinese",
        verbose=True
    )
    math_topics, math_probs = math_topic_model.fit_transform(math_docs)
    logger.info(f"主题数为：{nr_topic} 模型最优")

    # 优化主题分布图
    # ========== pyLDAvis可视化核心代码 ==========
    # 提取BERTopic的主题-词矩阵等数据，适配pyLDAvis

    logger.info(f"math_topic_model.c_tf_idf_.toarray():{math_topic_model.c_tf_idf_.toarray().shape}")
    logger.info(f"math_topic_model.probabilities_:{math_topic_model.probabilities_.shape}")


    vis_data = pyLDAvis.prepare(
        # 主题-词概率分布（权重）
        topic_term_dists=math_topic_model.c_tf_idf_.toarray()[1:],
        # 文档-主题分布
        doc_topic_dists=math_topic_model.probabilities_ / math_topic_model.probabilities_.sum(axis=1)[:, np.newaxis],
        # 每个词的文档频率（需和vectorizer的词汇表对应）
        doc_lengths=[len(doc) for doc in math_docs],
        # 词汇表
        vocab=math_topic_model.vectorizer_model.get_feature_names_out(),
        # 每个词在所有文档中出现的总次数
        term_frequency=math_topic_model.vectorizer_model.transform(math_docs).sum(axis=0).A1,
        # 调整可视化布局的参数（默认即可）
        sort_topics=False  # 保持BERTopic的主题编号顺序
    )

    # ========== 展示/保存可视化结果 ==========
    # 方式1：在Jupyter Notebook中直接展示
    pyLDAvis.display(vis_data)

    # 方式2：保存为HTML文件（本地打开查看）
    pyLDAvis.save_html(vis_data, f"./{title_keyword_dir}/math_topic_lda_vis_{current_time}.html")


    # 查看主题结果
    # edu_info = edu_topic_model.get_topic_info()
    math_info = math_topic_model.get_topic_info()
    logger.info(f"查看 BERTopic主题模型结果")

    # # print("教育主题数：", len(edu_info) - 1)  # 去掉 -1
    # print("数学主题数：", len(math_info) - 1)

    # 查看某个主题的关键词
    # for topic_id in edu_info["Topic"]:
    #     if topic_id != -1:
    #         # print("教育主题", topic_id, edu_topic_model.get_topic(topic_id))

    # 拼接完整的文件路径
    file_name = f"math_tile_keyword_{current_time}_{nr_topics_best}_best.txt"  # 文件名示例：file_2026-01-08_14-35-20.txt
    file_path = os.path.join(title_keyword_dir, file_name)

    with open(f"{file_path}", "w", encoding="utf8") as f:
        for topic_id in math_info["Topic"]:
            f.write(f"{topic_id}\n")
            if topic_id != -1:
                # print("数学主题", topic_id, math_topic_model.get_topic(topic_id))
                # logger.info(f"{math_topic_model.get_topic(topic_id)}")
                for kw_msg in math_topic_model.get_topic(topic_id):
                    f.write(f"_{nr_topics_best}_best 关键词：{kw_msg[0]}      频率：{kw_msg[1]}\n")
                    logger.info(f"{nr_topics_best}_best 关键词：{kw_msg[0]}      频率：{kw_msg[1]}\n")
        logger.info(f"查看 BERTopic主题模型的主题结果")

    # 查看某主题的代表评论
    # edu_topic_model.get_representative_docs(0)
    math_topic_model.get_representative_docs(0)
    logger.info(f"查看 BERTopic主题模型的某主题的代表评论")

    # 主题分布图
    # edu_fig = edu_topic_model.visualize_topics()

    # edu_fig.write_image("./fig_0118/edu_topics.png", scale=2)
    # edu_fig.write_html("./fig_0118/edu_topics.html")

    math_fig = math_topic_model.visualize_topics()

    math_fig.write_image(f"./fig_0118/math_topics_{current_time}_{nr_topics_best}_best.png", scale=2)
    math_fig.write_html(f"./fig_0118/math_topics_{current_time}_{nr_topics_best}_best.html")
    logger.info(f"保存BERTopic主题模型的主题分布图")

    # 关键词对比条形图
    # edu_fig = edu_topic_model.visualize_barchart(
    #     top_n_topics=10,
    #     n_words=10
    # )
    #
    # edu_fig.write_image("./fig_0118/edu_keywords.png", scale=2)
    # edu_fig.write_html("./fig_0118/edu_keywords.html")
    math_fig = math_topic_model.visualize_barchart(
        top_n_topics=10,
        n_words=20
    )

    math_fig.write_image(f"./fig_0118/math_keywords_{current_time}_{nr_topics_best}_best.png", scale=2)
    math_fig.write_html(f"./fig_0118/math_keywords_{current_time}_{nr_topics_best}_best.html")
    logger.info(f"保存BERTopic主题模型的关键词对比条形图")

    # 主题层级结构对比
    # edu_fig = edu_topic_model.visualize_hierarchy()
    # edu_fig.write_image("./fig_0118/edu_hierarchy.png", scale=2)

    math_fig = math_topic_model.visualize_hierarchy()
    math_fig.write_image(f"./fig_0118/math_hierarchy_{current_time}_{nr_topics_best}_best.png", scale=2)
    logger.info(f"保存BERTopic主题模型的主题层级结构对比")

    # 主题相似度热力图
    # edu_fig = edu_topic_model.visualize_heatmap()
    # edu_fig.write_image("./fig_0118/edu_heatmap.png", scale=2)

    math_fig = math_topic_model.visualize_heatmap()
    math_fig.write_image(f"./fig_0118/math_heatmap_{current_time}_{nr_topics_best}_best.png", scale=2)
    logger.info(f"保存BERTopic主题模型的主题相似度热力图")
