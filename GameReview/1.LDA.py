# 主题分析

import os
import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import gensim.corpora as corpora
import jieba
import jieba.posseg as pseg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyLDAvis
from gensim.models import CoherenceModel
from gensim.models import LdaModel
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# 路径定义

output_path = 'D:/LDA-gamereviews/result_0114_t'
file_path = 'D:/LDA-gamereviews/data'
os.makedirs(output_path, exist_ok=True)
data = pd.read_csv("D:/LDA-gamereviews/data/game_comments_t.csv")
#data = pd.read_excel("D:/LDA-gamereviews/data/LDA_comment-redo.xlsx")
dic_file = "D:/LDA-gamereviews/stop_dict/dict.txt"
stop_file = "D:/LDA-gamereviews/stop_dict/hit_stopwords.txt"

# 全局加载jieba字典和停用词
jieba.load_userdict(dic_file)
stopwords = set(open(stop_file, encoding='utf-8').read().splitlines())

# 1. 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格，使其更具学术感
plt.style.use('ggplot')


# 中文分词函数
def chinese_word_cut(text):
    # 中文文本清洗与分词
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return ""

    flag_list = ['n', 'nz', 'vn', 'v', 'a', 'ad', 'an']
    words = [w.word for w in pseg.cut(text)
             if w.flag in flag_list and w.word not in stopwords and len(w.word) > 1]
    words = [re.sub(u'[^\u4e00-\u9fa5]', '', w) for w in words if w.strip()]
    return " ".join(words)


# 计算一致性分数函数
def calculate_coherence(texts, n_topics, vectorizer):
    """
    计算LDA模型的一致性分数
    """
    # 准备gensim需要的格式
    tokenized_texts = [text.split() for text in texts if text.strip()]

    # 创建字典和语料库
    dictionary = corpora.Dictionary(tokenized_texts)
    # 过滤极端值
    dictionary.filter_extremes(no_below=5, no_above=0.6)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    # 训练LDA模型
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        random_state=42,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )

    # 计算一致性分数
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence='c_v',  # 使用c_v一致性指标，也可以使用'u_mass', 'c_uci', 'c_npmi'
        processes = 10    # 这里需要添加进程控制，否则就会爆内存 导致软件闪退
    )

    return coherence_model.get_coherence()


# 三、LDA分析函数a
def lda_analysis(df, text_col, label):
    # 对指定文本列进行LDA主题分析

    print(f"\n正在分析: {label}文本，共{len(df)}条")
    # 1.分词
    df[f'{text_col}_cutted'] = df[text_col].apply(chinese_word_cut)
    texts = df[f'{text_col}_cutted'].dropna().tolist()

    # 2.向量化
    vectorizer = CountVectorizer(max_features=5000, max_df=0.6, min_df=5)
    tf = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # 3. LDA模型训练
    n_topics = 4
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=50,
        learning_method='batch',
        evaluate_every=5,
        random_state=42,
        n_jobs=15     # 不要设置那么大，会堵塞线程
    )
    lda.fit(tf)
    #
    #  3. 循环获取最佳主题数
    # topic_range = range(2, 11)  # 定义搜索范围，例如 2 到 10 个主题
    # coherences = []
    # models = []
    #
    # print(f"\n开始循环寻找最佳主题数 (范围: {topic_range.start}-{topic_range.stop - 1})...")
    #
    # for i in topic_range:
    #     # 训练当前主题数下的模型
    #     lda_test = LatentDirichletAllocation(
    #         n_components=i,
    #         max_iter=50,
    #         learning_method='batch',
    #         random_state=42,
    #         n_jobs=15  # 使用合适的CPU线程数，避免程序崩溃
    #     )
    #     lda_test.fit(tf)
    #
    #     # 计算当前模型的一致性分数
    #     # 注意：这里调用你之前定义的 calculate_coherence 函数
    #     current_coherence = calculate_coherence(texts, i, vectorizer)
    #     coherences.append(current_coherence)
    #     models.append(lda_test)  # 将模型存入列表以便后续调用
    #
    #     print(f"主题数 {i}: 一致性分数 (c_v) = {current_coherence:.4f}")
    #
    # # 找到一致性分数最高的索引
    # best_idx = np.argmax(coherences)
    # best_n_topics = list(topic_range)[best_idx]
    # best_lda_model = models[best_idx]  # 提取表现最好的模型
    #
    # print(f"\n--- 评估完成 ---")
    # print(f"确定的最佳主题数为: {best_n_topics}，最高一致性分数为: {coherences[best_idx]:.4f}")

    # 将最佳模型赋值给后续逻辑使用的变量
    # lda = best_lda_model
    # n_topics = best_n_topics
    # 4. 输出主题关键词
    print("\n各主题关键词:")
    n_top_words = 15
    topic_word = []
    #加入一列存储
    all_topic_words = []
    for idx, topic in enumerate(lda.components_):
        terms = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topic_word.append("!".join(terms))
        print(f"主题 {idx + 1} : {' | '.join(terms)}")
        all_topic_words.extend(terms)
    words_single_col_df = pd.DataFrame(all_topic_words, columns=['主题词'])

    # 5. 计算一致性分数

    print("\n计算一致性分数...")
    coherence_score = calculate_coherence(texts, n_topics, vectorizer)
    print(f"主题一致性分数 (c_v): {coherence_score:.4f}")

    # 6. 每条文本所属主题
    topics = lda.transform(tf)
    df[f'{label}_主题编号'] = topics.argmax(axis=1)
    df[f'{label}_主题概率'] = topics.max(axis=1)

    # 7. 保存结果
    save_path = os.path.join(output_path, f"LDA_{label}_结果.xlsx")
    df.to_excel(save_path, index=False)
    print(f"{label}结果已保存到: {save_path}")
    # --- 新增：保存主题词为单列 CSV ---
    words_csv_path = os.path.join(output_path, f"LDA_{label}_topicwords.csv")
    words_single_col_df.to_csv(words_csv_path, encoding='utf_8_sig', index=False)

    print(f"单列主题词清单已保存至: {words_csv_path}")

    # 8. 可视化(兼容 pyLDAvis 3.4.1)
    doc_topic_dists = topics
    topic_term_dists = lda.components_ / lda.components_.sum(axis=1)[:, None]
    doc_lengths = tf.sum(axis=1).A1  # 转为一维数组
    vocab = vectorizer.get_feature_names_out()
    term_frequency = np.asarray(tf.sum(axis=0)).ravel()

    # 使用 prepare 手动传入数据
    vis_data = pyLDAvis.prepare(
        topic_term_dists,
        doc_topic_dists,
        doc_lengths.tolist(),
        vocab.tolist(),
        term_frequency.tolist(),
        #sort_topics = False  # <--- 添加这一行，跳过报错的内部排序逻辑
    )
    html_path = os.path.join(output_path, f"LDA_{label}_可视化.html")
    pyLDAvis.save_html(vis_data, html_path)
    print(f"可视化文件保存为: {html_path}")

    # 9. 困惑度曲线和一致性曲线
    plexs = []
    coherences = []
    topic_range = range(2, 11)

    print("\n计算不同主题数下的困惑度和一致性...")
    for i in topic_range:
        lda_test = LatentDirichletAllocation(
            n_components=i,
            max_iter=50,
            learning_method='batch',
            random_state=42
        )
        lda_test.fit(tf)
        plexs.append(lda_test.perplexity(tf))

        # 计算每个主题数的一致性
        coherence_score_i = calculate_coherence(texts, i, vectorizer)
        coherences.append(coherence_score_i)
        print(f"主题数 {i}: 困惑度={lda_test.perplexity(tf):.2f}, 一致性={coherence_score_i:.4f}")

    # 绘制双Y轴图
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 困惑度曲线
    color = 'tab:red'
    ax1.set_xlabel('主题数')
    ax1.set_ylabel('困惑度', color=color)
    line1 = ax1.plot(topic_range, plexs, marker='o', color=color, label='困惑度')
    ax1.tick_params(axis='y', labelcolor=color)

    # 一致性曲线
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('一致性分数', color=color)
    line2 = ax2.plot(topic_range, coherences, marker='s', color=color, label='一致性')
    ax2.tick_params(axis='y', labelcolor=color)

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.title(f'{label} 困惑度与一致性曲线')
    plt.grid(True)
    plt.savefig(os.path.join(output_path, f"{label}_困惑度一致性曲线.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # # 单独的一致性曲线图
    # plt.figure(figsize=(8, 5))
    # plt.plot(topic_range, coherences, marker='s', color='blue', linewidth=2, label='一致性分数')
    # plt.title(f'{label} 主题一致性曲线')
    # plt.xlabel('主题数')
    # plt.ylabel('一致性分数 (c_v)')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(os.path.join(output_path, f"{label}_一致性曲线.png"), dpi=300, bbox_inches='tight')
    # plt.show()

    # 找到最佳主题数（一致性最高）
    best_topic_idx = np.argmax(coherences)
    best_topic_num = topic_range[best_topic_idx]
    best_coherence = coherences[best_topic_idx]
    print(f"\n最佳主题数: {best_topic_num} (一致性分数: {best_coherence:.4f})")

    return df, topic_word, coherence_score, best_topic_num, best_coherence


# 四、主程序
if __name__ == "__main__":
    # 解决中文显示问题
    #plt.rcParams['font.sans-serif'] = ['SimHei']
    #plt.rcParams['axes.unicode_minus'] = False

    # 全局加载jieba字典和停用词
    #jieba.load_userdict(dic_file)
    #stopwords = set(open(stop_file, encoding='utf-8').read().splitlines())
    # 1. 取出初始数据
    media_df = data[['name', 'game_name', 'labels', 'description', 'contents', 'score']].dropna(
        subset=['contents']).copy()

    # 2. 执行去重
    initial_count = len(media_df)

    # 方式 A：简单去重（完全一样的评论）
    media_df = media_df.drop_duplicates(subset=['contents'], keep='first')

    # 方式 B：清洗后去重（更严格，去除只差标点符号或空格的重复）
    # 先创建一个临时列存储纯文本，去重后再删除
    media_df['temp_clean'] = media_df['contents'].str.replace(r'[^\u4e00-\u9fa5]', '', regex=True)
    media_df = media_df.drop_duplicates(subset=['temp_clean'], keep='first')
    media_df = media_df.drop(columns=['temp_clean'])

    final_count = len(media_df)
    print(f"--- 数据预处理 ---")
    print(f"原始数据量: {initial_count}")
    print(f"去重后数据量: {final_count}")
    print(f"删除了 {initial_count - final_count} 条重复记录")
    print(f"------------------")




    # 分别取出媒体文本和评论文本
    #media_df = data[['steam_appid', 'name', 'header_image', 'voted_up','review','review_time', 'review_en']].dropna(subset=['review_en'])

    #media_df = data[['name','game_name','labels','description','contents','score']].dropna(
        #subset=['contents'])
    # public_df = data[['uid','time','comment_text']].dropna(subset=['comment_text'])

    # 分别运行LDA
    media_result, media_topics, media_coherence, media_best_topic, media_best_coherence = lda_analysis(media_df,
                                                                                                       'contents',

                                                                                                       '游戏评论')
    # public_result, public_topics, public_coherence, public_best_topic, public_best_coherence = lda_analysis(public_df, 'comment_text', '公众端')

    # 输出总体结果
    print("\n" + "=" * 50)
    print("分析结果汇总:")
    print(f"游戏评论分析:")
    print(f"  - 最终使用主题数: 5")
    print(f"  - 主题一致性分数: {media_coherence:.4f}")
    print(f"  - 建议最佳主题数: {media_best_topic} (一致性: {media_best_coherence:.4f})")
    # print(f"公众端分析:")
    # print(f"  - 最终使用主题数: 5")
    # print(f"  - 主题一致性分数: {public_coherence:.4f}")
    # print(f"  - 建议最佳主题数: {public_best_topic} (一致性: {public_best_coherence:.4f})")
    print("=" * 50)

    print("\n全部分析完成!结果与可视化文件已保存在result文件夹。")