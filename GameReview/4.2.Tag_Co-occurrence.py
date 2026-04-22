import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np

# 1. 环境配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 颜色定义 ---
PINK_GROUP = {"焦虑表达", "知识难度大", "游戏玩法难度大"}
PINK_FILL, PINK_BORDER = "#d67a83", "#8e3e4d"
GREEN_GROUP = {"放松舒缓", "经典数学游戏", "对学习有帮助"}
GREEN_FILL, GREEN_BORDER = "#45a07a", "#1e4635"
EDGE_PINK, EDGE_GREEN, EDGE_GREY = "#d67a83", "#45a07a", "#b0b0b0"

def run_smart_layout_visualization():
    input_path = r"D:\LDA-gamereviews\data\game_tags_analysis.xlsx"
    output_path = r"D:\LDA-gamereviews\data\tag_topology_smart.pdf"

    if not os.path.exists(input_path):
        print(f"错误：找不到文件 {input_path}")
        return

    # 数据加载
    df = pd.read_excel(input_path)
    G = nx.Graph()
    target_tags = list(PINK_GROUP.union(GREEN_GROUP))
    node_weights = {tag: 0 for tag in target_tags}
    edge_weights = {}

    for _, row in df.iterrows():
        raw_val = str(row.get('标签', ''))
        current_tags = list(set([t.strip() for t in raw_val.replace(',', '，').split('，') if t.strip() in target_tags]))
        for t in current_tags:
            node_weights[t] += 1
        for i in range(len(current_tags)):
            for j in range(i + 1, len(current_tags)):
                pair = tuple(sorted([current_tags[i], current_tags[j]]))
                edge_weights[pair] = edge_weights.get(pair, 0) + 1

    for tag, count in node_weights.items():
        if count > 0: G.add_node(tag, size=count)
    for (u, v), w in edge_weights.items():
        G.add_edge(u, v, weight=w)

    # --- 布局：权重越大，连线越短 ---
    # k 控制节点间距，weight 决定引力
    pos = nx.spring_layout(G, weight='weight', k=1.2, iterations=100, seed=42)

    fig, ax = plt.subplots(figsize=(12, 11), facecolor='white')

    # 准备边数据
    pink_edges, green_edges, grey_edges = [], [], []
    pink_widths, green_widths, grey_widths = [], [], []
    all_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_w = max(all_weights) if all_weights else 1

    for u, v in G.edges():
        w = G[u][v]['weight']
        width = 2 + (w / max_w) * 8
        if u in PINK_GROUP and v in PINK_GROUP:
            pink_edges.append((u, v)); pink_widths.append(width)
        elif u in GREEN_GROUP and v in GREEN_GROUP:
            green_edges.append((u, v)); green_widths.append(width)
        else:
            grey_edges.append((u, v)); grey_widths.append(width)

    # 绘制边和节点
    nx.draw_networkx_edges(G, pos, edgelist=grey_edges, width=grey_widths, edge_color=EDGE_GREY, alpha=0.4, connectionstyle='arc3,rad=0.1', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=pink_edges, width=pink_widths, edge_color=EDGE_PINK, alpha=0.8, connectionstyle='arc3,rad=0.1', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=green_edges, width=green_widths, edge_color=EDGE_GREEN, alpha=0.8, connectionstyle='arc3,rad=0.1', ax=ax)

    for node in G.nodes():
        n_size = np.sqrt(G.nodes[node]['size']) * 20
        n_color = PINK_FILL if node in PINK_GROUP else GREEN_FILL
        n_border = PINK_BORDER if node in PINK_GROUP else GREEN_BORDER
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=n_size, node_color=n_color, edgecolors=n_border, linewidths=2, ax=ax)

    # --- 核心修改：标签动态向外推移 ---
    all_coords = np.array(list(pos.values()))
    center_point = all_coords.mean(axis=0)
    label_pos = {}
    offset_dist = 0.15 # 偏移量，数值越大离节点越远

    for node, coords in pos.items():
        direction = coords - center_point
        norm = np.linalg.norm(direction)
        if norm > 0.01:
            # 沿中心向外的向量方向偏移
            label_pos[node] = coords + (direction / norm) * offset_dist
        else:
            # 中心点附近的节点向上偏移
            label_pos[node] = coords + np.array([0, offset_dist])

    nx.draw_networkx_labels(G, label_pos, font_size=13, font_family='SimHei', font_weight='bold', ax=ax)

    ax.set_axis_off()
    plt.margins(0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    run_smart_layout_visualization()