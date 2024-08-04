import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from .typing_utils import *

def visualize_consistency(embeds: List[Tensor], labels: List[Tensor]):
    np.random.seed(42)
    embeddings = np.concatenate(embeds)

    # 执行t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # 定义两类的颜色
    colors = ['blue', 'orange']  # 蓝色和橙色分别代表两类

    # 可视化，使用标签区分不同颜色
    plt.figure(figsize=(10, 7))
    for label in np.unique(labels):
        indices = np.where(labels == label)
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], c=colors[label], label=f'Class {label}', s=5)
    plt.legend()
    plt.title('t-SNE Visualization of Graph Embeddings')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()