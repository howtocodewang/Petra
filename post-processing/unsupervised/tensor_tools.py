import tensortools as tt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import pandas
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score, silhouette_score


def align_to_y_axis(data):
    """
    Align the first node with respect to the 14th node to the Y-axis for all-time points to ensure the
    fish is facing North.

    Parameters:
        data (numpy.ndarray): The input data with shape (N * C * T * V) (e.g. 41390 bouts, 3 channels,
    82 frames, 19 nodes).

    Returns:
        numpy.ndarray: The aligned data with the same shape as input.
    """

    # 获取第1个和第14个点的X, Y坐标在第一个时间点上
    x_1 = data[:, 0, 0, 0]
    y_1 = data[:, 1, 0, 0]
    x_14 = data[:, 0, 0, 13]
    y_14 = data[:, 1, 0, 13]

    # 计算两点之间的角度
    angle_to_y_axis = np.pi/2
    angle = angle_to_y_axis - np.arctan2(y_14 - y_1, x_14 - x_1)

    # 为所有的点计算旋转后的坐标
    x = data[:, 0, :, :]
    y = data[:, 1, :, :]

    rotated_x = x * np.cos(angle)[:, np.newaxis, np.newaxis] - y * np.sin(angle)[:, np.newaxis, np.newaxis]
    rotated_y = x * np.sin(angle)[:, np.newaxis, np.newaxis] + y * np.cos(angle)[:, np.newaxis, np.newaxis]

    rotated_data = data.copy()
    rotated_data[:, 0, :, :] = rotated_x
    rotated_data[:, 1, :, :] = rotated_y

    return rotated_data


def compute_relative_angles(data):
    """
    Compute the angles of each point relative to the 14th point.

    Parameters:
        data (numpy.ndarray): The input 4-D data array.

    Returns:
        numpy.ndarray: The output 4-D array with the original x, y changed to the angles of each point relative to the
    14th point.
    """

    # 获取X, Y坐标
    x_coords = data[:, 0, :, :]
    y_coords = data[:, 1, :, :]

    # 获取第14个点的X, Y坐标
    x_14 = x_coords[:, :, 13]
    y_14 = y_coords[:, :, 13]

    # 计算与第14个点的X, Y坐标差
    delta_x = x_coords - np.expand_dims(x_14, axis=2)
    delta_y = y_coords - np.expand_dims(y_14, axis=2)

    # 使用arctan2计算角度
    angles_rad = np.arctan2(delta_y, delta_x)
    # angles_deg = np.degrees(angles_rad)  # 转换为度数

    # 调整形状以匹配输出格式
    return angles_rad[:, np.newaxis, :, :]


def tensor_decomposition(data, rank):
    """
    Perform tensor decomposition on the input data and visualize the decomposition results.

    Parameters:
        data (numpy.array): The four-dimensional tensor data to be decomposed.
        rank (int): The required rank of tensor decomposition.

    Returns:
        factors (numpy.array): The decomposed factors array.
    """

    # Fit an ensemble of models, 4 random replicates / optimization runs per model rank
    ensemble = tt.Ensemble(fit_method="cp_als")
    ensemble.fit(data, ranks=range(1, rank+1), replicates=4)

    fig, axes = plt.subplots(1, 2)
    tt.plot_objective(ensemble, ax=axes[0])  # plot reconstruction error as a function of num components.
    tt.plot_similarity(ensemble, ax=axes[1])  # plot model similarity as a function of num components.
    fig.tight_layout()

    # 里面可以取出来三个矩阵，分别是 swim_bout_factor, time_factor 和 node_factor
    factors = ensemble.factors(rank)[0]

    plt.show()

    return factors


def hierarchical_agglomerative_clustering(swim_bout_factor, n_clusters, distance, link):
    """
    Performs hierarchical clustering and visualizes the results using a dendrogram and a heatmap. Also, compute
    Davies-Bouldin and Silhouette scores for the clustering.

    Parameters:
        swim_bout_factor (numpy.array): The bout matrix to be clustered.
        n_clusters (int): The number of clusters to form.
        distance (str): The distance metric to use for the clustering.
        link (str): The linkage criterion to use for the clustering.

    Returns:
        labels (numpy.array): Cluster labels for each bout.
    """

    # 使用AgglomerativeClustering进行聚类
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric=distance, linkage=link)
    label = clustering.fit_predict(swim_bout_factor)

    # 打印一些评价指标
    db_score = davies_bouldin_score(swim_bout_factor, label)
    s_score = silhouette_score(swim_bout_factor, label)
    print(f'Davies-Bouldin Score: {db_score}')
    print(f'Silhouette Score: {s_score}')

    # 使用linkage进行聚类并获取排序
    linked = linkage(swim_bout_factor, link)
    order = leaves_list(linked)
    sorted_data = swim_bout_factor[order]

    # 设置图形布局
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 5], wspace=0.05)

    # 绘制dendrogram
    ax0 = fig.add_subplot(gs[0])
    dendrogram(linked, orientation='left', no_labels=True, ax=ax0, truncate_mode='level', p=5)
    ax0.set_axis_off()

    # 绘制热力图
    ax1 = fig.add_subplot(gs[1])
    sns.heatmap(sorted_data, cmap='viridis', yticklabels=False, cbar_kws={'label': 'Weight'}, ax=ax1)
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Swim bout number', labelpad=10)
    ax1.yaxis.set_label_position("right")  # 将ylabel移到图的右侧
    ax1.set_title('Hierarchical clustering of swim bouts')

    plt.show()

    # 返回聚类标签
    return label


if __name__ == '__main__':
    use_TC = False

    if use_TC:
        metadata = pandas.read_hdf(
            '/home/wangshuo/Datasets/SIMIT/AnimalPoseBehavior/zebrafish_PoseR/New Data in Mullen et al., '
            '2023/ZebTensor/bout_metadata_wclusters30.h5')
        quality = metadata['quality']
        good_indices = quality[quality == 'good'].index
        good_indices_list = good_indices.tolist()

        fishdata = np.load('/home/wangshuo/Datasets/SIMIT/AnimalPoseBehavior/zebrafish_PoseR/New Data in Mullen et al., '
                           '2023/ZebTensor/bouts.npy')
        fishdata = np.squeeze(fishdata, axis=-1)  # 移除尺寸为1的维度，因为只有一条鱼，不需要这个维度

        # 计算每个节点相对于第14个节点的角度
        fishdata_angles = compute_relative_angles(fishdata)
        # 移除尺寸为1的维度，因为角度把xy坐标换成角度以后每个节点只剩一个数了
        fishdata = np.squeeze(fishdata_angles, axis=1)  # fishdata是一个 N * T * V 的数组
        fishdata = fishdata[good_indices_list, :, :]  # 只取good quality的回合

        # tensor decomposition
        n_rank = 10  # 10 components
        tensor_factors = tensor_decomposition(fishdata, n_rank)

        # 存下来张量分解以后的矩阵，再做聚类就可以直接聚类，不用重新做分解了
        factors_names = ['bout_factors', 'frame_factors', 'node_factors']
        for idx, name in enumerate(factors_names):
            np.save(name, tensor_factors[idx])

        # 用swim_bout_factor来做聚类
        swim_bout_matrix = tensor_factors[0]

    elif not use_TC:
        swim_bout_matrix = np.load("bout_factors.npy")

    # hierarchical agglomerative clustering 聚类并画出树状图和热力图
    num_clusters = 30  # 希望聚出多少类
    metric = 'euclidean'  # 'euclidean', 'manhattan', 'cosine' 可选
    link = 'ward'  # 'ward', 'complete', 'average', 'single' 可选

    if use_TC:
        labels = hierarchical_agglomerative_clustering(swim_bout_matrix, num_clusters, metric, link)  # 获得聚类标签并画图
        np.save('labels.npy', labels)

    elif not use_TC:
        labels = np.load("labels.npy")

    # UMAP 降维到3维
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric=metric, n_components=3, spread=3)
    embedding = reducer.fit_transform(swim_bout_matrix)

    # 可视化UMAP结果
    figure = plt.figure()
    ax = figure.add_subplot(111, projection='3d')  # 创建3D子图

    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=labels, cmap='Spectral', s=0.1)
    ax.set_title('3D UMAP projection of the N x TC matrix')
    figure.colorbar(scatter)

    plt.show()
