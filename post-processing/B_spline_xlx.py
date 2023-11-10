from scipy.interpolate import UnivariateSpline as us
import numpy as np
import matplotlib.pyplot as plt
import json
import os

def separate_data(raw_data):
    """
    推理完的原始数据是399帧一起的，要先拆分开一帧一帧的。

    参数：
    raw_data (list of dict): 包含原始数据的列表，每个元素是一个字典，包含 'frame_id' 和 'keypoints' 字段。

    返回：
    unpacked_data (list of dict): 拆分成单帧后的数据列表，每个元素是一个字典，包含 'frame_id' 和 'keypoints' 字段。
    """

    unpacked_data = []
    frame_id_counter = 1

    for item in raw_data:
        frame_ids = item['frame_id']
        keypoints = item['keypoints']

        for i, frame_id in enumerate(frame_ids):
            new_dict = {'frame_id': frame_id_counter, 'keypoints': keypoints[i]}
            unpacked_data.append(new_dict)
            frame_id_counter += 1

    return unpacked_data


if __name__ == '__main__':
    # 读取 JSON 关键点坐标文件
    with open('/Users/lanxinxu/Desktop/vis_json/0605/ES_10V_16x_7dpf_02.json', 'r') as file:
        data = json.load(file)

    # 推理视频是399帧一起推的，先把所有399帧一起数据都拆成单帧的
    data = separate_data(data)  # 如果数据本来就是单帧的就不用分开

    # 创建保存可视化图像的文件夹
    save_folder = '/Users/lanxinxu/Desktop/vis_json/0605/test/ES_10V_16x_7dpf_02/curves'
    os.makedirs(save_folder, exist_ok=True)

    # 尾巴上点的个数
    n_tail_coords = 9

    # 设置插值拟合的smoothing_factor，越大则曲线越平滑
    smoothing_factor = 20

    # 提取每张图片的关键点信息，并绘制并保存对应的图像
    for i, image_data in enumerate(data):
        keypoints = image_data['keypoints']

        # 获取x和y的坐标
        x = np.array([kp[0] for kp in keypoints])
        y = np.array([kp[1] for kp in keypoints])

        # 改变关键点的坐标顺序，原始数据的顺序是【195372468】，改成【123456789】
        x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8] = \
            x[0], x[5], x[3], x[6], x[2], x[7], x[4], x[8], x[1]
        y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8] = \
            y[0], y[5], y[3], y[6], y[2], y[7], y[4], y[8], y[1]

        # 拟合过程
        t = np.zeros(n_tail_coords)
        t[1:] = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)
        t = np.cumsum(t)
        t /= t[-1]

        uq = np.linspace(int(np.min(x)), int(np.max(x)), 101)
        nt = np.linspace(0, 1, 100)

        # 三次样条插值拟合
        spline_y_coords = us(t, y, k=3, s=smoothing_factor)(nt)
        spline_x_coords = us(t, x, k=3, s=smoothing_factor)(nt)

        # 创建新的图像
        plt.figure()

        # 在拟合好的曲线上均匀生成9个点
        spline_coords = np.array([spline_y_coords, spline_x_coords])
        spline_nums = np.linspace(0, spline_coords.shape[1] - 1, n_tail_coords).astype(int)

        spline_coords = spline_coords[:, spline_nums]
        y_new = spline_coords[0].tolist()
        x_new = spline_coords[1].tolist()

        # 生成渐变色列表
        colors = np.linspace(0, 1, len(x_new))

        # 绘制拟合的曲线
        plt.plot(spline_x_coords, spline_y_coords, c='lightgray', zorder=1)
        # 绘制均匀生成的9个点
        plt.scatter(x_new[:], y_new[:], 25, c=colors, cmap='rainbow', zorder=2)

        # 设置图像标题和保存路径
        plt.title(f'Line {str(i + 1).zfill(4)}')
        save_path = os.path.join(save_folder, f'{str(i + 1).zfill(4)}.png')

        # 设置坐标轴的纵横比
        plt.axis('equal')

        # 保存图像
        plt.savefig(save_path)

        # 关闭当前图像
        plt.close()

        print(f'曲线绘制完成：{save_path}')