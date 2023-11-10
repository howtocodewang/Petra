import os
import time
import cv2
import numpy as np
from tqdm import tqdm


def merge_video(src_img_folder_path, dst_vid_saved_path, width=608, height=608, fps=250):
    path = src_img_folder_path # 图片序列所在目录，文件名：0.jpg 1.jpg ...
    dst_path = os.path.join(dst_vid_saved_path, 'zebrafish_lm_' + str(fps) + '.avi') # 生成的视频路径

    filelist = os.listdir(path)
    # filepref = [os.path.splitext(f)[0] for f in filelist]
    # filepref.sort(key = int) # 按数字文件名排序
    # filelist = [f + '.jpg' for f in filepref]
    filelist.sort()

    width = width
    height = height
    fps = fps
    # col_cnt = 2 #显示视频时的图片列数(倍数)
    # vw = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width * col_cnt, height))
    vw = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    for file in tqdm(filelist):
        if file.endswith('.png'):
            file = os.path.join(path, file)
            img = cv2.imread(file)
            # img = np.hstack((img, img))  # 如果并排两列显示
            vw.write(img)

    vw.release()


if __name__ == "__main__":
    src_img_folder_path='/home/wangshuo/Datasets/SIMIT/AnimalPoseBehavior/zebrafish_lm/test_dataset_frame'
    video_saved_folder_path = '/home/wangshuo/Datasets/SIMIT/AnimalPoseBehavior/zebrafish_lm/test_videos'
    merge_video(
        src_img_folder_path=src_img_folder_path,
        dst_vid_saved_path=video_saved_folder_path,
        width=608,
        height=608,
        fps=250
    )
    print('end')
