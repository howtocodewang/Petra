import os.path
from collections import defaultdict
import colorsys
import random
import json
import time

import cv2
import imageio
import numpy as np

from ultralytics import YOLO


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors


def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors


def getFilelist(folder_path, ext='avi'):
    filelist = []
    for root, dirs, files in os.walk((folder_path)):
        for filename in files:
            if filename.endswith(ext) and not filename.startswith('.'):
                # print("root", root)
                # print("dirs", dirs)
                # print("files", filename)
                filepath = os.path.join(root, filename)
                filelist.append(filepath)

    return filelist


def inference(src_img_path, model, results_saved_path, draw_skeleton, skeleton_type="fish_full"):
    # Loop through the video frames
    start_time = time.time()

    # Run pose detection on the frame
    results = model.predict(
        src_img_path,
        device=[1],
        save=True,
        save_txt=True,
        show=False,
        show_labels=False,
        show_conf=False,
        line_width=0,
        stream=False,
        boxes=False
    )

    # Visualize the results on the image
    annotated_image = results[0].plot(
        line_width=1,
        kpt_radius=2,
        kpt_line=draw_skeleton,
        labels=False,
        boxes=False,
        category=skeleton_type
    )
    keypoints = results[0].keypoints.xy.cpu()

    # print keypoints index number and x,y coordinates
    for fish_id, kpts in enumerate(keypoints):
        # for idx, kpt in enumerate(keypoints[0]):
        for idx, kpt in enumerate(kpts):
            # for idx, kpt in enumerate(results1[0].keypoints[0]):
            x = int(float(kpt[0]))
            y = int(float(kpt[1]))

            # cv2.putText(annotated_frame, f"{idx}:({x}, {y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
            #             cv2.LINE_AA)

            # cv2.putText(annotated_image, f"{idx}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
            #             cv2.LINE_AA)

    end_time = time.time()

    # fps = 1 / (end_time - start_time)

    # cv2.putText(annotated_frame, "FPS :" + str(int(fps)), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2,
    #             (255, 0, 255), 1, cv2.LINE_AA)

    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Save result image
    cv2.imwrite(results_saved_path, annotated_image)

    # Display the annotated frame
    cv2.imshow("PETRA Zebrafish Pose Estimation Tracking", annotated_image)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    proj_path = "/home/wangshuo/Code/SIMIT/Research/Petra/experiments"
    proj_name = "zebrafish"
    exp_name = "xlx_10_m"
    weights_selection = "best.pt"
    model_weights_path = os.path.join(proj_path, proj_name, exp_name, "weights", weights_selection)

    test_type = "images_m"
    exp_results_saved_folder = os.path.join(proj_path, "vis_results", exp_name, test_type)

    if not os.path.exists(exp_results_saved_folder):
        os.makedirs(exp_results_saved_folder)
    else:
        pass

    # Load the YOLOv8 model
    model = YOLO(model_weights_path)

    # Open the image folder
    image_folder = "/home/wangshuo/Datasets/SIMIT/AnimalPoseBehavior/zebrafish_xlx/yolo_type_dataset/images/test"

    if os.path.isdir(image_folder):
        img_path_list = getFilelist(image_folder, ext='jpg')
        for img_path in img_path_list:
            src_img_name = img_path.split('/')[-1]
            results_image_path = os.path.join(exp_results_saved_folder, "results_" + exp_name + "_" + src_img_name)
            inference(
                src_img_path=img_path,
                model=model,
                results_saved_path=results_image_path,
                draw_skeleton=False,
                skeleton_type="fish_xlx_10"
            )

    else:
        print("Incorrect input format!")