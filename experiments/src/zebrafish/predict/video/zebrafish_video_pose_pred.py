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


SUPPORTED_FILE = 'avi', 'mp4'


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


def getFilelist(folder_path, ext=SUPPORTED_FILE):
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


def inference(src_video_path, model_weights_path, results_saved_path, fps=30.0, skeleton_type="fish_full"):
    # Open the video file
    cap = cv2.VideoCapture(src_video_path)

    # Frame Rate
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))

    # Width
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # height
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # total number of frames
    frame_counter = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(results_saved_path, fourcc, fps, (width, height))

    # Load the YOLOv8 model
    model = YOLO(model_weights_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            start_time = time.time()

            # Run pose detection on the frame
            results = model.predict(
                frame,
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

            # Visualize the results on the frame
            annotated_frame = results[0].plot(kpt_radius=2, kpt_line=True, labels=False, boxes=False, category=skeleton_type)
            keypoints = results[0].keypoints.xy.cpu()

            # print keypoints index number and x,y coordinates
            for fish_id, kpts in enumerate(keypoints):
                # for idx, kpt in enumerate(keypoints[0]):
                for idx, kpt in enumerate(kpts):
                    # for idx, kpt in enumerate(results1[0].keypoints[0]):
                    # for idx, kpt in enumerate(results1[0].keypoints[0]):
                    x = int(float(kpt[0]))
                    y = int(float(kpt[1]))

                    # cv2.putText(annotated_frame, f"{idx}:({x}, {y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                    #             cv2.LINE_AA)

                    # cv2.putText(annotated_frame, f"{idx}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                    #             cv2.LINE_AA)

            end_time = time.time()

            # fps = 1 / (end_time - start_time)

            # cv2.putText(annotated_frame, "FPS :" + str(int(fps)), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2,
            #             (255, 0, 255), 1, cv2.LINE_AA)

            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            out.write(annotated_frame)

            # Display the annotated frame
            cv2.imshow("PETRA Zebrafish Pose Estimation Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Zebrafish_lm settings
    # proj_path = "/home/wangshuo/Code/SIMIT/Research/Petra/experiments"
    # proj_name = "zebrafish"
    # exp_name = "lm_19_m_w_fliplr_800"

    # Zebrafish_epilepsy settings
    proj_path = "/home/wangshuo/Code/SIMIT/Research/Petra/experiments"
    proj_name = "zebrafish"
    exp_name = "epileps_5_m"
    weights_selection = "best.pt"

    model_weights_path = os.path.join(proj_path, proj_name, exp_name, "weights", weights_selection)

    test_type = "video"
    task_type = "pose"
    exp_results_saved_folder = os.path.join(proj_path, "vis_results", exp_name, test_type, task_type)

    if not os.path.exists(exp_results_saved_folder):
        os.makedirs(exp_results_saved_folder)
    else:
        pass

    # # Load the YOLOv8 model
    # model = YOLO(model_weights_path)

    # Open the video file
    # video_path = "/home/wangshuo/Datasets/SIMIT/AnimalPoseBehavior/zebrafish_lm/test_videos/zebrafish_lm_50.avi"
    # video_path = "/home/wangshuo/Datasets/SIMIT/AnimalPoseBehavior/zebrafish_epilepsy/test_videos/short"
    video_path = "/home/wangshuo/Datasets/SIMIT/AnimalPoseBehavior/zebrafish_epilepsy/test_videos/short/crop_1_2_20s.mp4"

    if os.path.isdir(video_path):
        video_path_list = getFilelist(video_path, ext=SUPPORTED_FILE)
        for vid_path in video_path_list:
            src_vid_name = vid_path.split('/')[-1]
            results_video_path = os.path.join(exp_results_saved_folder, "results_" + exp_name + "_" + task_type
                                              + "_" + src_vid_name)
            inference(
                src_video_path=vid_path,
                model_weights_path=model_weights_path,
                results_saved_path=results_video_path
            )

    elif os.path.isfile(video_path):
        # src_vid_name = video_path.split('/')[-1][:-4]
        src_vid_name = video_path.split('/')[-1]
        results_video_path = os.path.join(exp_results_saved_folder, "results_" + exp_name + "_" + task_type
                                          + "_" + src_vid_name)
        inference(
            src_video_path=video_path,
            model_weights_path=model_weights_path,
            results_saved_path=results_video_path
        )

    else:
        print("Incorrect input format!")