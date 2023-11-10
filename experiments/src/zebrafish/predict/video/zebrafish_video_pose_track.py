import os.path
from collections import defaultdict
import colorsys
import random
import json

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.utils.plotting import Colors


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


def dict2txt(track_history, save_path):
    results = json.dumps(track_history)
    filename = open(save_path, 'w')  # dict转txt
    # for k, v in track_history.items():
    #     # filename.write(k + ':' + str(v))
    #     filename.write(str(k))
    #     filename.write(': ')
    #     filename.write(v)
    #     filename.write('\n')
    # filename.close()
    filename.write(results)


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


def tracking(
        src_video_path,
        results_saved_path,
        model_weights_path,
        tracker="bytetrack.yaml",
        track_target="keypoints",
        fps=30.0,
        skeleton_type="fish_full",
        palette=None,
        save_trajectory=True
):
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

    # Store the track history
    track_history = defaultdict(lambda: [])
    if save_trajectory:
        track_history_bak = defaultdict(lambda: [])

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(
                frame,
                device=[1],
                persist=True,
                show=True,
                tracker=tracker
            )

            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints = results[0].keypoints.xy.cpu()

            # Visualize the results on the frame
            annotated_frame = results[0].plot(kpt_radius=2, kpt_line=True, labels=False, boxes=False,
                                              category=skeleton_type)

            if track_target == "box_center":
                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if save_trajectory:
                        track_bak = track_history_bak[track_id]
                        track_bak.append((float(x), float(y)))  # x, y center point

                    if len(track) > 30:  # retain 90 tracks for 90 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=palette[track_id],
                                  thickness=2)

                # Display the annotated frame
                cv2.imshow("PETRA Zebrafish Tracking", annotated_frame)

            elif track_target == "keypoints":
                # Plot the tracks
                for kpt, track_id in zip(keypoints, track_ids):
                    x, y = kpt[5]
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if save_trajectory:
                        track_bak = track_history_bak[track_id]
                        track_bak.append((float(x), float(y)))  # x, y center point

                    if len(track) > 30:  # retain 90 tracks for 90 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=palette[track_id],
                                  thickness=2)

                # Display the annotated frame
                cv2.imshow("PETRA Zebrafish Tracking", annotated_frame)

            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            out.write(annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if save_trajectory:
                traj_saved_path = results_saved_path.split('.')[0] + "trajectory_results.txt"
                dict2txt(track_history_bak, traj_saved_path)
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Create color palette
    trajectory_palette = ncolors(10000)
    save_trajectory = True
    # track_type = "box_center"
    track_type = "keypoints"

    # Zebrafish_lm settings
    # proj_path = "/home/wangshuo/Code/SIMIT/Research/Petra/experiments"
    # proj_name = "zebrafish"
    # exp_name = "lm_19_m_w_fliplr_800"

    # Zebrafish_epilepsy settings
    proj_path = "/home/wangshuo/Code/SIMIT/Research/Petra/experiments"
    proj_name = "zebrafish"
    # exp_name = "epileps_5_m"
    exp_name = "xlx_10_m"
    weights_selection = "best.pt"

    model_weights_path = os.path.join(proj_path, proj_name, exp_name, "weights", weights_selection)

    test_type = "video"
    task_type = "track"
    exp_results_saved_folder = os.path.join(proj_path, "vis_results", exp_name, test_type, task_type)

    if not os.path.exists(exp_results_saved_folder):
        os.makedirs(exp_results_saved_folder)
    else:
        pass

    # Open the video file
    # video_path = "/home/wangshuo/Datasets/SIMIT/AnimalPoseBehavior/zebrafish_lm/test_videos/zebrafish_lm_50.avi"
    # video_path = "/home/wangshuo/Datasets/SIMIT/AnimalPoseBehavior/zebrafish_pku/test_video/test_02.mp4"
    # video_path = "/home/wangshuo/Datasets/SIMIT/AnimalPoseBehavior/zebrafish_epilepsy/test_videos/short"
    # video_path = "/home/wangshuo/Datasets/SIMIT/AnimalPoseBehavior/zebrafish_epilepsy/test_videos/short/crop_1_2_20s.mp4"
    video_path = "/home/wangshuo/Datasets/SIMIT/AnimalPoseBehavior/zebrafish_xlx/test_videos/zebrafish_social_Int.mp4"

    if os.path.isdir(video_path):
        video_path_list = getFilelist(video_path, ext=SUPPORTED_FILE)
        for vid_path in video_path_list:
            src_vid_name = vid_path.split('/')[-1]
            results_video_path = os.path.join(exp_results_saved_folder,
                                              "results_" + exp_name + "_" + task_type + "_" + track_type +"_" + src_vid_name)
            tracking(
                src_video_path=vid_path,
                results_saved_path=results_video_path,
                model_weights_path=model_weights_path,
                tracker="bytetrack.yaml",
                track_target="keypoints",
                fps=30.0,
                skeleton_type="fish_epilepsy_5",
                palette=trajectory_palette,
                save_trajectory=True
            )

    elif os.path.isfile(video_path):
        src_vid_name = video_path.split('/')[-1]
        results_video_path = os.path.join(exp_results_saved_folder,
                                          "results_" + exp_name + "_" + task_type + "_" + src_vid_name)
        tracking(
            src_video_path=video_path,
            results_saved_path=results_video_path,
            model_weights_path=model_weights_path,
            tracker="bytetrack.yaml",
            track_target="keypoints",
            fps=30.0,
            skeleton_type="fish_epilepsy_5",
            palette=trajectory_palette,
            save_trajectory=True
        )

    else:
        print("Incorrect input format!")