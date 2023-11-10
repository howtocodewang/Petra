import os

from ultralytics import YOLO
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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


if __name__ == "__main__":
    # Load the YOLOv8 model

    proj_path = "/home/wangshuo/Code/SIMIT/Research/Petra/experiments"
    proj_name = "zebrafish"
    # exp_name = "lm_19_m_w_fliplr_800"
    # weights_selection = "best.pt"
    # model_weights_path = os.path.join(proj_path, proj_name, exp_name, "weights", weights_selection)

    # model = YOLO(model_weights_path)

    # model = YOLO("/home/wangshuo/Code/SIMIT/Research/Petra/ultralytics/runs/pose/train3/weights/best.pt")

    # source = "/home/wangshuo/Datasets/SIMIT/AnimalPoseBehavior/zebrafish_pku/E-test behavior data XUll 20231020/ES_10V_16x_Gcamp6s-mRubby_7dpf_32/ES_10V_16x_Gcamp6s-mRubby_7dpf_32_2023_06_03__03_39_19.avi"
    video_folder_path = "/home/wangshuo/Datasets/SIMIT/AnimalPoseBehavior/zebrafish_pku/E-test behavior data XUll 20231020"
    video_list = getFilelist(video_folder_path, ext='avi')

    for vid in  video_list:
        model = YOLO("/home/wangshuo/Code/SIMIT/Research/Petra/ultralytics/runs/pose/train3/weights/best.pt")
        source = vid
        vid_name = source.split("/")[-1][:-4]
        saved_folder = os.path.join(proj_path, "vis_results")

        results = model.predict(
            source,
            project=saved_folder,
            name=vid_name,
            device=[0],
            save=True,
            save_txt=True,
            show=False,
            show_labels=False,
            show_conf=False,
            line_width=0,
            stream=False,
            boxes=False,
        )



    print("debug")