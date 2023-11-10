from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import torch
import cv2

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    model = YOLO("/home/wangshuo/Code/SIMIT/Research/Petra/ultralytics/runs/pose/train3/weights/best.pt")
    # model.to(device)

    source = "/home/wangshuo/Datasets/SIMIT/AnimalPoseBehavior/zebrafish_pku/E-test behavior data XUll 20231020/ES_10V_16x_Gcamp6s_mRubby_7dpf_42/ES_10V_16x_Gcamp6s_mRubby_7dpf_42_2023_09_20__22_05_56.avi"

    results = model.predict(
        source,
        device=[0],
        save=True,
        save_txt=True,
        show=False,
        show_labels=False,
        show_conf=False,
        line_width=0,
        stream=False,
        boxes=False
    )
    # results = model(source=source)

    print("debug")