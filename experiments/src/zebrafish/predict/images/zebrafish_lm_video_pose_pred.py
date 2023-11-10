from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import torch
import cv2


# """ val """
# model = YOLO("/home/wangshuo/Code/SIMIT/Research/Petra/ultralytics/runs/pose/train3/weights/best.pt")
# metrics = model.val()
# metrics.pose.map

""" Predict """
# model=YOLO("/home/wangshuo/Code/SIMIT/Research/Petra/ultralytics/runs/pose/train2/weights/best.pt")
# test_img = Image.open("/home/wangshuo/Code/SIMIT/Research/Petra/ultralytics/ultralytics/assets/test.jpeg")
# results = model.predict(source=test_img, save=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = YOLO("/home/wangshuo/Code/SIMIT/Research/Petra/experiments/zebrafish/epileps_5_m/weights/best.pt")
# model = YOLO("/home/wangshuo/Code/SIMIT/Research/Petra/ultralytics/runs/pose/train3/weights/best.pt")
model.to(device)

source = "/home/wangshuo/Datasets/SIMIT/AnimalPoseBehavior/zebrafish_epilepsy/yolo_type_dataset/images/test"
# source = "/home/wangshuo/Datasets/SIMIT/AnimalPoseBehavior/zebrafish_lm/test_videos/zebrafish_lm_50.avi"
# source = "/home/wangshuo/Datasets/SIMIT/AnimalPoseBehavior/zebrafish_pku/E-test behavior data XUll 20231020/ES_10V_16x_Gcamp6s-mRubby_7dpf_31/ES_10V_16x_Gcamp6s-mRubby_7dpf_31_2023_06_03__01_10_30.avi"
# source = "/home/wangshuo/Datasets/SIMIT/AnimalPoseBehavior/zebrafish_pku/yolo_type_dataset/images/test/0278.jpg"

results = model.predict(
    source,
    device=[0],
    save=True,
    save_txt=True,
    show=False,
    show_labels=False,
    show_conf=False,
    line_width=0,
    boxes=False
)
# results = model(source=source)

print("debug")