from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import torch
import cv2


""" Train """
# Load a model
# model = YOLO('yolov8n-pose.yaml')  # build a new model from YAML
# model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8m-zebrafish-pose.yaml').load('experiments/pretrained_weights/pose/yolov8m-pose.pt')  # build from YAML and transfer weights
# model = YOLO('yolov8n-fish-pose.yaml')

# Train the model
# results = model.train(data='zebrafish-pku-pose.yaml', epochs=100, imgsz=256, device=[0, 1])
results = model.train(
    data='zebrafish-lm-pose.yaml',
    epochs=400,
    batch=64,
    imgsz=512,
    device=[0],
    project='zebrafish',
    name='lm_19_m',
    exist_ok=False
)

print("debug")