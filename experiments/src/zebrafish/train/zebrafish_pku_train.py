from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import torch
import cv2


""" Train """
# Load a model
# model = YOLO('yolov8n-pose.yaml')  # build a new model from YAML
# model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n-zebrafish-pose.yaml').load('experiments/pretrained_weights/pose/yolov8n-pose.pt')  # build from YAML and transfer weights
# model = YOLO('yolov8n-fish-pose.yaml')

# Train the model
# results = model.train(data='zebrafish-pku-pose.yaml', epochs=100, imgsz=256, device=[0, 1])
results = model.train(
    data='zebrafish-pku-pose.yaml',
    epochs=400,
    batch=16,
    imgsz=256,
    device=[0, 1],
    project='zebrafish',
    name='pku_9_n',
    exist_ok=False
)

print("debug")