from ultralytics import YOLO

# Load a model
model = YOLO("/runs/pose/train3/weights/best.pt")  # load a pretrained model (recommended for training)
success = model.export(format="onnx", simplify=True)  # export the model to onnx format
assert success