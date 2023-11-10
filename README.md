# Petra
An Animal Behavior Analysis Toolbox based Self-supervised Learning

## Content
1. Setup

### Setup

1. Assuming a fresh [Anaconda](https://www.anaconda.com/download/) distribution, you can install the dependencies with:
``` shell
conda env create -n petra python=3.10 # creating python environment for petra
conda activate petra
```
2. Install the latest [PyTorch](https://pytorch.org/get-started/locally/) following official instruction.

3. Download Petra:
``` shell
git clone https://github.com/howtocodewang/Petra.git
git checkout master # swtich to master branch
```

4. Download and install [Ultralytics](https://docs.ultralytics.com/quickstart/):
``` shell
# Navigate to the cloned directory
cd Petra

# Clone the ultralytics repository
git clone https://github.com/ultralytics/ultralytics

# Navigate to the cloned directory
cd ultralytics

# Install the package in editable mode for development
pip install -e .
```

### Training your own data

1. Create your own dataset as YOLO type：
``` shell
cd Petra/dataset

# After annotation using [Labelme](https://github.com/wkentaro/labelme)
# Split all data into training/validation/test
# Set your raw data path in line 136 in create_dataset.py
dataset_root = "path/to/your/own/dataset/path/"

# Assuming all the images and labels have been saved in the folder named "raw_data"
# The split dataset will be saved in "yolo_type_dataset":
python create_dataset.py
```

2. Transfer raw Labelme format to COCO format:
``` shell
python labelme2coco.py --class_name fish --input path/to/your/yolo_type_dataset/labelme --output path/to/your/yolo_type_dataset/coco --join_num the num of keypoints (10)
```

3. Transfer COCO format to YOLO format:
``` shell
python coco2yolo.py --json_path path/to/your/yolo_type_dataset/coco --save_path path/to/your/yolo_type_dataset/labels
```

Now you have got a split dataset, all images are saved in ```path/to/your/yolo_type_dataset/images```, all labels are saved in ```path/to/your/yolo_type_dataset/labels```
You can start training!

4. Modify the dataset configuration used for training:
``` Shell
# Navigate to the folder saved config files
cd Petra/yolov8-configs

# modify dataset path in line 11, line 12, line 13, line 14 of zebrafish-xlx-pose.yaml as your dataset path setting:
path: /path/to/your/yolo_type_dataset  # dataset root dir
train: images/train  # train images (relative to 'path') 4 images
val: images/val  # val images (relative to 'path') 4 images
test: images/test # test images (optional)

# modfiy the number of keypoints in line 17 of zebrafish-xlx-pose.yaml:
kpt_shape: [10, 3]  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
```

5. Modify the yolov8 model architecture:
``` shell
# modify the number of keypoints in line 6 of yolov8-zebrafish-pose.yaml
kpt_shape: [5, 3]  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)   for epilepsy
```

6. Move the modifed ```zebrafish-xlx-pose.yaml``` to ```path/to/your/Petra/ultralytics/ultralytics/cfg/datasets```
7. Move the modifed ```yolov8-zebrafish-pose.yaml``` to ```path/to/your/Petra/ultralytics/ultralytics/cfg/models/v8```
8. Start training your own dataet:
``` shell
# Navigate to training scripts folder
cd Petra/experiments/src/zebrafish/train

# modify the line 12 of zebrafish_xlx_train.py:
model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training), if you didn't download the pretrained model, it will download automatically
# model = YOLO('yolov8m-zebrafish-pose.yaml').load('/home/wangshuo/Code/SIMIT/Research/Petra/experiments/pretrained_weights/pose/yolov8m-pose.pt')  # build from YAML and transfer weights

# modify line 18， 23， 24
results = model.train(
    data='zebrafish-xlx-pose.yaml',
    epochs=400,
    batch=64,
    imgsz=416,
    device=[0],
    project='path/to/save/log/',
    name='project_name',
    exist_ok=False
)

# Start training
python zebrafish_xlx_train.py
```

### Predict and tracking

1. Pose estimation using trained weights
``` shell
# Navigate to the predict folder

cd Petra/experiments/src/zebrafish/predict/video
python zebrafish_video_pose_pred.py
```
2. Pose tracking using Bytetrack
``` shell
# Navigate to the predict folder

cd Petra/experiments/src/zebrafish/predict/video
python zebrafish_video_pose_track.py
```
