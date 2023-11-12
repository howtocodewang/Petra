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

1. Modify visualization functions to draw keypoints and skeletons for animals, Navigate to the ```Petra/ultralytics/ultralytics/engine/results.py```
``` shell
def plot(
        self,
        conf=True,
        line_width=None,
        font_size=None,
        font='Arial.ttf',
        pil=False,
        img=None,
        im_gpu=None,
        kpt_radius=5,
        kpt_line=True,
        labels=True,
        boxes=True,
        masks=True,
        probs=True,
        category='person' # add this line
    ):
```
``` shell
annotator = Annotator(
            deepcopy(self.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
            example=names,
            category=category) # add this line
```
2. Navigate to the ```Petra/ultralytics/ultralytics/utils/plotting.py```
``` shell
def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc', category='person' # add this):
```
``` shell
# add this lines for visualization
# Pose
if category == 'person':
    self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                     [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    self.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

elif category == 'fish_tail':
    self.skeleton = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8]]

    self.limb_color = colors.pose_palette[[9, 9, 7, 7, 0, 0, 16, 16]]
    self.kpt_color = colors.pose_palette[[16, 16, 16, 0, 0, 0, 9, 9, 9]]

elif category == 'fish_full':
    self.skeleton = [[1, 18], [1, 14], [18, 17], [18, 16], [17, 19], [16, 19], [14, 12], [14, 13],
                     [12, 15], [13, 15], [19, 11], [15, 11], [11, 2], [2, 6], [6, 5], [5, 7], [7, 3],
                     [3, 9], [9, 8], [8, 10], [10, 4]]

    self.limb_color = colors.pose_palette[[9, 9, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 16, 16, 16, 16, 16, 16, 16, 16, 16]]
    self.kpt_color = colors.pose_palette[[16, 0, 8, 8, 8, 8, 8, 8, 8, 8, 0, 9, 9, 9, 9, 9, 9, 9, 9]]

elif category == 'fish_epilepsy_5':
    # self.skeleton = [[1, 4], [4, 3], [3, 5], [5, 2], [1, 4], [1, 3], [1, 5], [1, 2]]
    #
    # self.limb_color = colors.pose_palette[[9, 7, 16, 8, 0, 0, 0, 0]]

    self.skeleton = [[1, 4], [4, 3], [3, 5], [5, 2]]

    self.limb_color = colors.pose_palette[[9, 7, 16, 8]]
    self.kpt_color = colors.pose_palette[[16, 9, 8, 7, 6]]

elif category == 'fish_xlx_10':
    # self.skeleton = [[1, 4], [4, 3], [3, 5], [5, 2], [1, 4], [1, 3], [1, 5], [1, 2]]
    #
    # self.limb_color = colors.pose_palette[[9, 7, 16, 8, 0, 0, 0, 0]]

    self.skeleton = [[0, 1], [0, 2], [1, 3], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]]

    self.limb_color = colors.pose_palette[[9, 9, 7, 16, 8, 12, 16, 7, 5, 3]]
    self.kpt_color = colors.pose_palette[[16, 9, 9, 8, 7, 6, 5, 4, 3, 2]]
```
``` shell
 def kpts(self, kpts, shape=(640, 640), radius=5, kpt_line=True):
    """
    Plot keypoints on the image.

    Args:
        kpts (tensor): Predicted keypoints with shape [17, 3]. Each keypoint has (x, y, confidence).
        shape (tuple): Image shape as a tuple (h, w), where h is the height and w is the width.
        radius (int, optional): Radius of the drawn keypoints. Default is 5.
        kpt_line (bool, optional): If True, the function will draw lines connecting keypoints
                                   for human pose. Default is True.

    Note: `kpt_line=True` currently only supports human pose plotting.
    """
    if self.pil:
        # Convert to numpy first
        self.im = np.asarray(self.im).copy()
    nkpt, ndim = kpts.shape
    # modify as follow
    # is_pose = nkpt == 17 and ndim == 3
    is_pose = nkpt == self.kpt_color.shape[0] and ndim == 3
    kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
```
3. Pose estimation using trained weights
``` shell
# Navigate to the predict folder

cd Petra/experiments/src/zebrafish/predict/video
python zebrafish_video_pose_pred.py
```
4. Pose tracking using Bytetrack
``` shell
# Navigate to the predict folder

cd Petra/experiments/src/zebrafish/predict/video
python zebrafish_video_pose_track.py
```
