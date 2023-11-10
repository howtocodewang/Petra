import os.path
import random
import shutil

import numpy as np

from data_utils import *


image_ext = 'jpg', 'png'
video_ext = 'mp4'


def create_yolo_type_folder(dataset_root, folder_name="yolo_type_dataset", overwrite=True):
    yolo_type_dataset_path = os.path.join(dataset_root, folder_name)
    createdir(path=yolo_type_dataset_path, overwrite=overwrite)

    sub_folders = ["images", "labels", "coco", "labelme"]
    dataset_splits = ["train", "val", "test"]

    for sf in sub_folders:
        sf_path = os.path.join(yolo_type_dataset_path, sf)
        createdir(path=sf_path, overwrite=overwrite)
        for split in  dataset_splits:
            split_path = os.path.join(sf_path, split)
            createdir(path=split_path, overwrite=overwrite)

    print("Successfully create YOLO tyep dataset folder at the location: {}".format(yolo_type_dataset_path))
    return yolo_type_dataset_path


def dataset_split(full_data_list, train_perc=70, val_perc=20, test_perc=10, full_data=True, last_sample2="train"):
    train_ratio = train_perc / 100
    val_ratio = val_perc / 100
    test_ratio = test_perc / 100
    tgt_ratio = (train_perc + val_perc + test_perc) / 100

    if tgt_ratio > 1.0:
        raise Exception("Train, val, and test should not have data in common!")
    elif tgt_ratio < 1.0 and not full_data:
        print("Not all data is used")
    elif tgt_ratio == 1.0 and full_data:
        print("Not all data is used")
    else:
        raise Exception("Wrong dataset split settings, please check!")

    if test_ratio == 0.0:
        print("No test data will be used after training stage!")

    print("Start splitting the whole dataset...")

    num_sample = len(full_data_list)
    random.shuffle(full_data_list)
    split_dataset_list = []

    num_train_sample = int(num_sample * train_ratio )
    train_dataset = full_data_list[: num_train_sample]
    split_dataset_list.append(train_dataset)

    num_val_sample = int(num_sample * val_ratio)
    val_dataset = full_data_list[num_train_sample: num_train_sample + num_val_sample]
    split_dataset_list.append(val_dataset)

    if test_ratio != 0.0:
        num_test_sample = int(num_sample * test_ratio)
        test_dataset = full_data_list[num_train_sample + num_val_sample:]
        split_dataset_list.append(test_dataset)
    else:
        test_dataset = []

    concat_list = sum(split_dataset_list, [])
    inter_sample = set(split_dataset_list[0]).intersection(*split_dataset_list[1:])

    if len(inter_sample) == 0:
        if concat_list == full_data_list:
            print("Successfully split the whole dataset!")
        elif concat_list != full_data_list and full_data:
            diff_list = list(set(full_data_list).difference(set(concat_list)))
            if last_sample2 == "train":
                train_dataset = sum([train_dataset, diff_list], [])
            elif last_sample2 == "val":
                val_dataset = sum([val_dataset, diff_list], [])
            elif test_ratio != 0.0 and last_sample2 == "test":
                test_dataset = sum([test_dataset, diff_list], [])
            print("Successfully split the whole dataset (not assigned samples are belonged to {} dataset)!".format(
                last_sample2))
        elif not full_data:
            print("Successfully split the whole dataset (Not all data is used)!")
    else:
        raise Warning("Each sub-dataset have samples in common!")

    split_dataset_info = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    }

    return split_dataset_info


def assign2folder(tgt_dataset_path, split_dataset_dict, label_ext='json', method='softlink'):
    final_dataset_info = {}
    for set_name, subset in split_dataset_dict.items():
        print("Processing data in {} dataset".format(set_name))
        for img_path in tqdm(subset):
            img_name = img_path.split('/')[-1]
            src_path_img = img_path
            dst_path_img = os.path.join(tgt_dataset_path, "images", set_name, img_name)

            label_path = img_path.split('.')[0] + '.' + label_ext
            label_name = label_path.split('/')[-1]
            src_path_label = label_path
            dst_path_label = os.path.join(tgt_dataset_path, "labelme", set_name, label_name)

            if os.path.exists(src_path_label):
                if method == 'softlink':
                    os.symlink(src_path_img, dst_path_img)
                    os.symlink(src_path_label, dst_path_label)
                elif method == 'copy':
                    shutil.copy(src_path_img, dst_path_img)
                    shutil.copy(src_path_label, dst_path_label)
            else:
                print("{} is not labelled".format(img_path))
                continue

    final_dataset_info["train"] = getFilelist(folder_path=os.path.join(tgt_dataset_path, "images", "train"))
    final_dataset_info["val"] = getFilelist(folder_path=os.path.join(tgt_dataset_path, "images", "val"))
    final_dataset_info["test"] = getFilelist(folder_path=os.path.join(tgt_dataset_path, "images", "test"))

    print("Successfully build a yolo-like dataset")

    return final_dataset_info


if __name__ == "__main__":
    dataset_root = "/home/wangshuo/Datasets/SIMIT/AnimalPoseBehavior/zebrafish_xlx"
    raw_data_path = os.path.join(dataset_root, "raw_data")
    is_build_yolo_dataset = True
    yolo_dataset_path = os.path.join(dataset_root, "yolo_type_dataset")

    if is_build_yolo_dataset or not os.path.exists(yolo_dataset_path):
        yolo_dataset_path = create_yolo_type_folder(
            dataset_root=dataset_root,
            folder_name="yolo_type_dataset",
            overwrite=True
        )
    else:
        yolo_dataset_path = os.path.join(dataset_root, "yolo_type_dataset")

    full_image_list = getFilelist(folder_path=raw_data_path, ext=image_ext)
    full_image_list.sort()

    split_dataset = dataset_split(
        full_data_list=full_image_list,
        train_perc=80,
        val_perc=10,
        test_perc=10,
        full_data=True,
        last_sample2='train'
    )

    print("The number of full dataset: ", len(full_image_list))

    train_list = split_dataset['train']
    print("The number of train: ", len(train_list))

    val_list = split_dataset['val']
    print("The number of val: ", len(val_list))

    test_list = split_dataset['test']
    print("The number of test: ", len(test_list))

    final_dataset = assign2folder(
        tgt_dataset_path=yolo_dataset_path,
        split_dataset_dict=split_dataset,
        label_ext='json',
        method='copy'
    )

    # print('debug')
    print("Successfully split raw dataset and saved in {}".format(yolo_dataset_path))