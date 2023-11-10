import math
import cv2
import numpy as np
from matplotlib import pyplot as plt

import _base
from utils import *


def show_result(img,
                result,
                skeleton=None,
                kpt_score_thr=0.3,
                bbox_color='green',
                pose_kpt_color=None,
                pose_link_color=None,
                text_color='white',
                radius=4,
                thickness=1,
                font_scale=0.5,
                bbox_thickness=1,
                win_name='',
                show=False,
                show_keypoint_weight=False,
                wait_time=0,
                out_file=None):
    """Draw `result` over `img`.

    Args:
        img (str or Tensor): The image to be displayed.
        result (list[dict]): The results to draw over `img`
            (bbox_result, pose_result).
        skeleton (list[list]): The connection of keypoints.
            skeleton is 0-based indexing.
        kpt_score_thr (float, optional): Minimum score of keypoints
            to be shown. Default: 0.3.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
            If None, do not draw keypoints.
        pose_link_color (np.array[Mx3]): Color of M links.
            If None, do not draw links.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        radius (int): Radius of circles.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        win_name (str): The window name.
        show (bool): Whether to show the image. Default: False.
        show_keypoint_weight (bool): Whether to change the transparency
            using the predicted confidence scores of keypoints.
        wait_time (int): Value of waitKey param.
            Default: 0.
        out_file (str or None): The filename to write the image.
            Default: None.

    Returns:
        Tensor: Visualized img, only if not `show` or `out_file`.
    """
    # img = mmcv.imread(img)
    img = cv2.imread(img)
    img = img.copy()

    bbox_result = []
    bbox_labels = []
    pose_result = []
    for res in result:
        if 'bbox' in res:
            bbox_result.append(res['bbox'])
            bbox_labels.append(res.get('label', None))
        pose_result.append(res['keypoints'])

    if bbox_result:
        bboxes = np.vstack(bbox_result)
        # draw bounding boxes
        imshow_bboxes(
            img,
            bboxes,
            labels=bbox_labels,
            colors=bbox_color,
            text_color=text_color,
            thickness=bbox_thickness,
            font_scale=font_scale,
            show=False)

    if pose_result:
        imshow_keypoints(img, pose_result, skeleton, kpt_score_thr,
                         pose_kpt_color, pose_link_color, radius,
                         thickness)

    if show:
        _base.imshow(img, win_name, wait_time)

    if out_file is not None:
        cv2.imwrite(img, out_file)

    return img