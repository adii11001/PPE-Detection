import torch


def compute_iou(rect1: list | torch.Tensor, rect2: list | torch.Tensor):
    """
    Intersection over Union (IoU):
    IoU = Area of intersection / Area of union
    Ranges from 0 to 1 and IoU >= 0.5 is considered good.
    """
    width1 = rect1[2] - rect1[0]
    width2 = rect2[2] - rect2[0]

    height1 = rect1[3] - rect1[1]
    height2 = rect2[3] - rect2[1]

    area_rect1 = width1 * height1
    area_rect2 = width2 * height2

    top_left_x = max(rect1[0], rect2[0])
    top_left_y = max(rect1[1], rect2[1])
    bottom_right_x = min(rect1[2], rect2[2])
    bottom_right_y = min(rect1[3], rect2[3])

    width_is = bottom_right_x - top_left_x
    height_is = bottom_right_y - top_left_y

    intersect_area = max(0, width_is) * max(0, height_is)
    union_area = area_rect1 + area_rect2 - intersect_area

    return intersect_area / union_area
