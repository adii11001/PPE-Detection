import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as t
from matplotlib import patches
from src.datasets.yoloDataset import YOLODataset


def draw_bbox(image: torch.Tensor, annotations: list):
    colors = ['red', 'orange', 'purple', 'pink',
              'blue', 'green', 'olive',
              'brown', 'gray', 'gold']
    random.shuffle(colors)

    # De-normalization so imd doesn't look like folk valley: pixel =(og - mean) / std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean

    # Display image
    img = np.array(image.permute(1, 2, 0))

    _, ax = plt.subplots(1)
    ax.imshow(img)

    for annotation in annotations:
        x_min, y_min, x_max, y_max = annotation["bbox"]

        width = x_max - x_min
        height = y_max - y_min

        color = colors[annotation["class_id"] % len(colors)]
        rect = patches.Rectangle((x_min, y_min), width, height,
                                 linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    plt.show()

if __name__ == "__main__":
    transform_pipeline = t.Compose(
        [
            t.ToTensor(),
            t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image_dataset = YOLODataset(root="C:/Users/admin/Desktop/git/PPE-Detection/services/model/dataset/train",
                                transforms=transform_pipeline)

    image, annotations = image_dataset[0]

    draw_bbox(image, annotations)
