from pathlib import Path

import torch
import torchvision.datasets
from src.datasets.yoloDataset import YOLODataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm


def custom_collate_fn(batch):
    batched_image_tensor = []
    batched_label_tensor = []
    for image, label in batch:
        batched_image_tensor.append(image)
        batched_label_tensor.append(label)
    batched_image_tensor = torch.stack(batched_image_tensor)

    return (batched_image_tensor, batched_label_tensor)

def custom_dataloader(dataset: torchvision.datasets.VisionDataset,
                      batch_size: int,
                      num_workers: int=0):
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      shuffle=True,
                      collate_fn=custom_collate_fn)


if __name__ == "__main__":
    transforms_pipeline = v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset_path = Path(
        "C:/Users/admin/Desktop/git/PPE-Detection/services/model/dataset"
    )
    train_dataset = YOLODataset(
        root=f"{dataset_path}/train", transforms=transforms_pipeline
    )
    test_dataset = YOLODataset(
        root=f"{dataset_path}/test", transforms=transforms_pipeline
    )

    print("Len of training dataset:", len(train_dataset))

    train_dataloader = custom_dataloader(
        dataset=train_dataset, batch_size=4, num_workers=0
    )

    test_dataloader = custom_dataloader(
        dataset=test_dataset, batch_size=4, num_workers=0
    )

    pbar = tqdm(train_dataloader, desc="Training", unit="batch")

    for batch_idx, batch in enumerate(pbar):
        if batch_idx % 10 == 0:
            pbar.set_postfix(
                {
                    "Batch": batch_idx,
                    "Shape": list(batch[0].shape),  # clean look: [4, 3, 224, 224]
                }
            )
