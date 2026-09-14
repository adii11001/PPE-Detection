from pathlib import Path

import torchvision.transforms as t
from PIL import Image
from torchvision.datasets import VisionDataset


class YOLODataset(VisionDataset):
    def __init__(self, root: str, transforms: t.transforms.Compose):
        """
        set up your dataset (paths, annotations, transforms, and other configurations).
        :param root: path dataset/train or dataset/test or dataset/validation
        :param transforms: transformation pipeline
        """
        super().__init__(root=root, transform=transforms) # Initialize VisionDataset
        self.root = Path(root)
        self.img_dir = self.root / "images"
        self.label_dir = self.root / "labels"

        self.images = list(self.img_dir.glob(pattern="*.jpg")) # stores all of them image paths

    def __len__(self):
        """
        return the total number of samples
        :return: return the total number of samples.
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        fetch a single sample (and its label) by index
        :param index: image index
        :return: image, annotations
        """
        image_path = self.images[index] # Get image path

        image = Image.open(image_path).convert("RGB") # Open image from path and convert to rgb
        img_width, img_height = image.size

        label_path = self.label_dir / f"{image_path.stem}.txt" # Get the label path using the image

        annotations = []
        with open(label_path, mode="r") as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.split())

                # convert normalized (0-1) coords to pixel coords
                x_center *= img_width
                width *= img_width
                y_center *= img_height
                height *= img_height

                # convert center coords to corner coords
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2

                annotations.append({
                    "class_id":int(class_id),
                    "bbox":[x_min, y_min, x_max, y_max]
                })

        if self.transform:
            image = self.transform(image)
        return image, annotations

if __name__ == "__main__":
    transform_pipeline = t.Compose(
        [
            t.ToTensor(),
            t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = YOLODataset(
        root="C:/Users/admin/Desktop/git/PPE-Detection/services/model/dataset/train",
        transforms=transform_pipeline,
    )
    print("Shape of the image tensor: ", train_dataset[0][0].shape)
    print("Annotations: ", train_dataset[0][1])
