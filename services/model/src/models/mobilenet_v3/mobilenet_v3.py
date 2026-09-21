import os.path
import time
from functools import partial

# import src.utils as utils
import torch
import yaml
from src.dataloaders.dataloader import custom_dataloader
from src.datasets.yoloDataset import YOLODataset
from torchvision.transforms import v2
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from tqdm import tqdm


def build_ppe_model():
    """
    Adds the PPE head to the pretrained MobileNet V3 model. We have to create a custom classification head for the
    MobileNet v3 model for our dataset which has 10 classes.
    :return: returns ssdlite320_mobilenet_v3_large with default weights and number of classes set to
    number of classes in PPE
    """
    with open("../../../dataset/data.yaml", "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    nc = data["nc"]

    model = ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,
    )

    in_channels = det_utils.retrieve_out_channels(
        model=model.backbone,  # The module whose out channels we need
        size=(320, 320),  # The input of mobilenet_v3 large is 320 x 320
    )  # return a list of out channels as there are multiple feature maps

    num_anchors = (
        model.anchor_generator.num_anchors_per_location()
    )  # number of anchors for each feature map
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    model.classification_head = SSDLiteClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=nc + 1,
        norm_layer=norm_layer,
    )
    return model


def evaluation_pipeline(model, X: torch.Tensor, annotations: dict[str, list]):
    """
    Function to calculate the precision, recall, F1 score, mAP, IoU
    :param annotations: python dictionary
    :param model: ssdlite320_mobilenet_v3_large
    :param X: list(torch.tensor)x
    :return:
    """
    pass


def training_pipeline(model: SSD, epochs: int, train_dataloader: DataLoader):
    """
    The training pipeline for the model.
    1. Set to training mode
    2. Zero gradient
    3. Forward pass
    4. Loss computation
    5. Backward propagation (for parameter x in the model, d(loss) / dx is computed and stored in x.grad)
    6. Optimizer step (updates the value of x using the gradient x.grad)
    :param model:
    :param epochs:
    :param train_dataloader:
    :return:
    """
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)

    log = {"loss": [], "time": []}
    for epoch in range(epochs):
        start_time = time.perf_counter()
        epoch_loss = 0
        prog_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}", leave=False
        )
        model.train()
        for batch_idx, batch in enumerate(prog_bar):
            images, labels = batch

            # zero gradients for each batch
            optimizer.zero_grad()

            # loss computation
            loss_dict = model(images, labels)
            loss = sum(loss for loss in loss_dict.values())
            epoch_loss += loss.item()

            # backward propagation
            loss.backward()

            # Optimizer step
            optimizer.step()

            prog_bar.set_postfix(loss=f"{loss.item():.4f}")
        end_time = time.perf_counter()
        avg_loss = epoch_loss / len(train_dataloader)
        time_taken = end_time - start_time
        log["loss"].append(avg_loss)
        log["time"].append(time_taken)
        print(f" Avg loss: {avg_loss} | Time: {time_taken * 1000:.3f}ms")

    torch.save(model.state_dict(), "./saved_model_weights/model.pth")


def export_to_onnx(model: SSD):
    """
    Function to convert the model to onnx
    :param model: ssdlite320_mobilenet_v3_large
    :return: None
    """
    model.eval()
    input_tensor = torch.rand((1, 3, 64, 64), dtype=torch.float32)

    torch.onnx.export(
        model=model,
        args=(input_tensor),
        f="ssdlite320_mobilenet_v3.onnx",
        input_names=["input"],
        dynamo=False,  # supporting model export using TorchDynamo
    )
    print("Model exported to onnx")


def parameter_calc(model):
    """
    Computes the total size of the model in MB
    :param model: the CNN model for computing the total number of parameters
    :return: (param_size: int, buffer_size: int, size_all_mb: int)
    """
    param_size = 0
    for param in model.parameters():
        param_size += (
            param.nelement() * param.element_size()
        )  # no.of elem * size of each element (4B here)
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return (param_size, buffer_size, size_all_mb)


def latency_calc(model, x=None):
    """
    Computes the latency for inference on each of the input image and returns mean and median latency
    :param model: the CNN model for computing the total number of parameters
    :param x: input tensor (can be None)
    :return: mean and median latency
    """
    if x is None:
        x = [torch.rand(3, 320, 320)]
    model.eval()
    latencies = []
    for _ in range(10):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        end = time.perf_counter()
        latencies.append(end - start)
    mean = sum(latencies) / len(latencies)
    median = sorted(latencies)[len(latencies) // 2]
    return mean, median


def write_metrics(model: SSD):
    """
    Writes the evaluation metrics into metrics.txt
    :param model: ssdlite320_mobilenet_v3_large
    :return: None
    """
    param_size, buffer_size, model_size = parameter_calc(model)
    mean_lat, med_lat = latency_calc(model)
    with open("metrics.txt", "w") as f:
        f.write(f"Parameters size: {param_size}\n")
        f.write(f"Buffers size: {buffer_size}\n")
        f.write(f"Model size: {model_size:.3f}MB\n")
        f.write(f"Inference mean latency: {mean_lat * 1000:.3f}ms\n")
        f.write(f"Inference median latency: {med_lat * 1000:.3f}ms\n")


if __name__ == "__main__":
    model = build_ppe_model()
    if os.path.exists("./saved_model_weights/model.pth"):
        model.load_state_dict(torch.load("./saved_model_weights/model.pth"))
    transform_pipeline = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True)
        ]
    )
    train_dataset = YOLODataset(root="../../../dataset/train", transforms=transform_pipeline)
    train_dataloader = custom_dataloader(train_dataset, 4, 0)
    training_pipeline(model, 1, train_dataloader)
    # model.eval()
    #
    # with torch.no_grad():
    #     predictions = model(
    #         x
    #     )  # [{"boxes": torch.tensor, "scores": torch.tensor, "labels": torch.tensor}]
    #
    # # pred = predictions[0]
    # # bbox.draw_bbox(x[0], pred)
    #
    # write_metrics(model)
    #
    # print("Evaluation written into metrics.txt")
    #
    # # export_to_onnx(model)
