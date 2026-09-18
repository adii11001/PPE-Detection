import time

import src.utils as utils
import torch
import yaml
from src.datasets.yoloDataset import YOLODataset
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.ssd import SSD


def build_ppe_model():
    """
    Adds the PPE head to the pretrained MobileNet V3 model.
    :return: returns ssdlite320_mobilenet_v3_large with default weights and number of classes set to
    number of classes in PPE
    """
    with open('../../dataset/data.yaml', 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    nc = data["nc"]

    model = ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,
        num_classes=nc + 1
    )

    return model


def evaluation_pipeline(model, X: torch.Tensor, annotations: dict[str, list]):
    """
    Function to calculate the precision, recall, F1 score, mAP, IoU
    :param annotations: python dictionary
    :param model: ssdlite320_mobilenet_v3_large
    :param X: list(torch.tensor)
    :return:
    """
    model.eval()
    with torch.no_grad():
        predictions = model(X)


def training_pipeline(model: SSD, train_dataset: YOLODataset):
    pass


def export_to_onnx(model: SSD):
    model.eval()
    input_tensor = torch.rand((1, 3, 64, 64), dtype=torch.float32)

    torch.onnx.export(
        model=model,
        args=(input_tensor),
        f="ssdlite320_mobilenet_v3.onnx",
        input_names=["input"],
        dynamo=False  # supporting model export using TorchDynamo
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
        param_size += param.nelement() * param.element_size()  # no.of elem * size of each element (4B here)
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
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
    model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    # x = [torch.rand(3, 320, 320)]
    # model.eval()
    #
    # with torch.no_grad():
    #     predictions = model(x)  # [{"boxes": torch.tensor, "scores": torch.tensor, "labels": torch.tensor}]
    #
    # pred = predictions[0]
    # bbox.draw_bbox(x[0], pred)
    #
    # write_metrics(model)
    #
    # print("Evaluation written into metrics.txt")

    export_to_onnx(model)
