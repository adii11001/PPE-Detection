import time

import torch
import yaml
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.ssd import SSD

from src.datasets.yoloDataset import YOLODataset

def build_ppe_model(model: SSD):
    """
    Adds the PPE head to the pretrained MobileNet V3 model
    :param model: ssdlite320 mobilenet v3 large model (for this project)
    :return: returns ssdlite320_mobilenet_v3_large with default weights and number of classes set to
    number of classes in PPE
    """
    with open('services/model/dataset/data.yaml', 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    nc = data["nc"]

    return ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,
        num_classes=nc
    )

def training_pipeline(model: SSD, train_dataset: YOLODataset):
    pass

def parameter_calc(model):
    """
    Computes the total size of the model in MB
    :param model: the CNN model for computing the total number of parameters
    :return: (param_size: int, buffer_size: int, size_all_mb: int)
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size() # no.of elem * size of each element (4B here)
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



if __name__ == "__main__":
    model = ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    )
    x = [torch.rand(3, 320, 320)]
    model.eval()

    with torch.no_grad():
        predictions = model(x)

    pred = predictions[0]
    # bbox.draw_bbox(x[0], pred)

    param_size, buffer_size, model_size = parameter_calc(model)
    lat_mean, lat_median = latency_calc(model)

    with open("metrics.txt", "w") as f:
        f.write(f"Parameters size: {param_size}\n")
        f.write(f"Buffers size: {buffer_size}\n")
        f.write(f"Model size: {model_size:.3f}MB\n")
        f.write(f"Inference mean latency: {lat_mean * 1000:.6f}ms\n")
        f.write(f"Inference median latency: {lat_median * 1000:.6f}ms\n")
    print("Evaluation written into metrics.txt")
