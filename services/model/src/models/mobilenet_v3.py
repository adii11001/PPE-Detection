from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from dataclasses import dataclass

@dataclass
class ModelProfile:
    name: str
    model: str

PROFILES = {
    "edge": ModelProfile(
        name='edge',
        model_id=""
    )
}
