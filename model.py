import sys
sys.path.append("E:/projekty python/PyTorch course/going_modular")

from torch import nn # noqa 5501
import utils # noqa 5501
import torchvision # noqa 5501


def create_effnetb2(out_features,
                    device):
    effnetb2_weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transforms = effnetb2_weights.transforms()
    model = torchvision.models.efficientnet_b2(weights=effnetb2_weights).to(device) # noqa 5501

    for param in model.features.parameters():
        param.requires_grad = False

    utils.set_seeds(42)

    # # Set cllasifier to suit problem
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1408,
                  out_features=out_features,
                  bias=True).to(device))

    model.name = "effnetb2"
    return model, transforms
