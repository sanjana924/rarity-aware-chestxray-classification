
import os
import re
import torch
import torch.nn as nn
from torchvision.models import densenet121

WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "pretrained",
    "densenet121-a639ec97.pth"
)


def create_model(pretrained=True):
    model = densenet121(weights=None)

    if pretrained:
        state_dict = torch.load(
            WEIGHTS_PATH,
            map_location="cpu",
            weights_only=True
        )

        # Support older DenseNet checkpoint naming.
        pattern = re.compile(
            r"^(.*denselayer\d+\.(?:norm|relu|conv))"
            r"\.([12]\.(?:weight|bias|running_mean|running_var))$"
        )

        for key in list(state_dict.keys()):
            match = pattern.match(key)
            if match:
                new_key = match.group(1) + match.group(2)
                state_dict[new_key] = state_dict.pop(key)

        model.load_state_dict(state_dict, strict=True)
        print("Pretrained ImageNet weights loaded!")

    model.classifier = nn.Linear(
        model.classifier.in_features,
        14
    )

    return model


if __name__ == "__main__":
    model = create_model(pretrained=True)

    print("DenseNet-121 ready!")
    print("Output classes:", model.classifier.out_features)


