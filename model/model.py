from torch import nn
from efficientnet_pytorch import EfficientNet


class EfNetModel(nn.Module):
    def __init__(self, num_classes=33):
        super().__init__()

        self.model = EfficientNet.from_pretrained('efficientnet-b4')
        self.model.fc_out_features = num_classes

        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x
