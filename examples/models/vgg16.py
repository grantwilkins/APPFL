import torch
import torch.nn as nn


class VGG16MNIST(nn.Module):
    def __init__(self, num_channel, num_classes, num_pixel):
        super(VGG16MNIST, self).__init__()
        self.features = self._make_layers(num_channel)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self, num_channel):
        layers = []
        in_channels = num_channel
        vgg = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]
        for x in vgg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        return nn.Sequential(*layers)
