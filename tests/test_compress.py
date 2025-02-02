from appfl.compressor.compressor import *
import numpy as np
from appfl.config import *
import torch
import torch.nn as nn
import math
import appfl.misc.utils as my_utils
import copy
from torchvision.models.resnet import BasicBlock, ResNet18_Weights
from torchvision import *


class CNN(nn.Module):
    def __init__(self, num_channel, num_classes, num_pixel):
        super().__init__()
        self.conv1 = nn.Conv2d(
            num_channel, 32, kernel_size=5, padding=0, stride=1, bias=True
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.act = nn.ReLU(inplace=True)

        ###
        ### X_out = floor{ 1 + (X_in + 2*padding - dilation*(kernel_size-1) - 1)/stride }
        ###
        X = num_pixel
        X = math.floor(1 + (X + 2 * 0 - 1 * (5 - 1) - 1) / 1)
        X = X / 2
        X = math.floor(1 + (X + 2 * 0 - 1 * (5 - 1) - 1) / 1)
        X = X / 2
        X = int(X)

        self.fc1 = nn.Linear(64 * X * X, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.maxpool(x)
        x = self.act(self.conv2(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


class ResNet(models.resnet.ResNet):
    # ResNet 18 Architecture Implementation to adapt grayscale and 28 X 28 pixel size input
    def __init__(self, block, layers, num_classes, grayscale):
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2.0 / n) ** 0.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        logits = self.fc(x)

        return logits


class AlexNetMNIST(nn.Module):
    def __init__(self, num_channel, num_classes, num_pixel):
        super(AlexNetMNIST, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.num_channel = num_channel

        # Calculate the size of the flattened features after convolution and pooling
        self.flat_feature_size = self._get_flat_feature_size(num_pixel)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.flat_feature_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _get_flat_feature_size(self, num_pixel):
        with torch.no_grad():
            dummy = torch.ones(1, self.num_channel, num_pixel, num_pixel)
            dummy = self.features(dummy)
            dummy = self.avgpool(dummy)
            return dummy.view(1, -1).size(1)


def resnet18(num_channel, num_classes=-1, pretrained=0):
    model = None

    if num_channel == 1:
        if num_classes < 0 or pretrained > 0:
            weights = ResNet18_Weights.verify(ResNet18_Weights.IMAGENET1K_V1)
            num_classes = len(weights.meta["categories"])
            model = ResNet(
                block=BasicBlock,
                layers=[2, 2, 2, 2],
                num_classes=num_classes,
                grayscale=False,
            )
            model.load_state_dict(weights.get_state_dict(progress=True))
        else:
            model = ResNet(
                block=BasicBlock,
                layers=[2, 2, 2, 2],
                num_classes=num_classes,
                grayscale=True,
            )

    else:
        if num_classes < 0 or pretrained > 0:
            model = models.resnet18(pretrained=True)
        else:
            model = models.resnet18(pretrained=False, num_classes=num_classes)

    return model


def test_basic_compress(cfg: Config) -> None:
    # Create a compressor
    compressor = Compressor(cfg)

    # Create a random 1D array
    ori_data = np.random.rand(1000)
    ori_shape = ori_data.shape
    ori_dtype = ori_data.dtype

    # Compress the array
    cmpr_data_bytes = compressor.compress(ori_data=ori_data)
    cmpr_data = np.frombuffer(cmpr_data_bytes, dtype=np.uint8)
    # Decompress the array
    dec_data = compressor.decompress(
        cmp_data=cmpr_data, ori_shape=ori_shape, ori_dtype=ori_dtype
    )
    # Check if the decompressed array is the same as the original array
    (max_diff, _, _) = compressor.verify(ori_data=ori_data, dec_data=dec_data)
    assert max_diff < cfg.compressor_error_bound


def my_flatten_model_params(model: nn.Module) -> np.ndarray:
    params = {}
    for name, param in model.state_dict().items():
        params[name] = param.data.numpy().flatten()
    return params


def test_model_compress(cfg: Config) -> None:
    # Create a compressor
    compressor = Compressor(cfg)
    # Define the AlexNetMNIST model
    model = AlexNetMNIST(num_channel=1, num_classes=10, num_pixel=28)

    flat_dict = my_flatten_model_params(model)
    total_size = 0
    comp_size = 0
    for k, v in flat_dict.items():
        cmpr_arr = compressor.compress(ori_data=v)
        total_size += len(v)
        comp_size += len(cmpr_arr)
        print(k, v.shape, len(cmpr_arr), len(v) * 4 / len(cmpr_arr))
    print(total_size, comp_size, total_size * 4 / comp_size)

    model_copy = copy.deepcopy(model)

    params = my_utils.flatten_model_params(model)
    ori_shape = params.shape
    ori_dtype = params.dtype

    # Compress the model
    cmpr_params_bytes = compressor.compress(ori_data=params)

    print(4 * len(params) / len(cmpr_params_bytes))
    # Decompress the model
    dec_params = compressor.decompress(
        cmp_data=cmpr_params_bytes, ori_shape=ori_shape, ori_dtype=ori_dtype
    )
    # Check if the decompressed model is the same as the original model
    (_, _, _) = compressor.verify(ori_data=params, dec_data=dec_params)

    # Reasseble the model
    my_utils.unflatten_model_params(model, dec_params)
    # Check if the reassembled model is the same shape as the original model
    for p, p_copy in zip(model.parameters(), model_copy.parameters()):
        assert p.shape == p_copy.shape


if __name__ == "__main__":
    # Config setup
    cfg = OmegaConf.structured(Config)
    cfg.compressed_weights_client = True
    cfg.compressor = "SZ3"
    cfg.compressor_lib_path = "/Users/grantwilkins/SZ3/build/tools/sz3c/libSZ3c.dylib"
    # cfg.compressor_lib_path = "/Users/grantwilkins/SZ/build/sz/libSZ.dylib"
    cfg.compressor_error_bound = 0.1
    cfg.compressor_error_mode = "REL"

    # Tests to run
    test_basic_compress(cfg=cfg)
    test_model_compress(cfg=cfg)
