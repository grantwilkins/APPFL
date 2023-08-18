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
from torch.sparse import *
import zstd
import pickle
from scipy import sparse
import time


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


def magnitude_prune(model: nn.Module, prune_ratio: float):
    """
    Perform magnitude-based pruning on a PyTorch model.

    Args:
    model: the PyTorch model to prune.
    prune_ratio: the percentage of weights to prune in each layer.

    Returns:
    The pruned model.
    """
    # Make a copy of the model
    model_copy = copy.deepcopy(model)
    nonzero_total = 0
    param_total = 0
    for _, param in model_copy.named_parameters():
        if param.requires_grad:
            # Flatten the tensor to 1D for easier percentile calculation
            param_flattened = param.detach().cpu().numpy().flatten()
            # Compute the threshold as the (prune_ratio * 100) percentile of the absolute values
            threshold = np.percentile(np.abs(param_flattened), prune_ratio * 100)
            # Create a mask that will be True for the weights to keep and False for the weights to prune
            mask = torch.abs(param) > threshold
            # Apply the mask
            param.data.mul_(mask.float())
            nonzero_total += torch.count_nonzero(param.data).item()
            param_total += param.data.numel()
    print(nonzero_total, param_total, nonzero_total / param_total)
    return model_copy


def flatten_too(model: torch.nn.Module) -> np.ndarray:
    # Concatenate all of the tensors in the model's state_dict into a 1D tensor
    flat_params = [
        param.view(-1).detach().cpu().numpy() for _, param in model.named_parameters()
    ]
    max_length = max(param.size for param in flat_params)
    padded_params = [
        np.pad(param, (0, max_length - param.size)) for param in flat_params
    ]
    double_array = np.stack(padded_params)

    # Convert the tensor to a numpy array and return it
    return double_array


def unflatten_too(model: torch.nn.Module, flat_params):
    i = 0
    model_copy = copy.deepcopy(model)
    for _, param in model_copy.named_parameters():
        param = flat_params[i, : np.prod(param.shape)].reshape(param.shape)
        i = i + 1
    return model_copy.state_dict()


def flatten_too_too(model: torch.nn.Module):
    flat_dict = {}
    for name, param in model.named_parameters():
        flat_dict[name] = param.view(-1).detach().cpu().numpy()
    return flat_dict


def test_model_compress(cfg: Config, model: nn.Module) -> None:
    # Create a compressor
    compressor = Compressor(cfg)
    # Define the AlexNetMNIST model
    # model = resnet18(num_channel=1, num_classes=10, pretrained=1)
    # model = AlexNetMNIST(num_channel=3, num_classes=10, num_pixel=32)
    # model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    model_copy = copy.deepcopy(model)
    # pruned_model = magnitude_prune(model, 0.5)
    # Flatten the model parameters)
    compressed_weights = {}
    weights_size = 0
    size = 0
    time1 = time.time()
    (comp_model, _, lossless_comp_ratio) = compressor.compress_model(model, 1e10)
    decomp_model = compressor.decompress_model(
        compressed_model=comp_model, model=model, param_count_threshold=1e10
    )
    time2 = time.time() - time1
    print(time2, lossless_comp_ratio)
    print("Score = " + str(lossless_comp_ratio / time2))

    # Reasseble the model
    # Check if the reassembled model is the same shape as the original model
    for p, p_copy in zip(model.parameters(), model_copy.parameters()):
        assert p.shape == p_copy.shape


if __name__ == "__main__":
    # Config setup
    cfg = OmegaConf.structured(Config)
    cfg.compressed_weights_client = True
    cfg.compressor = "SZ2"
    cfg.lossless_compressor = "blosc"
    cfg.compressor_lib_path = "/Users/grantwilkins/SZ/build/sz/libSZ.dylib"
    # cfg.compressor_lib_path = "/Users/grantwilkins/SZ/build/sz/libSZ.dylib"
    cfg.compressor_error_bound = 0.1
    cfg.compressor_error_mode = "REL"
    compressors = ["blosc", "zstd", "zlib", "gzip", "xz"]
    models_test = [
        models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1),
    ]
    for model in models_test:
        for compressor in compressors:
            cfg.lossless_compressor = compressor
            test_model_compress(cfg=cfg, model=model)
