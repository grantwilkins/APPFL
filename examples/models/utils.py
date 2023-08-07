from .cnn import CNN, ComplexCNN
from .resnet import ResNet18, ResNet101, ResNet152, ResNet50
from .alexnet import AlexNetMNIST, AlexNetCIFAR, AlexNetCaltech
from .vgg16 import VGG16MNIST, VGG16CIFAR
from .lenet5 import LeNet5
from .mobilenet import MobileNetV2


def get_model(args):
    ## User-defined model
    if args.model == "CNN":
        model = CNN(args.num_channel, args.num_classes, args.num_pixel)
    if args.model == "ComplexCNN":
        model = ComplexCNN(args.num_channel, args.num_classes, args.num_pixel)
    if args.model == "AlexNetMNIST":
        model = AlexNetMNIST(args.num_channel, args.num_classes, args.num_pixel)
    if args.model == "AlexNetCIFAR":
        model = AlexNetCIFAR(args.num_classes)
    if args.model == "AlexNetCaltech":
        model = AlexNetCaltech(args.num_classes)
    if args.model == "VGG16MNIST":
        model = VGG16MNIST(args.num_channel, args.num_classes, args.num_pixel)
    if args.model == "VGG16CIFAR":
        model = VGG16CIFAR(args.num_classes)
    if args.model == "LeNet5":
        model = LeNet5(args.num_classes)
    if args.model == "ResNet101":
        model = ResNet101(num_classes=args.num_classes, channels=args.num_channel)
    if args.model == "ResNet50":
        model = ResNet50(num_classes=args.num_classes, channels=args.num_channel)
    if args.model == "ResNet18":
        model = ResNet18(
            num_channel=args.num_channel,
            num_classes=args.num_classes,
            pretrained=args.pretrained,
        )
    if args.model == "ResNet152":
        model = ResNet152(num_classes=args.num_classes, channels=args.num_channel)
    if args.model == "MobileNetV2":
        model = MobileNetV2(n_classes=args.num_classes, ch_in=args.num_channel)
    return model
