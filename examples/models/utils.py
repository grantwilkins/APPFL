from .cnn import CNN, ComplexCNN
from .resnet import resnet18, resnet101
from .alexnet import AlexNetMNIST, AlexNetCIFAR, AlexNetCaltech
from .vgg16 import VGG16MNIST, VGG16CIFAR
from .lenet5 import LeNet5


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
    if args.model == "resnet101":
        model = resnet101(args.num_classes, grayscale=False)
    if args.model == "resnet18":
        model = resnet18(
            num_channel=args.num_channel,
            num_classes=args.num_classes,
            pretrained=args.pretrained,
        )
    return model
