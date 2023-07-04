import torch

from fedot_ind.core.operation.optimization.sfp_tools import _check_nonzero_filters, _prune_filters


class SimpleConvNet3(torch.nn.Module):
    """Convolutional neural network with two convolutional layers
    Args:
        num_classes: number of classes.
        in_channels: The number of input channels of the first layer.
    """

    def __init__(
            self,
            num_classes: int = 10,
            in_channels: int = 3,
            hidden1: int = 32,
            hidden2: int = 64,
            hidden3: int = 128,
    ):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, hidden1, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden1, hidden2, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden2, hidden3, kernel_size=7),
            torch.nn.ReLU(),
        )
        self.drop_out = torch.nn.Dropout()
        self.fc = torch.nn.Linear(hidden3, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.flatten(out, 1)
        out = self.drop_out(out)
        out = self.fc(out)
        return out


def prune_simple_conv_net(model: SimpleConvNet3) -> SimpleConvNet3:
    assert isinstance(model, SimpleConvNet3), "Supports only SimpleConvNet models"
    sd = model.state_dict()
    hidden1 = _check_nonzero_filters(sd['layer1.0.weight'])
    sd['layer1.0.weight'] = _prune_filters(sd['layer1.0.weight'], saving_filters=hidden1)
    sd['layer1.0.bias'] = sd['layer1.0.bias'][hidden1].clone()
    hidden2 = _check_nonzero_filters(sd['layer2.0.weight'])
    sd['layer2.0.weight'] = _prune_filters(
        sd['layer2.0.weight'],
        saving_filters=hidden2,
        saving_channels=hidden1
    )
    sd['layer2.0.bias'] = sd['layer2.0.bias'][hidden2].clone()
    hidden3 = _check_nonzero_filters(sd['layer3.0.weight'])
    sd['layer3.0.weight'] = _prune_filters(
        sd['layer3.0.weight'],
        saving_filters=hidden3,
        saving_channels=hidden2
    )
    sd['layer3.0.bias'] = sd['layer3.0.bias'][hidden3].clone()
    sd['fc.weight'] = sd['fc.weight'][:, hidden3].clone()
    model = SimpleConvNet3(
        num_classes=model.fc.out_features,
        in_channels=model.layer1[0].in_channels,
        hidden1=hidden1.numel(),
        hidden2=hidden2.numel(),
        hidden3=hidden3.numel(),
    )
    model.load_state_dict(sd)
    return model


def load_simple_conv_net(state_dict_path: str) -> SimpleConvNet3:
    sd = torch.load(state_dict_path)
    model = SimpleConvNet3(
        num_classes=sd['fc.weight'].shape[0],
        in_channels=sd['layer1.0.weight'].shape[1],
        hidden1=sd['layer1.0.weight'].shape[0],
        hidden2=sd['layer2.0.weight'].shape[0],
        hidden3=sd['layer3.0.weight'].shape[0],
    )
    model.load_state_dict(sd)
    return model
