from datetime import datetime
import torch
import numpy as np
from torch.utils.data import Subset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet18, resnet50
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from fedot_ind.core.architecture.experiment.nn_experimenter import ClassificationExperimenter, FitParameters
from fedot_ind.core.operation.optimization.svd_tools import decompose_module
from fedot_ind.core.operation.decomposition.decomposed_conv import DecomposedConv2d

from optim.supplemented_conv import SupplementedConv2d

metrics = torch.tensor(
    [
        0.8803827751196172,
        0.7636363636363637,
        0.7214611872146119,
        0.8942307692307692,
        0.5833333333333334,
        0.9463414634146342,
        0.6097560975609756,
        0.8651162790697675,
        0.7821782178217823,
        0.7562189054726369,
        0.9494949494949495,
        0.6784140969162996,
        0.6632653061224489,
        0.8461538461538461,
        0.7358490566037736,
        0.9565217391304348,
        0.7914438502673796,
        0.8686868686868686,
        0.6956521739130435,
        0.4263959390862944,
        0.6407766990291262
    ],
    device='cuda'
)


def change_layers(model: torch.nn.Module) -> None:
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            change_layers(module)

        if isinstance(module, DecomposedConv2d):
            module.compose()
            new_module = SupplementedConv2d(
                base_conv=module,
            )
            setattr(model, name, new_module)


DEC_MOD = 'spatial'
# DEC_MOD = 'channel'
PATH = f'/media/n31v/data/results/LUSC_SVD/ResNet18_SVD_{DEC_MOD}_O-10_H-0.1/0_0/train_16.sd.pt'

model = resnet18(num_classes=21)
decompose_module(model=model, decomposing_mode=DEC_MOD)
model.load_state_dict(torch.load(PATH))
change_layers(model)

exp = ClassificationExperimenter(
    model=model,
    name=f'ResNet18_{DEC_MOD}_W_balanced',
    loss=CrossEntropyLoss(weight=(1 - metrics)),
)

dataset = ImageFolder(
    root='/media/n31v/data/datasets/Land-Use_Scene_Classification/images',
    transform=Compose([
        ToTensor(),
        Resize((200, 200), antialias=True),
        Normalize(mean=(0.462, 0.471, 0.440), std=(0.287, 0.279, 0.269))
    ])
)
ds_folds = np.load('/media/n31v/data/datasets/Land-Use_Scene_Classification/images/folds.npy')
train_ds = Subset(dataset=dataset, indices=ds_folds[0, 0, :])
val_ds = Subset(dataset=dataset, indices=ds_folds[0, 1, :])
dl_params = {'batch_size': 32, 'num_workers': 8}

start_t = datetime.now()
exp.fit(p=FitParameters(
    dataset_name='LUSC_SVD',
    train_dl=DataLoader(train_ds, shuffle=True, **dl_params),
    val_dl=DataLoader(val_ds, **dl_params),
    num_epochs=30,
    optimizer_params={'lr': 0.0001},
    lr_scheduler=StepLR,
    lr_scheduler_params={'step_size': 5, 'gamma': 0.2},
    summary_path='/media/n31v/data/results',
    models_path='/media/n31v/data/results',
    description='0_0',
    class_metrics=True
))
print(f'Total time: {datetime.now() - start_t}')
