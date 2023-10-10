from datetime import datetime
from functools import partial
import logging
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.nn import CrossEntropyLoss
from torchvision.datasets import wrap_dataset_for_transforms_v2, VOCSegmentation
from torchvision.transforms import v2 as T
from fedot_ind.core.models.cnn.unet import UNet
from fedot_ind.core.architecture.experiment.nn_experimenter import SegmentationExperimenter, FitParameters
from fedot_ind.core.operation.optimization.structure_optimization import SFPOptimization, SVDOptimization

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(name)s - %(message)s')

CE = CrossEntropyLoss(ignore_index=255)


def CEloss(pred, y: torch.Tensor):
    y = y.type(torch.cuda.LongTensor)
    y = torch.squeeze(y)
    return CE(pred, y)


start_t = datetime.now()
model = UNet(n_channels=3, n_classes=21)
exp = SegmentationExperimenter(
    model=model,
    loss=CEloss,
)

ds_params = dict(
    root='/media/n31v/data/datasets/VOC',
    transforms=T.Compose([T.ToTensor(), T.Resize((400, 500), antialias=True)])
)
train_ds = wrap_dataset_for_transforms_v2(VOCSegmentation(**ds_params))
val_ds = wrap_dataset_for_transforms_v2(VOCSegmentation(**ds_params, image_set='val'))

dl_params = dict(batch_size=8, num_workers=8)
fit_params = FitParameters(
    dataset_name='VOCSegmentation',
    train_dl=DataLoader(dataset=train_ds, shuffle=True, **dl_params),
    val_dl=DataLoader(dataset=val_ds, **dl_params),
    optimizer=partial(torch.optim.SGD, lr=0.001),
    num_epochs=1000,
    # lr_scheduler=partial(ReduceLROnPlateau, mode='max', factor=0.3, patience=10, verbose=True),
    lr_scheduler=partial(CosineAnnealingWarmRestarts, T_0=20),
    models_path='/media/n31v/data/results/',
    summary_path='/media/n31v/data/results/',
    validation_period=5,
    description='SGD/CosineAnnealingWarmRestarts',
    class_metrics=True
)

exp.fit(fit_params)

print(f'Total time: {datetime.now() - start_t}')
