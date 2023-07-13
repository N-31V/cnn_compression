from datetime import datetime
from functools import partial
import logging
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torchvision.datasets import VOCDetection
from torchvision.transforms import ToTensor
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, fasterrcnn_resnet50_fpn
from fedot_ind.core.architecture.experiment.nn_experimenter import ObjectDetectionExperimenter, FitParameters
from fedot_ind.core.operation.optimization.structure_optimization import SFPOptimization, SVDOptimization


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(name)s - %(message)s')


class_to_idx = {'person': 1, 'bird': 2, 'cat': 3, 'cow': 4, 'dog': 5, 'horse': 6, 'sheep': 7, 'aeroplane': 8,
                'bicycle': 9, 'boat': 10, 'bus': 11, 'car': 12, 'motorbike': 13, 'train': 14, 'bottle': 15, 'chair': 16,
                'diningtable': 17, 'pottedplant': 18, 'sofa': 19, 'tvmonitor': 20}


def voc2coco(target):
    boxes = []
    labels = []
    area = []
    for object in target['annotation']['object']:
        labels.append(class_to_idx[object['name']])
        box = [object['bndbox']['xmin'], object['bndbox']['ymin'], object['bndbox']['xmax'], object['bndbox']['ymax']]
        box = [int(b) for b in box]
        boxes.append(box)
        area.append((box[2] - box[0]) * (box[3] - box[1]))

    return {
        'boxes': torch.tensor(boxes, dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.int64),
        'image_id': torch.tensor([0]),
        'area': torch.tensor(area, dtype=torch.float32),
        'iscrowd': torch.zeros(len(labels), dtype=torch.int64),
    }


start_t = datetime.now()

exp = ObjectDetectionExperimenter(
    model=ssdlite320_mobilenet_v3_large()
)

train_ds = VOCDetection('/media/n31v/data/datasets/VOC', transform=ToTensor(), target_transform=voc2coco)
val_ds = VOCDetection('/media/n31v/data/datasets/VOC', image_set='val', transform=ToTensor(), target_transform=voc2coco)

dl_params = dict(batch_size=16, num_workers=8, collate_fn=lambda x: tuple(zip(*x)))
fit_params = FitParameters(
    dataset_name='VOC',
    train_dl=DataLoader(dataset=train_ds, shuffle=True, **dl_params),
    val_dl=DataLoader(dataset=val_ds, **dl_params),
    num_epochs=500,
    # lr_scheduler=partial(ReduceLROnPlateau, factor=0.3, patience=10, verbose=True),
    lr_scheduler=partial(CosineAnnealingWarmRestarts, T_0=10, T_mult=2),
    models_path='/media/n31v/data/results/',
    summary_path='/media/n31v/data/results/',
    class_metrics=True,
    validation_period=5,
    description='CosineAnnealingWarmRestarts'
)

exp.fit(fit_params)

print(f'Total time: {datetime.now() - start_t}')
