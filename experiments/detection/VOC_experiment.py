from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision.transforms import ToTensor
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, fasterrcnn_resnet50_fpn
from fedot_ind.core.architecture.experiment.nn_experimenter import ObjectDetectionExperimenter, FitParameters
from fedot_ind.core.operation.optimization.structure_optimization import SFPOptimization, SVDOptimization

class_to_idx = {'person': 1, 'bird': 2, 'cat': 3, 'cow': 4, 'dog': 5, 'horse': 6, 'sheep': 7, 'aeroplane': 8,
                'bicycle': 9, 'boat': 10, 'bus': 11, 'car': 12, 'motorbike': 13, 'train': 14, 'bottle': 15, 'chair': 16,
                'dining table': 17, 'potted plant': 18, 'sofa': 19, 'tv/monitor':20}

def voc2coco(target):
    for object in target['object']:



    return {
        'boxes': torch.tensor(boxes, dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.int64),
        'image_id': torch.tensor([0]),
        'area': torch.tensor(area, dtype=torch.float32),
        'iscrowd': torch.zeros(labels.shape[0], dtype=torch.int64),
    }


start_t = datetime.now()

exp = ObjectDetectionExperimenter(
    model=ssdlite320_mobilenet_v3_large()
)

train_ds = VOCDetection('/media/n31v/data/datasets/VOC', transform=ToTensor())
val_ds = VOCDetection('/media/n31v/data/datasets/VOC', image_set='val', transform=ToTensor())

dl_params = dict(batch_size=16, num_workers=8, collate_fn=lambda x: tuple(zip(*x)))
fit_params = FitParameters(
    dataset_name='VOC',
    train_dl=DataLoader(dataset=train_ds, shuffle=True, **dl_params),
    val_dl=DataLoader(dataset=val_ds, **dl_params),
    num_epochs=100,
    models_path='/media/n31v/data/results/',
    summary_path='/media/n31v/data/results/',
    class_metrics=True
)

exp.fit(fit_params)

print(f'Total time: {datetime.now() - start_t}')
