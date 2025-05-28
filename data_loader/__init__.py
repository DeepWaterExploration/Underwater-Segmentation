from .cityscapes import CitySegmentation
from .underwater_seg import SUIMDataset

datasets = {
    'citys': CitySegmentation,
    'suim': SUIMDataset
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
