from mmcv.utils import Registry
PIPELINES = Registry('pipeline')

from .compose import Compose
from .formatting import ImageToTensor, Collect
from .test_time_aug import MultiScaleFlipAug
from .transforms import Resize, RandomFlip, Normalize

