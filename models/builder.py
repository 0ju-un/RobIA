
from .depth_estimators import (CoEx, RobIA)

models_lut = {
    'coex': CoEx,
    'robia': RobIA,
}

def build_model(model_name):
    if model_name in models_lut:
        return models_lut[model_name]
    else:
        raise NotImplementedError

