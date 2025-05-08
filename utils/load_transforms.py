"""
This file contains a transforms class that contains the different transforms given to it by the configuration file.
"""

from monai import transforms
import munch
from utils import my_transforms


class load_transforms:
    def __init__(self, kwargs):
        self.args = kwargs

    def init_transforms(self):
        self._init_transforms('transforms')
        self._init_transforms('valTransforms')
        self._init_transforms('testTransforms')

    def _init_transforms(self, _T):
        T = self.args.get(_T, None)
        if T is not None:
            T = munch.Munch({**self.args.defaultTransforms, **T})
            Transforms = []
            for transform in T:
                if hasattr(transforms, transform):
                    transformFunc = getattr(transforms, transform)
                elif hasattr(my_transforms, transform):
                    transformFunc = getattr(my_transforms, transform)
                _transform = transformFunc(**T[transform])
                Transforms.append(_transform)
            setattr(self, _T, transforms.Compose(Transforms))
