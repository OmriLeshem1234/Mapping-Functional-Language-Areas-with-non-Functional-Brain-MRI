"""
This file is a dataloader class which does the following:
- Creates a data list.
- Creates a dataset from the data list.
- Creates a dataloader to wrap the dataset.
"""

import monai
from pathlib2 import Path


class dataloader:
    def __init__(self, args, transform):
        self.args = args
        self.transform = transform

    def create_dataloader(self):
        self.transform.init_transforms()
        for set in self.args.sets:
            self._create_data_list(set)
            self._create_dataset(set)
            self._create_dataloader(set)

    def _create_data_list(self, _T):
        json_path = Path(self.args.project_dir) / self.args.json_path
        data_list = monai.data.load_decathlon_datalist(json_path,
                                                       data_list_key=_T,
                                                       is_segmentation=getattr(self.args, 'is_segmentation', False),
                                                       base_dir=self.args.project_dir)
        setattr(self, _T + '_list', data_list)

    def _create_dataset(self, _T):
        dataList = getattr(self, _T + '_list')
        datasetFunc = getattr(monai.data, self.args.dataset_type, 'Dataset')
        dataset = datasetFunc(data=dataList, transform=getattr(self.transform, self.args.set2transforms[_T]))
        setattr(self, _T + '_dataset', dataset)

    def _create_dataloader(self, _T):
        dataset = getattr(self, _T + '_dataset')
        dataloader = monai.data.DataLoader(dataset=dataset,
                                           batch_size=self.args.batch_size,
                                           shuffle=self.args.shuffle_data)
        setattr(self, _T + '_dataloader', dataloader)
