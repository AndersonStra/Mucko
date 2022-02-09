from typing import Dict, List, Union

import json



from tqdm import tqdm


class ImageFeaturesHdfReader(object):
    """
    A reader for HDF files containing pre-extracted image features. A typical HDF file is expected
    to have a column named "image_id", and another column named "features".

    Example of an HDF file:
    ```
    visdial_train_faster_rcnn_bottomup_features.h5
       |--- "image_ids" [shape: (num_images, )]
       |--- "image_features" [shape: (num_images, num_proposals, feature_size)]
       |--- "image_relations" [shape: (num_images, num_proposals, num_proposals, feature_size)]

       +--- .attrs ("split", "train")
    ```
    Refer ``$PROJECT_ROOT/data/extract_bottomup.py`` script for more details about HDF structure.

    Parameters
    ----------
    features_hdfpath : str
        Path to an HDF file containing VisDial v1.0 train, val or test split image features.
    in_memory : bool
        Whether to load the whole HDF file in memory. Beware, these files are sometimes tens of GBs
        in size. Set this to true if you have sufficient RAM - trade-off between speed and memory.
    """

    def __init__(self, features_hdfpath: str, in_memory):

        self.features_hdfpath = features_hdfpath
        self._in_memory = in_memory

        with h5py.File(self.features_hdfpath, "r") as features_hdf:
            self.image_id_list = list(features_hdf["image_ids"])
            self.features = [None] * len(self.image_id_list)
            self.relations = [None] * len(self.image_id_list)  # relation

    def __len__(self):
        return len(self.image_id_list)

    def __getitem__(self, image_id: int):
        index = self.image_id_list.index(image_id)
        if self._in_memory:
            # Load features during first epoch, all not loaded together as it has a slow start.
            if self.features[index] is not None:
                image_id_features = self.features[index]
                image_id_relations = self.relations[index]  # relation
            else:
                with h5py.File(self.features_hdfpath, "r") as features_hdf:
                    image_id_features = features_hdf["image_features"][index]
                    image_id_relations = features_hdf["image_relations"][index]
                    self.features[index] = image_id_features
                    self.relations[index] = image_id_relations

        else:
            # Read chunk from file everytime if not loaded in memory.
            with h5py.File(self.features_hdfpath, "r") as features_hdf:
                image_id_features = features_hdf["image_features"][index]
                image_id_relations = features_hdf["image_relations"][index]  

        return image_id_features, image_id_relations

    def keys(self) -> List[int]:
        return self.image_id_list

