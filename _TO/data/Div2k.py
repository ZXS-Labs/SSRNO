# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DIV2K dataset: DIVerse 2K resolution high quality images.
Adapted from TF Datasets: https://github.com/tensorflow/datasets/"""

import os
from pathlib import Path

import datasets

_CITATION = """
@InProceedings{Agustsson_2017_CVPR_Workshops,
author = {Agustsson, Eirikur and Timofte, Radu},
title = {NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
url = "http://www.vision.ee.ethz.ch/~timofter/publications/Agustsson-CVPRW-2017.pdf",
month = {July},
year = {2017}
} 
"""

_DESCRIPTION = """
DIV2K dataset: DIVerse 2K resolution high quality images as used for the challenges @ NTIRE (CVPR 2017 and 
CVPR 2018) and @ PIRM (ECCV 2018)
"""

_HOMEPAGE = "https://data.vision.ee.ethz.ch/cvl/DIV2K/"

_LICENSE = """
Please notice that this dataset is made available for academic research purpose only. All the images are
collected from the Internet, and the copyright belongs to the original owners. If any of the images belongs to 
you and you would like it removed, please kindly inform the authors, and they will remove it from the dataset 
immediately.
"""

_DL_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/"

_DL_URLS = {
    "train_hr": _DL_URL + "DIV2K_train_HR.zip",
    "valid_hr": _DL_URL + "DIV2K_valid_HR.zip",
    "train_bicubic_x2": _DL_URL + "DIV2K_train_LR_bicubic_X2.zip",
    "train_unknown_x2": _DL_URL + "DIV2K_train_LR_unknown_X2.zip",
    "valid_bicubic_x2": _DL_URL + "DIV2K_valid_LR_bicubic_X2.zip",
    "valid_unknown_x2": _DL_URL + "DIV2K_valid_LR_unknown_X2.zip",
    "train_bicubic_x3": _DL_URL + "DIV2K_train_LR_bicubic_X3.zip",
    "train_unknown_x3": _DL_URL + "DIV2K_train_LR_unknown_X3.zip",
    "valid_bicubic_x3": _DL_URL + "DIV2K_valid_LR_bicubic_X3.zip",
    "valid_unknown_x3": _DL_URL + "DIV2K_valid_LR_unknown_X3.zip",
    "train_bicubic_x4": _DL_URL + "DIV2K_train_LR_bicubic_X4.zip",
    "train_unknown_x4": _DL_URL + "DIV2K_train_LR_unknown_X4.zip",
    "valid_bicubic_x4": _DL_URL + "DIV2K_valid_LR_bicubic_X4.zip",
    "valid_unknown_x4": _DL_URL + "DIV2K_valid_LR_unknown_X4.zip",
    "train_bicubic_x8": _DL_URL + "DIV2K_train_LR_x8.zip",
    "valid_bicubic_x8": _DL_URL + "DIV2K_valid_LR_x8.zip",
    "train_realistic_mild_x4": _DL_URL + "DIV2K_train_LR_mild.zip",
    "valid_realistic_mild_x4": _DL_URL + "DIV2K_valid_LR_mild.zip",
    "train_realistic_difficult_x4": _DL_URL + "DIV2K_train_LR_difficult.zip",
    "valid_realistic_difficult_x4": _DL_URL + "DIV2K_valid_LR_difficult.zip",
    "train_realistic_wild_x4": _DL_URL + "DIV2K_train_LR_wild.zip",
    "valid_realistic_wild_x4": _DL_URL + "DIV2K_valid_LR_wild.zip",
}

_DATA_OPTIONS = [
    "bicubic_x2", "bicubic_x3", "bicubic_x4", "bicubic_x8", "unknown_x2",
    "unknown_x3", "unknown_x4", "realistic_mild_x4", "realistic_difficult_x4",
    "realistic_wild_x4"
]


class Div2kConfig(datasets.BuilderConfig):
    """BuilderConfig for Div2k."""

    def __init__(self, name, **kwargs):
        """Constructs a Div2kConfig."""
        if name not in _DATA_OPTIONS:
            raise ValueError("data must be one of %s" % _DATA_OPTIONS)

        super(Div2kConfig, self).__init__(name=name, **kwargs)
        self.data = name
        self.download_urls = {
            "train_lr_url": _DL_URLS["train_" + self.data],
            "valid_lr_url": _DL_URLS["valid_" + self.data],
            "train_hr_url": _DL_URLS["train_hr"],
            "valid_hr_url": _DL_URLS["valid_hr"],
        }


class Div2k(datasets.GeneratorBasedBuilder):
    """DIV2K dataset: DIVerse 2K resolution high quality images."""

    BUILDER_CONFIGS = [
        Div2kConfig(version=datasets.Version("2.0.0"), name=data) for data in _DATA_OPTIONS
    ]

    DEFAULT_CONFIG_NAME = "bicubic_x2"

    def _info(self):
        features = datasets.Features(
            {
                "lr": datasets.Value("string"),
                "hr": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        extracted_paths = dl_manager.download_and_extract(
            self.config.download_urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "lr_path": extracted_paths["train_lr_url"],
                    "hr_path": os.path.join(extracted_paths["train_hr_url"], "DIV2K_train_HR"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "lr_path": extracted_paths["valid_lr_url"],
                    "hr_path": str(os.path.join(extracted_paths["valid_hr_url"], "DIV2K_valid_HR")),
                },
            ),
        ]

    def _generate_examples(
            self, lr_path, hr_path
    ):
        """ Yields examples as (key, example) tuples. """
        extensions = {'.png'}
        for file_path in sorted(Path(lr_path).glob("**/*")):
            if file_path.suffix in extensions:
                file_path_str = str(file_path.as_posix())
                yield file_path_str, {
                    "lr": file_path_str,
                    "hr": str((Path(hr_path) / f"{str(file_path.name)[:4]}.png").as_posix())
                }
