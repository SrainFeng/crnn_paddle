# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：crnn_paddle 
@File    ：__init__.py.py
@Author  ：Srain
@Date    ：2021/5/20 16:58
"""

from ._360cc import _360CC
from ._own import _OWN


def get_dataset(config):

    if config.DATASET.DATASET == "360CC":
        return _360CC
    elif config.DATASET.DATASET == "OWN":
        return _OWN
    else:
        raise NotImplemented()
