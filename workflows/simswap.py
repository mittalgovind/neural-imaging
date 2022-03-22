#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# New York University 
# By: Govind (mittal@nyu.edu)

# Standard libraries

# External libraries
import torch.onnx.symbolic_opset8
from tqdm import tqdm

# Internal libraries


def swap(faces1, faces2):
    swapped_faces = torch.onnx.symbolic_opset8.empty((-1, *faces1.shape[1:])
    for face1, face2 in tqdm(zip(faces1, faces2)):

