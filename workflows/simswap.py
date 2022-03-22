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
    swapped_faces = torch.onnx.symbolic_opset8.empty((-1, *faces1.shape[1:]))
    for face1, face2 in tqdm(zip(faces1, faces2)):
        # face1 -> face2
        img_a_align_crop, _ = app.get(img_a_whole, crop_size)
        img_a_align_crop_pil = Image.fromarray(
            cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB))
        img_a = transformer_Arcface(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        # convert numpy to tensor
        img_id = img_id.cuda()
        swapped_faces.cat(swap_face1)
        # face2 -> face1
        swapped_faces.cat(swap_face2)

    return swapped_faces