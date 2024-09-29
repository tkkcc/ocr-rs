#!/usr/bin/env python
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import json
from typing import Self
from torch import Tensor


origin_onnx = Path(__file__).parent / "rec.onnx"


def load_charset() -> list[str]:
    charset = open(Path(__file__).parent / "ppocr_keys_v1.txt").read()
    charset = [""] + charset.split("\n") + [' ']
    return charset


#
# def test_ocr_speed(i0: Tensor, net: OCR):
#     from timeit import default_timer as timer
#
#     net(i0)
#     start = timer()
#     for i in range(10):
#         net(i0)
#     print((timer() - start) / 10 * 1000, "ms")


@torch.no_grad()
def test_ocr(
    img: str,
    test_speed: bool = False,
    save_to_safetensor: bool = False,
    save_to_tflite: bool = False,
):

    i0 = Image.open(img)
    i0 = (
        torch.from_numpy(np.array(i0))
        .type(torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        # .mean(1, keepdim=True)
    )
    i0 = torch.nn.functional.interpolate(
        i0, (48, int(i0.shape[-1] * 48 / i0.shape[-2])), mode="bilinear"
    )
    i0 = (i0 / 255.0 - 0.5) / 0.5
    i0 = i0.numpy()

    import onnxruntime as ort

    ort_session = ort.InferenceSession(origin_onnx)
    # print(ort_session.get_inputs()[0].name)
    # print(i0.shape)
    outputs = ort_session.run(
        None,
        {"x": i0},
    )

    outputs = outputs[0].argmax(-1).squeeze()
    # print(outputs)
    charset = load_charset()
    out = "".join(charset[int(i)] for i in outputs)
    # __import__('pdb').set_trace()
    print(out)

    if test_speed:
        from timeit import default_timer as timer

        start = timer()
        for i in range(10):
            outputs = ort_session.run(
                None,
                {"x": i0},
            )
        print((timer() - start) / 10 * 1000, "ms")


def main():
    import typer

    typer.run(test_ocr)
