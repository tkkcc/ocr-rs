#!/usr/bin/env python
from pathlib import Path
import pdb

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import json
from typing import Self
from torch import Tensor, import_ir_module
import typer

app = typer.Typer()

root_path =Path(__file__).parent
origin_onnx = root_path / "rec.onnx"


def load_charset() -> list[str]:
    charset = open(root_path / "ppocr_keys_v1.txt").read()
    charset = [""] + charset.split("\n") + [" "]
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


@app.command()
@torch.no_grad()
def test_onnx(
    img: str,
    test_speed: bool = False,
):
    i0 = Image.open(img)
    i0 = (
        torch.from_numpy(np.array(i0)).type(torch.float32).permute(2, 0, 1).unsqueeze(0)
        # .mean(1, keepdim=True)
    )
    i0 = torch.nn.functional.interpolate(
        i0, (48, int(i0.shape[-1] * 48 / i0.shape[-2])), mode="bilinear"
    )
    i0 = (i0 / 255.0 - 0.5) / 0.5
    i0 = i0.numpy()

    import onnxruntime as ort

    ort_session = ort.InferenceSession(origin_onnx)
    outputs = ort_session.run(
        None,
        {"x": i0},
    )

    outputs = outputs[0].argmax(-1).squeeze()
    charset = load_charset()
    out = "".join(charset[int(i)] for i in outputs)
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


@app.command()
@torch.no_grad()
def test_ncnn(
    img: str,
    test_speed: bool = False,
):
    i0 = Image.open(img)
    i0 = (
        torch.from_numpy(np.array(i0)).type(torch.float32).permute(2, 0, 1).unsqueeze(0)
        # .mean(1, keepdim=True)
    )
    i0 = torch.nn.functional.interpolate(
        i0, (48, int(i0.shape[-1] * 48 / i0.shape[-2])), mode="bilinear"
    )
    i0 = (i0 / 255.0 - 0.5) / 0.5
    i0 = i0.numpy()

    import ncnn

    with ncnn.Net() as net:
        net.load_param(str(root_path / "rec.ncnn.param"))
        net.load_model(str(root_path / "rec.ncnn.bin"))

        x = ncnn.Mat(i0.squeeze(0))
        with net.create_extractor() as ex:
            ex.input("in0", x.clone())

            _, out0 = ex.extract("out0")
            # print(out0.c, out0.h, out0.w)
            y = np.array(out0).argmax(-1).tolist()
            # print(y)

        charset = load_charset()
        out = "".join(charset[int(i)] for i in y)
        print(out)
    if test_speed:
        net.load_param(str(root_path / "rec.ncnn.param"))
        net.load_model(str(root_path / "rec.ncnn.bin"))

        x = ncnn.Mat(i0.squeeze(0))
        from timeit import default_timer as timer

        start = timer()
        for i in range(10):
            with net.create_extractor() as ex:
                ex.input("in0", x.clone())
                _, out0 = ex.extract("out0")
        print((timer() - start) / 10 * 1000, "ms")

    print("==> origin onnx result:")
    test_onnx(img, test_speed)


def main():
    app()
