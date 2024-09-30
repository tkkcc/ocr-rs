#!/usr/bin/env python
from pathlib import Path
from timeit import default_timer

import numpy as np
import torch
from torch.export import Dim, dynamic_dim
import torch.nn as nn
from PIL import Image
import json
from typing import Self
from torch import Tensor
import typer

app = typer.Typer()
root_path = Path(__file__).parent


class Down(nn.Module):
    def __init__(self, ci: int, co: int):
        super().__init__()
        self.m = nn.Conv2d(ci, co, 3, padding=1, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.m(x)
        return x * x.sigmoid()


class Res(nn.Module):
    def __init__(self, ci: int, c: int):
        super().__init__()
        self.m0 = nn.Conv2d(ci, c, 3, padding=1)
        self.m1 = nn.Conv2d(c, ci, 1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        x0 = x
        x = self.m0(x)
        x *= x.sigmoid()
        x = self.m1(x)
        return x + x0


class OCR(nn.Module):
    def __init__(self):
        super().__init__()

        self.m = nn.Sequential(
            Down(1, 24),
            Res(24, 24),
            Res(24, 24),
            Down(24, 96),
            nn.Conv2d(96, 48, 1),
            Res(48, 192),
            Res(48, 192),
            Res(48, 192),
            Down(48, 192),
            nn.Conv2d(192, 64, 1),
            Res(64, 256),
            Res(64, 256),
            Res(64, 256),
        )

        c = 512
        self.lstm = nn.LSTM(
            input_size=c, hidden_size=c, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(in_features=c * 2, out_features=8210)

    def forward(self, x) -> Tensor:
        x = self.m(x)
        # __import__("pdb").set_trace()

        # x = x.permute(3, 0, 1, 2)
        # w, b, c, h = x.shape
        # x = x.view(w, b, c * h)

        b, c, h, w = x.shape
        x = x.permute(0, 3, 1, 2).view(b, w, -1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

    def load_from_onnx(self) -> Self:
        onnx_weight = load_onnx(origin_onnx)

        self.m[0].m.weight.copy_(onnx_weight["391"])
        self.m[0].m.bias.copy_(onnx_weight["392"])

        self.m[1].m0.weight.copy_(onnx_weight["394"])
        self.m[1].m0.bias.copy_(onnx_weight["395"])
        self.m[1].m1.weight.copy_(onnx_weight["397"])
        self.m[1].m1.bias.copy_(onnx_weight["398"])

        self.m[2].m0.weight.copy_(onnx_weight["400"])
        self.m[2].m0.bias.copy_(onnx_weight["401"])
        self.m[2].m1.weight.copy_(onnx_weight["403"])
        self.m[2].m1.bias.copy_(onnx_weight["404"])

        self.m[3].m.weight.copy_(onnx_weight["406"])
        self.m[3].m.bias.copy_(onnx_weight["407"])

        self.m[4].weight.copy_(onnx_weight["409"])
        self.m[4].bias.copy_(onnx_weight["410"])

        self.m[5].m0.weight.copy_(onnx_weight["412"])
        self.m[5].m0.bias.copy_(onnx_weight["413"])
        self.m[5].m1.weight.copy_(onnx_weight["415"])
        self.m[5].m1.bias.copy_(onnx_weight["416"])
        self.m[6].m0.weight.copy_(onnx_weight["418"])
        self.m[6].m0.bias.copy_(onnx_weight["419"])
        self.m[6].m1.weight.copy_(onnx_weight["421"])
        self.m[6].m1.bias.copy_(onnx_weight["422"])
        self.m[7].m0.weight.copy_(onnx_weight["424"])
        self.m[7].m0.bias.copy_(onnx_weight["425"])
        self.m[7].m1.weight.copy_(onnx_weight["427"])
        self.m[7].m1.bias.copy_(onnx_weight["428"])

        self.m[8].m.weight.copy_(onnx_weight["430"])
        self.m[8].m.bias.copy_(onnx_weight["431"])
        self.m[9].weight.copy_(onnx_weight["433"])
        self.m[9].bias.copy_(onnx_weight["434"])

        self.m[10].m0.weight.copy_(onnx_weight["436"])
        self.m[10].m0.bias.copy_(onnx_weight["437"])
        self.m[10].m1.weight.copy_(onnx_weight["439"])
        self.m[10].m1.bias.copy_(onnx_weight["440"])
        self.m[11].m0.weight.copy_(onnx_weight["442"])
        self.m[11].m0.bias.copy_(onnx_weight["443"])
        self.m[11].m1.weight.copy_(onnx_weight["445"])
        self.m[11].m1.bias.copy_(onnx_weight["446"])
        self.m[12].m0.weight.copy_(onnx_weight["448"])
        self.m[12].m0.bias.copy_(onnx_weight["449"])
        self.m[12].m1.weight.copy_(onnx_weight["451"])
        self.m[12].m1.bias.copy_(onnx_weight["452"])

        def lstm_state_permute(w):
            w = w.chunk(4)
            w = torch.cat((w[0], w[2], w[3], w[1]), 0)
            return w

        self.lstm.weight_ih_l0.copy_(lstm_state_permute(onnx_weight["498"][0]))
        self.lstm.weight_ih_l0_reverse.copy_(lstm_state_permute(onnx_weight["498"][1]))
        self.lstm.weight_hh_l0.copy_(lstm_state_permute(onnx_weight["499"][0]))
        self.lstm.weight_hh_l0_reverse.copy_(lstm_state_permute(onnx_weight["499"][1]))

        self.lstm.bias_ih_l0.copy_(lstm_state_permute(onnx_weight["497"][0, :2048]))
        self.lstm.bias_ih_l0_reverse.copy_(
            lstm_state_permute(onnx_weight["497"][1, :2048])
        )
        self.lstm.bias_hh_l0.copy_(lstm_state_permute(onnx_weight["497"][0, 2048:]))
        self.lstm.bias_hh_l0_reverse.copy_(
            lstm_state_permute(onnx_weight["497"][1, 2048:])
        )

        self.fc.weight.copy_(onnx_weight["135"])
        self.fc.bias.copy_(onnx_weight["136"])

        return self

    def export_onnx(self, path: str | Path):
        # torch.onnx.dynamo_export(
        #     self,
        #     torch.ones(1, 1, 64, 128),
        # ).save(path)

        torch.onnx.export(
            self,
            torch.ones(1, 1, 64, 128),
            str(path),
            input_names=["x"],
            dynamic_axes={"x": [3]},
        )

        torch.save(self.state_dict(), root_path / "ddddocr.pth")

        import pnnx

        opt_model = pnnx.export(
            self,
            str(root_path / "ddddocr.pth"),
            [torch.ones(1, 1, 64, 128)],
            input_shapes=[[1, 1, 64, 16]],
            input_shapes2=[[1, 1, 64, 4000]],
            input_types="f32",
            input_types2="f32",
            fp16=False,
        )

        # weight = load_onnx(path)
        # weight_origin = load_onnx(origin_onnx)
        # print((weight["onnx::LSTM_331"] - weight_origin["498"]).abs().mean())
        # print((weight["onnx::LSTM_332"] - weight_origin["499"]).abs().mean())
        # print((weight["onnx::LSTM_330"] - weight_origin["497"]).abs().mean())
        # print((weight["fc.weight"] - weight_origin["135"]).abs().mean())
        # print((weight["fc.bias"] - weight_origin["136"]).abs().mean())
        # print((weight["m.1.m0.bias"] - weight_origin["395"]).abs().mean())
        # print((weight["m.1.m0.weight"] - weight_origin["394"]).abs().mean())

    def test_origin_onnx(self, i0: Tensor):
        import onnxruntime as ort

        ort_session = ort.InferenceSession(origin_onnx)
        outputs = ort_session.run(
            None,
            {"input1": i0.numpy()},
        )

        charset = load_charset()
        out = "".join(charset[int(i)] for i in outputs[0].argmax(-1))
        print(out)

    def export_safetensor(self, path: str | Path):
        from safetensors.torch import save_file
        import re

        weight = dict()
        for k, v in self.state_dict().items():
            if k.startswith("lstm") and k.endswith("_reverse"):
                old = k
                k = re.sub(r"lstm.(.*)_reverse", r"lstm_reverse.\1", k)
                print(f"rename {old} => {k}")

            weight[k] = v

        save_file(weight, path)

    def export_tflite(self, path: str | Path):
        import ai_edge_torch
        from torch.export import Dim

        edge_model = ai_edge_torch.convert(
            self.eval(),
            (torch.ones(1, 1, 64, 128),),
            dynamic_shapes=dict(x={3: Dim("width")}),
        )


def load_onnx(path: str | Path) -> dict[str, Tensor]:
    import onnx

    onnx_model = onnx.load(path)
    INTIALIZERS = onnx_model.graph.initializer
    weight = {}
    for initializer in INTIALIZERS:
        array = onnx.numpy_helper.to_array(initializer)
        weight[initializer.name] = torch.tensor(array)
    return weight


origin_onnx = root_path / "common.onnx"


def load_charset() -> list[str]:
    charset = json.load(open(root_path / "charset.json"))
    return charset


def test_ocr_speed(i0: Tensor, net: OCR):
    from timeit import default_timer as timer

    net(i0)
    start = timer()
    for i in range(10):
        net(i0)
    print((timer() - start) / 10 * 1000, "ms")


@app.command()
@torch.no_grad()
def test_torch(
    img: str,
    test_speed: bool = False,
    export_onnx: bool = False,
    export_safetensor: bool = False,
    export_tflite: bool = False,
):
    net = OCR().load_from_onnx()

    i0 = Image.open(img)
    i0 = (
        torch.from_numpy(np.array(i0))
        .type(torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .mean(1, keepdim=True)
    )
    i0 = torch.nn.functional.interpolate(
        i0, (64, int(i0.shape[-1] * 64 / i0.shape[-2])), mode="bilinear"
    )
    i0 = (i0 / 255.0 - 0.5) / 0.5

    out = net(i0)
    if test_speed:
        test_ocr_speed(i0, net)

    out = out.argmax(-1).squeeze().tolist()
    charset = load_charset()
    out = "".join(charset[i] for i in out)
    print(out)

    # net.test_origin_onnx(i0)
    if export_onnx:
        net.export_onnx(root_path / "ddddocr.onnx")
    if export_safetensor:
        net.export_safetensor(root_path / "ddddocr.safetensors")
    if export_tflite:
        net.export_tflite(root_path / "ddddocr.tflite")


@app.command()
@torch.no_grad()
def test_onnx(
    img: str,
    test_speed: bool = False,
):
    i0 = Image.open(img)
    i0 = (
        torch.from_numpy(np.array(i0))
        .type(torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .mean(1, keepdim=True)
    )
    i0 = torch.nn.functional.interpolate(
        i0, (64, int(i0.shape[-1] * 64 / i0.shape[-2])), mode="bilinear"
    )
    i0 = (i0 / 255.0 - 0.5) / 0.5

    import onnxruntime as ort

    # ort_session = ort.InferenceSession(origin_onnx)
    ort_session = ort.InferenceSession(root_path / "ddddocr.onnx")
    name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(
        None,
        {name: i0.numpy()},
    )

    charset = load_charset()
    out = "".join(charset[int(i)] for i in outputs[0].argmax(-1).squeeze())
    print(out)

    if test_speed:
        start = default_timer()
        for i in range(10):
            outputs = ort_session.run(
                None,
                {name: i0.numpy()},
            )
        print((default_timer() - start) / 10 * 1000, "ms")


@app.command()
@torch.no_grad()
def test_ncnn(
    img: str,
    test_speed: bool = False,
):
    i0 = Image.open(img)
    i0 = (
        torch.from_numpy(np.array(i0))
        .type(torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .mean(1, keepdim=True)
    )
    i0 = torch.nn.functional.interpolate(
        i0, (64, int(i0.shape[-1] * 64 / i0.shape[-2])), mode="bilinear"
    )
    i0 = (i0 / 255.0 - 0.5) / 0.5

    import ncnn

    x = ncnn.Mat(i0.numpy())
    with ncnn.Net() as net:
        net.load_param(str(root_path / "ddddocr.ncnn.param"))
        net.load_model(str(root_path / "ddddocr.ncnn.bin"))

        with net.create_extractor() as ex:
            ex.input("in0", x.clone())
            _, out0 = ex.extract("out0")

            y = out0.numpy().argmax(-1).squeeze().tolist()

        charset = load_charset()
        out = "".join(charset[int(i)] for i in y)
        print(out)

        if test_speed:
            start = default_timer()
            for i in range(10):
                with net.create_extractor() as ex:
                    ex.input("in0", x.clone())
                    _, out0 = ex.extract("out0")
            print((default_timer() - start) / 10 * 1000, "ms")


def main():
    app()
