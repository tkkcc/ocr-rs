#!/usr/bin/env python
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class Down(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.m = nn.Conv2d(ci, co, 3, padding=1, stride=2)

    def forward(self, x):
        x = self.m(x)
        return x * x.sigmoid()


class Res(nn.Module):
    def __init__(self, ci, c):
        super().__init__()
        self.m0 = nn.Conv2d(ci, c, 3, padding=1)
        self.m1 = nn.Conv2d(c, ci, 1, padding=0)

    def forward(self, x):
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
        self.lstm = nn.LSTM(input_size=c, hidden_size=c, bidirectional=True)
        self.fc = nn.Linear(in_features=c * 2, out_features=8210)

    def forward(self, x):
        x = self.m(x)
        # __import__("pdb").set_trace()

        x = x.permute(3, 0, 1, 2)
        w, b, c, h = x.shape
        x = x.view(w, b, c * h)
        x, _ = self.lstm(x)

        t, _, h = x.shape
        # print(68, t,h)
        x = x.view(t * b, h)
        x = self.fc(x)
        x = x.view(t, b, -1)
        return x


@torch.no_grad()
def test_ocr(img: str):
    import onnx
    import json
    from onnx import numpy_helper

    charset = json.load(open(Path(__file__).parent / "charset.json"))

    net = OCR()

    onnx_weight_path = Path(__file__).parent / "common.onnx"
    onnx_model = onnx.load(onnx_weight_path)
    INTIALIZERS = onnx_model.graph.initializer
    onnx_weight = {}
    for initializer in INTIALIZERS:
        W = numpy_helper.to_array(initializer)
        onnx_weight[initializer.name] = torch.tensor(W)

    net.m[0].m.weight.copy_(onnx_weight["391"])
    net.m[0].m.bias.copy_(onnx_weight["392"])

    net.m[1].m0.weight.copy_(onnx_weight["394"])
    net.m[1].m0.bias.copy_(onnx_weight["395"])
    net.m[1].m1.weight.copy_(onnx_weight["397"])
    net.m[1].m1.bias.copy_(onnx_weight["398"])

    net.m[2].m0.weight.copy_(onnx_weight["400"])
    net.m[2].m0.bias.copy_(onnx_weight["401"])
    net.m[2].m1.weight.copy_(onnx_weight["403"])
    net.m[2].m1.bias.copy_(onnx_weight["404"])

    net.m[3].m.weight.copy_(onnx_weight["406"])
    net.m[3].m.bias.copy_(onnx_weight["407"])

    net.m[4].weight.copy_(onnx_weight["409"])
    net.m[4].bias.copy_(onnx_weight["410"])

    net.m[5].m0.weight.copy_(onnx_weight["412"])
    net.m[5].m0.bias.copy_(onnx_weight["413"])
    net.m[5].m1.weight.copy_(onnx_weight["415"])
    net.m[5].m1.bias.copy_(onnx_weight["416"])
    net.m[6].m0.weight.copy_(onnx_weight["418"])
    net.m[6].m0.bias.copy_(onnx_weight["419"])
    net.m[6].m1.weight.copy_(onnx_weight["421"])
    net.m[6].m1.bias.copy_(onnx_weight["422"])
    net.m[7].m0.weight.copy_(onnx_weight["424"])
    net.m[7].m0.bias.copy_(onnx_weight["425"])
    net.m[7].m1.weight.copy_(onnx_weight["427"])
    net.m[7].m1.bias.copy_(onnx_weight["428"])

    net.m[8].m.weight.copy_(onnx_weight["430"])
    net.m[8].m.bias.copy_(onnx_weight["431"])
    net.m[9].weight.copy_(onnx_weight["433"])
    net.m[9].bias.copy_(onnx_weight["434"])

    net.m[10].m0.weight.copy_(onnx_weight["436"])
    net.m[10].m0.bias.copy_(onnx_weight["437"])
    net.m[10].m1.weight.copy_(onnx_weight["439"])
    net.m[10].m1.bias.copy_(onnx_weight["440"])
    net.m[11].m0.weight.copy_(onnx_weight["442"])
    net.m[11].m0.bias.copy_(onnx_weight["443"])
    net.m[11].m1.weight.copy_(onnx_weight["445"])
    net.m[11].m1.bias.copy_(onnx_weight["446"])
    net.m[12].m0.weight.copy_(onnx_weight["448"])
    net.m[12].m0.bias.copy_(onnx_weight["449"])
    net.m[12].m1.weight.copy_(onnx_weight["451"])
    net.m[12].m1.bias.copy_(onnx_weight["452"])

    def lstm_state_permute(w):
        w = w.chunk(4)
        w = torch.cat((w[0], w[2], w[3], w[1]), 0)
        return w

    net.lstm.weight_ih_l0.copy_(lstm_state_permute(onnx_weight["498"][0]))
    net.lstm.weight_ih_l0_reverse.copy_(lstm_state_permute(onnx_weight["498"][1]))
    net.lstm.weight_hh_l0.copy_(lstm_state_permute(onnx_weight["499"][0]))
    net.lstm.weight_hh_l0_reverse.copy_(lstm_state_permute(onnx_weight["499"][1]))

    net.lstm.bias_ih_l0.copy_(lstm_state_permute(onnx_weight["497"][0, :2048]))
    net.lstm.bias_ih_l0_reverse.copy_(lstm_state_permute(onnx_weight["497"][1, :2048]))
    net.lstm.bias_hh_l0.copy_(lstm_state_permute(onnx_weight["497"][0, 2048:]))
    net.lstm.bias_hh_l0_reverse.copy_(lstm_state_permute(onnx_weight["497"][1, 2048:]))

    net.fc.weight.copy_(onnx_weight["135"])
    net.fc.bias.copy_(onnx_weight["136"])

    i0 = Image.open(img)
    i0 = torch.from_numpy(np.array(i0)) / 255.0
    i0 = (i0 - 0.5) / 0.5
    i0 = i0.permute(2, 0, 1).unsqueeze(0).mean(1, keepdim=True)
    i0 = torch.nn.functional.interpolate(
        i0, (64, int(i0.shape[-1] * 64 / i0.shape[-2])), mode="bilinear"
    )

    out: torch.Tensor = net(i0)
    out = out.argmax(-1).squeeze().tolist()
    out = "".join(charset[i] for i in out)
    print(out)

    # torch.onnx.export(
    #     net,
    #     torch.ones(1, 1, 64, 128),
    #     "tmp.onnx",
    # )
    #
    # onnx_model = onnx.load("tmp.onnx")
    # INTIALIZERS = onnx_model.graph.initializer
    # onnx_weight_new = {}
    # for initializer in INTIALIZERS:
    #     W = numpy_helper.to_array(initializer)
    #     onnx_weight_new[initializer.name] = torch.tensor(W)

    # print(onnx_weight["498"].shape)
    # print((onnx_weight_new["onnx::LSTM_331"] - onnx_weight["498"]).abs().mean())
    # print((onnx_weight_new["onnx::LSTM_332"] - onnx_weight["499"]).abs().mean())
    # print((onnx_weight_new["onnx::LSTM_330"] - onnx_weight["497"]).abs().mean())
    # print((onnx_weight_new["fc.weight"] - onnx_weight["135"]).abs().mean())
    # print((onnx_weight_new["fc.bias"] - onnx_weight["136"]).abs().mean())
    # print((onnx_weight_new["m.1.m0.bias"] - onnx_weight["395"]).abs().mean())
    # print((onnx_weight_new["m.1.m0.weight"] - onnx_weight["394"]).abs().mean())

    # import onnxruntime as ort
    #
    # ort_session = ort.InferenceSession(onnx_weight_path)
    # outputs = ort_session.run(
    #     None,
    #     {"input1": i0.numpy()},
    # )
    # out = "".join(charset[int(i)] for i in outputs[0].argmax(-1))
    # print(out)
