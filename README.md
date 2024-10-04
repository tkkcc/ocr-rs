# OCR models port to rust

fast to slow on android cpu:  
ncnn,ort > candle > tract > burn(ndarray)

not tested:  
mnn,tvm,paddlelite,litert

## ddddocr

1. support chinese / english single line text
2. support text captcha

based on [onnx model](https://github.com/sml2h3/ddddocr/blob/master/ddddocr/common.onnx)

### run
```sh
# candle
cargo run --release -p ddddocr_candle sample/79.png

# ort
cargo run --release -p ddddocr_ort sample/79.png

# pytorch
cd python
rye sync
rye run ddddocr test-torch ../sample/79.png

# onnxruntime
rye run ddddocr test-onnx ../sample/79.png

# ncnn
rye run ddddocr test-torch ../sample/79.png --export-onnx
rye run ddddocr test-ncnn ../sample/79.png
```


### benchmark

on my laptop (run above with --test-speed)

```txt
pytorch
79: 9.3ms
longsingleline: 51.9ms

onnxruntime
79: 4.2ms
longsingleline: 35ms

ncnn
79: 7.6ms
longsingleline: 64ms

candle(default feature)
79: 65ms
longsingleline: 770ms

ort(default feature)
79: 3ms
longsingleline: 42ms
```

on genymotion android 7.0 x86
```txt
onnxruntime in kotlin
79: 24ms
longsingleline: 280ms

candle in rust
79: 84ms
longsingleline: 1222ms

tract in rust
79: 170ms
longsingleline: 3007ms

```


## paddleocr

1. support multilingual
2. support multiline and singleline

origin model is from https://paddlepaddle.github.io/PaddleOCR/model/index.html
```sh
wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar
tar xvf ch_PP-OCRv4_rec_infer.tar
wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar
tar xvf ch_PP-OCRv4_det_infer.tar
```
then converted to onnx via paddle2onnx
```sh
paddle2onnx --model_dir ./ch_PP-OCRv4_rec_infer/ \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file rec.onnx \
--opset_version 19 \
--enable_onnx_checker True
paddle2onnx --model_dir ./ch_PP-OCRv4_det_infer/ \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file det.onnx \
--opset_version 19 \
--enable_onnx_checker True
```

reference

- https://github.com/jingsongliujing/OnnxOCR
- https://github.com/7rah/paddleocr-rust-ncnn

### run

```sh
# onnxruntime
cd python
rye sync
rye run paddleocr test-onnx ../sample/79.png

# ncnn
cd python
rye run pnnx paddleocr/rec.onnx 'inputshape=[1,3,48,16]f32' 'inputshape2=[1,3,48,32000]f32' fp16=0
rye run pnnx paddleocr/det.onnx 'inputshape=[1,3,32,32]f32' 'inputshape2=[1,3,3200,3200]f32' fp16=0
rye run paddleocr test-ncnn ../sample/79.png
```

### benchmark

on my laptop (run above with --test-speed)

```txt
onnxruntime
79: 5ms
longsingleline: 54ms

ncnn
79: 2.4ms
longsingleline: 34ms
```

on genymotion android 7.0 x86
```txt
onnxruntime(intra_thread=1) in kotlin
79: 15ms
longsingleline: 199ms

onnxruntime(intra_thread=2) in rust
79: 12ms
longsingleline: 135ms

ncnn(thread=2) in rust
79: 8ms
longsingleline: 93ms

```

on genymotion android 11.0 x86_64
```txt
onnxruntime in kotlin
longsingleline: 145ms
```
on android studio emulator 9.0 x86
```txt
onnxruntime in kotlin
longsingleline: 113ms
```
on android studio emulator 12.0 x86_64
```txt
onnxruntime in kotlin
longsingleline: 104ms
```

## mlkit

1. multiline mode only
2. closed source

### benchmark

on android studio emulator 12.0 x86_64
```txt
longsingleline(height=48): 264ms
```
