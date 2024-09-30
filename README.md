# OCR models port to rust

## ddddocr

based on ddddocr's [onnx model](https://github.com/sml2h3/ddddocr/blob/master/ddddocr/common.onnx)

1. support chinese / english single line text
2. support text captcha

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

benchmark on my laptop (run above with --test-speed)
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

benchmark on genymotion android 7.0 x86
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


## paddleocr v4

based on rec model in https://github.com/jingsongliujing/OnnxOCR

```sh
# onnxruntime
cd python
rye sync
rye run paddleocr test-onnx ../sample/79.png

# ncnn
cd python
rye run pnnx paddleocr/rec.onnx 'inputshape=[1,3,48,16]f32' 'inputshape2=[1,3,48,4000]f32' fp16=0
rye run paddleocr test-ncnn ../sample/79.png
```

benchmark on my laptop (run above with --test-speed)
```txt
onnxruntime
79: 5ms
longsingleline: 54ms

ncnn
79: 2.4ms
longsingleline: 34ms
```

benchmark on genymotion android 7.0 x86
```txt
onnxruntime in kotlin
79: 15ms
longsingleline: 199ms
```
benchmark on genymotion android 11.0 x86_64
```txt
onnxruntime in kotlin
longsingleline: 145ms
```
benchmark on android studio emulator 9.0 x86
```txt
onnxruntime in kotlin
longsingleline: 113ms
```
benchmark on android studio emulator 12.0 x86_64
```txt
onnxruntime in kotlin
longsingleline: 104ms
```

## mlkit

```txt
benchmark on android studio emulator 12.0 x86_64
longsingleline(height=48): 264ms
```
