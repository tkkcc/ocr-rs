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

# pytorch port
cd python
rye run ddddocr_pytorch ../sample/79.png
```

benchmark on my laptop (run above with --test-speed)
```txt
pytorch
79: 9.3ms
longsingleline: 51.9ms

candle(default feature)
79: 65ms
longsingleline: 770ms

ort(default feature)
79: 3ms
longsingleline: 47ms
```

benchmark on android emulator 7.0 x86
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
