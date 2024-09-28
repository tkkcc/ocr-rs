# OCR models port to rust

## ddddocr

based on ddddocr's [onnx model](https://github.com/sml2h3/ddddocr/blob/master/ddddocr/common.onnx)

1. support chinese / english single line text
2. support text captcha

```sh
# candle port
cargo run --release -p ddddocr_candle sample/79.png

# pytorch port
cd python
rye run ddddocr_pytorch ../sample/79.png
```

benchmark on my laptop (run with --test-speed)
```txt
pytorch vs candle(default feature)
79.png: 9.3ms vs 70ms
longsingleline.png: 51.9ms vs 903ms
```
