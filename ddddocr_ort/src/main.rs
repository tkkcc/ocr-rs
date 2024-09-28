use std::{fs::File, io::BufReader, path::PathBuf, time::Instant};

use clap::Parser;
use image::{imageops::FilterType, GenericImageView};
use ndarray::{Array, Dimension};
use ndarray_stats::QuantileExt;

#[derive(Parser)]
struct Arg {
    i0: PathBuf,
    #[arg(long)]
    test_speed: bool,
}

fn main() {
    use ort::{GraphOptimizationLevel, Session};

    let weight = PathBuf::from_iter(&[
        env!("CARGO_MANIFEST_DIR"),
        "..",
        "python",
        "ddddocr_pytorch",
        "common.onnx",
    ]);

    let model = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .with_intra_threads(4)
        .unwrap()
        .commit_from_file(weight)
        .unwrap();

    let arg = Arg::parse();

    let i0 = image::ImageReader::open(arg.i0).unwrap().decode().unwrap();
    let h = i0.height();
    let w = i0.width();
    let i0 = i0.resize_exact((w * 64 / h) as _, 64, FilterType::CatmullRom);

    let mut data = Array::zeros((1, 1, 64, (w * 64 / h) as _));
    for pixel in i0.pixels() {
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b, _] = pixel.2 .0;
        data[[0, 0, y, x]] = ((((r + g + b) as f32) / 3.0 / 255.) - 0.5) / 0.5;
        // input[[0, 1, y, x]] = (g as f32) / 255.;
        // input[[0, 2, y, x]] = (b as f32) / 255.;
    }

    let input = ort::inputs![model.inputs.first().unwrap().name.clone() => data.clone()].unwrap();
    let outputs = model.run(input).unwrap();
    let k = model.outputs.first().unwrap().name.clone();
    let predictions = outputs[k].try_extract_tensor::<f32>().unwrap();
    if arg.test_speed {
        let start = Instant::now();
        for i in 0..10 {
            let input =
                ort::inputs![model.inputs.first().unwrap().name.clone() => data.clone()].unwrap();
            let outputs = model.run(input).unwrap();
        }
        dbg!(start.elapsed().as_millis() / 10);
    }
    dbg!(&predictions.shape());

    // let out = predictions.argmax().unwrap();
    // let out: Vec<_> = out.as_array_view().into_iter().copied().collect();
    // dbg!(&out);

    // let charset = PathBuf::from_iter(&[
    //     env!("CARGO_MANIFEST_DIR"),
    //     "..",
    //     "python",
    //     "ddddocr_pytorch",
    //     "charset.json",
    // ]);
    // let charset: Vec<String> =
    //     serde_json::from_reader(BufReader::new(File::open(charset).unwrap())).unwrap();

    // let out: String = out
    //     .into_iter()
    //     .map(|i| charset[i as usize].clone())
    //     .collect();
    // println!("{}", out);
}
