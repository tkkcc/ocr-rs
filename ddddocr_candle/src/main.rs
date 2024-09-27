use std::{env, error::Error, fs::File, io::BufReader, path::PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::{
    conv2d, linear, lstm, ops::sigmoid, Conv2d, Conv2dConfig, Linear, Module, VarBuilder, VarMap,
    LSTM, RNN,
};
use image::GenericImageView;

struct Down {
    m: Conv2d,
}
impl Down {
    fn load(vb: VarBuilder, ci: usize, co: usize) -> candle_core::Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            stride: 2,
            ..Default::default()
        };
        Ok(Self {
            m: conv2d(ci, co, 3, cfg, vb.pp("m"))?,
        })
    }
}

impl Module for Down {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let x = &self.m.forward(&xs)?;
        x * sigmoid(x)?
    }
}
struct Res {
    m0: Conv2d,
    m1: Conv2d,
}
impl Res {
    fn load(vb: VarBuilder, ci: usize, c: usize) -> candle_core::Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        Ok(Self {
            m0: conv2d(ci, c, 3, cfg, vb.pp("m0"))?,
            m1: conv2d(c, ci, 1, Default::default(), vb.pp("m1"))?,
        })
    }
}
impl Module for Res {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = self.m0.forward(xs)?;
        x = (&x * sigmoid(&x)?)?;
        x = self.m1.forward(&x)?;
        xs + x
    }
}

struct Ocr {
    m: Vec<Box<dyn Module>>,
    lstm: LSTM,
    lstm_reverse: LSTM,
    fc: Linear,
}
impl Ocr {
    fn load(vb: VarBuilder) -> candle_core::Result<Self> {
        let mut i = 0;
        let mut count = || -> usize {
            i += 1;
            i - 1
        };
        let c = 512;
        Ok(Self {
            m: vec![
                Box::new(Down::load(vb.pp(count()), 1, 24)?),
                Box::new(Res::load(vb.pp(count()), 24, 24)?),
                Box::new(Res::load(vb.pp(count()), 24, 24)?),
                Box::new(Down::load(vb.pp(count()), 24, 96)?),
                Box::new(conv2d(96, 48, 1, Default::default(), vb.pp(count()))?),
                Box::new(Res::load(vb.pp(count()), 48, 192)?),
                Box::new(Res::load(vb.pp(count()), 48, 192)?),
                Box::new(Res::load(vb.pp(count()), 48, 192)?),
                Box::new(Down::load(vb.pp(count()), 48, 192)?),
                Box::new(conv2d(192, 64, 1, Default::default(), vb.pp(count()))?),
                Box::new(Res::load(vb.pp(count()), 64, 256)?),
                Box::new(Res::load(vb.pp(count()), 64, 256)?),
                Box::new(Res::load(vb.pp(count()), 64, 256)?),
            ],
            lstm: lstm(c, c, Default::default(), vb.pp("lstm"))?,
            lstm_reverse: lstm(c, c, Default::default(), vb.pp("lstm_reverse"))?,
            fc: linear(c * 2, 8210, vb.pp("fc"))?,
        })
    }
}

impl Module for Ocr {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = xs.clone();
        for (i, m) in self.m.iter().enumerate() {
            x = m.forward(&x)?;
        }
        let (b, c, h, w) = x.dims4()?;
        x = x.reshape((b, c * h, w))?;
        let inp_sequence = x.chunk(w, 2)?;

        let mut state = vec![self.lstm.zero_state(b)?];
        for inp in inp_sequence.iter() {
            // dbg!(inp.shape());
            // dbg!(states.last().unwrap().h.shape());
            state.push(self.lstm.step(&inp.squeeze(2)?, &state.last().unwrap())?);
        }
        let mut state_reverse = vec![self.lstm_reverse.zero_state(b)?];
        for inp in inp_sequence.iter().rev() {
            state_reverse.push(
                self.lstm_reverse
                    .step(&inp.squeeze(2)?, &state_reverse.last().unwrap())?,
            );
        }
        let h: Vec<_> = state
            .into_iter()
            .skip(1)
            .zip(state_reverse.into_iter().skip(1).rev())
            .map(|(a, b)| Tensor::cat(&[a.h, b.h], 1).unwrap())
            .collect();
        // dbg!(h[0].shape(), h.len());
        // panic!();
        x = Tensor::stack(&h, 0)?;
        x = self.fc.forward(&x)?;

        // Ok(Tensor::randn(0., 1., (39, 1, 8024), &Device::Cpu)?)
        Ok(x)
    }
}

fn main() -> anyhow::Result<()> {
    let img = env::args().nth(1).expect("require image path");
    let img = image::ImageReader::open(img)
        .unwrap()
        .decode()
        .unwrap()
        .to_rgb8();

    let h = img.height();
    let w = img.width();

    let data = img.into_raw();
    let data = Tensor::from_vec(data, (h as usize, w as usize, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .unsqueeze(0)?;
    let data = data
        .to_dtype(DType::F32)?
        .mean_keepdim(1)?
        .interpolate2d(64, (w * 64 / h) as usize)?;
    let data = (((data / 255.0)? - 0.5)? / 0.5)?;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    let net = Ocr::load(vb)?;
    let out = net.forward(&data)?;

    // let out = ocr(img)?;
    let out: Vec<u32> = out.argmax(2)?.squeeze(1)?.to_vec1()?;

    let mut charset = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    charset.extend(["..", "python", "ddddocr_pytorch", "charset.json"]);
    let charset: Vec<String> = serde_json::from_reader(BufReader::new(File::open(charset)?))?;

    let out: String = out
        .into_iter()
        .map(|i| charset[i as usize].clone())
        .collect();
    println!("{}", out);
    Ok(())
}
