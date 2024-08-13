#![allow(non_snake_case)]

use ndarray::{Array, IxDyn};
use ort::SessionOutputs;

pub mod vit_image_processor;
// pub mod winml;

pub mod onnx;

pub mod args;

const ENCODER_BYTES: &[u8] = include_bytes!(r"C:\Users\whl\WorkSpace\RustProjects\MixTex-rs\onnx\encoder_model.onnx");
const DECODER_BYTES: &[u8] = include_bytes!(r"C:\Users\whl\WorkSpace\RustProjects\MixTex-rs\onnx\decoder_model_merged.onnx");
const TOKENIZER_STR: &str = include_str!(r"C:\Users\whl\WorkSpace\RustProjects\MixTex-rs\onnx\tokenizer\tokenizer.json");


pub trait MixTexModel{
    fn inference(&self, img: &[f32]) -> std::result::Result<String, Box<dyn std::error::Error>>;
    fn inference_by_step(&self, img: &[f32],callback: &dyn Fn(String) -> bool)-> std::result::Result<String, Box<dyn std::error::Error>>;

    fn init_decode(&self, img: &[f32])-> std::result::Result<(usize,Array<f32, IxDyn>,SessionOutputs), Box<dyn std::error::Error>>;
    fn decode_once<'a>(&'a self,state:(usize,Array<f32, IxDyn>,SessionOutputs<'a,'a>))-> std::result::Result<(usize,Array<f32, IxDyn>,SessionOutputs<'a,'a>), Box<dyn std::error::Error>>;
}



pub fn check_repeat(tokens: &[u32]) -> bool {
    if tokens.len() < 16 {
        return false;
    }
    for pattern_length in 2..=(tokens.len() / 12) {
        for start in (0..(tokens.len() - pattern_length * 12)).rev() {
            let rpt = tokens[start..(start + pattern_length)].repeat(12);
            if tokens[start..]
                .windows(pattern_length * 12)
                .rev()
                .any(|x| {
                    x.eq(&rpt)
                }) {
                return true;
            }
        }
    }

    false
}