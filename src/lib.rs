#![allow(non_snake_case)]
pub mod vit_image_processor;
pub mod winml;

pub mod onnx;

const ENCODER_BYTES: &[u8] = include_bytes!(r"C:\Users\whl\WorkSpace\RustProjects\MixTex-rs\onnx\encoder_model.onnx");
const DECODER_BYTES: &[u8] = include_bytes!(r"C:\Users\whl\WorkSpace\RustProjects\MixTex-rs\onnx\decoder_model_merged.onnx");
const TOKENIZER_STR: &str = include_str!(r"C:\Users\whl\WorkSpace\RustProjects\MixTex-rs\onnx\tokenizer\tokenizer.json");

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