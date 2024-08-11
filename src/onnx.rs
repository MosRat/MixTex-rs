use std::str::FromStr;

use ndarray::{concatenate, prelude::*};
use ort::{CUDAExecutionProvider, GraphOptimizationLevel, Session};
use tokenizers::Tokenizer;

use crate::{check_repeat, DECODER_BYTES, ENCODER_BYTES, TOKENIZER_STR};

const MAX_LENGTH: usize = 512;
pub struct MixTexOnnx {
    encoder_session: Session,
    decoder_session: Session,
    tokenizer: Tokenizer,
}

impl MixTexOnnx {
    pub fn build() -> Result<Self, anyhow::Error> {
        let encoder_builder = Session::builder()?;
        let decoder_builder = Session::builder()?;

        let _encoder_cuda = CUDAExecutionProvider::default()
            .with_device_id(0)
            .with_arena_extend_strategy(ort::ArenaExtendStrategy::NextPowerOfTwo)
            .with_memory_limit(2 * 1024 * 1024 * 1024)
            .with_conv_algorithm_search(ort::CUDAExecutionProviderCuDNNConvAlgoSearch::Exhaustive)
            .with_copy_in_default_stream(true);

        let _decoder_cuda = CUDAExecutionProvider::default()
            .with_device_id(0)
            .with_arena_extend_strategy(ort::ArenaExtendStrategy::NextPowerOfTwo)
            .with_memory_limit(2 * 1024 * 1024 * 1024)
            .with_conv_algorithm_search(ort::CUDAExecutionProviderCuDNNConvAlgoSearch::Exhaustive)
            .with_copy_in_default_stream(true);
        // let decoder_dm = DirectMLExecutionProvider::default().with_device_id(2);

        // if !ort::ExecutionProvider::is_available(&cuda)? {
        //     anyhow::bail!("Please compile ONNX Runtime with CUDA!")
        // }

        // ort::ExecutionProvider::register(&cuda, &builder).map_err(|v| {
        //     anyhow::anyhow!("Please check if ONNX Runtime is compiled with CUDA support: {v}")
        // })?;
        // println!("CUDA:{:?} DirectML:{:?}", encoder_cuda.is_available().unwrap(), decoder_dm.is_available().unwrap());
        let encoder_session = encoder_builder
            .with_execution_providers(
                [
                    // encoder_cuda.build(),
                    // dm.build()
                ]
            )?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(8)?
            .with_inter_threads(8)?
            .commit_from_memory(ENCODER_BYTES)?;
        let decoder_session = decoder_builder
            .with_execution_providers(
                [
                    // decoder_cuda.build(),
                    // decoder_dm.build(),
                    // dm.build()
                ]
            )?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            // .with_parallel_execution(true)?
            .with_intra_threads(12)?
            .with_inter_threads(12)?
            .commit_from_memory(DECODER_BYTES)?;
        Ok(MixTexOnnx {
            encoder_session,
            decoder_session,
            tokenizer: Tokenizer::from_str(TOKENIZER_STR).expect("Fail to load tokenizer"),
        })
    }
    pub fn inference(&self, img: &[f32]) -> std::result::Result<String, Box<dyn std::error::Error>> {
        // eprintln!("Start inference!");
        let start = std::time::Instant::now();

        let encoder_result = self.encoder_session.run(ort::inputs! {"pixel_values" => ([1,3,448,448],img)}?)?;
        let hidden_state = encoder_result["last_hidden_state"].try_extract_tensor::<f32>()?;
        let mut decode_input_ids = array![[0,0,30000_i64]];
        let mut result_idx = [0_u32; MAX_LENGTH];
        let fill_tensor = Array::<f32, _>::zeros((1, 12, 1, 64).f());
        // (1, 2, 3).f();

        // eprintln!("Encode end, start decoder loop");

        let check_rate = MAX_LENGTH / 64;

        for i in 0..MAX_LENGTH {
            // let start_loop = std::time::Instant::now();

            let decoder_result = self.decoder_session.run(ort::inputs! {
            "encoder_hidden_states" => hidden_state.view(),
            "input_ids"=> decode_input_ids.view(),
            "use_cache_branch"=>array![false],
            "past_key_values.0.key"=>fill_tensor.view(),
            "past_key_values.0.value"=>fill_tensor.view(),
            "past_key_values.1.key"=>fill_tensor.view(),
            "past_key_values.1.value"=>fill_tensor.view(),
            "past_key_values.2.key"=>fill_tensor.view(),
            "past_key_values.2.value"=>fill_tensor.view(),
            }?)?;
            // println!("---->loop {i} {:?} ",start_loop.elapsed());
            let logits = decoder_result["logits"].try_extract_tensor::<f32>()?;
            let (next_token_id, _): (usize, _) = logits.slice(s![0,-1,..])
                .iter()
                .enumerate()
                .max_by(|&(_, x), &(_, y)| {
                    x.partial_cmp(&y).unwrap()
                })
                .unwrap();
            result_idx[i] = next_token_id as u32;
            if next_token_id == 30000 {
                break;
            }
            decode_input_ids = concatenate![Axis(1),decode_input_ids,array![[next_token_id as i64]]];
            if ((i + 1) % check_rate == 0) && check_repeat(&result_idx[..=i]) {
                break
            }
        }
        eprintln!("\x1b[31mTime cost:\x1b[32m{:?}\x1b[0m", start.elapsed());

        Ok(self.tokenizer.decode(&result_idx, true).unwrap())
    }
}