#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbImage, ImageReader};
    use MixTex_rs::onnx::MixTexOnnx;
    use MixTex_rs::vit_image_processor::{preprocess, rescale_and_normalize, resize};
    use MixTex_rs::winml::{WinMLDeviceType, MixTexWinML};

    #[test]
    fn test_load_model() {
        let model_cpu = MixTexWinML::build( None);
        let model_dm = MixTexWinML::build(Some(WinMLDeviceType::DirectML));
    }
    #[test]
    fn test_raw_inference() {
        let model = MixTexWinML::build(None);
        let img = preprocess("test.png");
        eprintln!("{}", model.inference(&img).unwrap());
    }
    #[test]
    fn test_batch_inference() {
        let model = MixTexWinML::build(None);
        for _ in 0..10 {
            let img = preprocess("test.png");
            eprintln!("{}", model.inference(&img).unwrap());
        }
    }

    #[test]
    fn test_onnx_inference() {
        let model = MixTexOnnx::build().unwrap();
        let img = preprocess("test.png");
        eprintln!("{}", model.inference(&img).unwrap());
    }
}