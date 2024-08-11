#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbImage, ImageReader};
    use MixTex_rs::vit_image_processor::{preprocess, rescale_and_normalize, resize};

    #[test]
    fn test_read_image() {
        let img = ImageReader::open("test.png").unwrap().decode().unwrap();
        println!("{:?} {:?} {:?}", img.color(), img.height(), img.width())
    }

    #[test]
    fn test_resize_image() {
        let img = ImageReader::open("test.png").unwrap().decode().unwrap();
        println!("{:?} {:?} {:?}", img.color(), img.height(), img.width());
        let img = resize(img);
        println!("{:?} {:?} {:?}", img.color(), img.height(), img.width());
    }
    #[test]
    fn test_rescale_and_normalize() {
        let img = ImageReader::open("test.png").unwrap().decode().unwrap();
        println!("{:?} {:?} {:?}", img.color(), img.height(), img.width());
        let img = resize(img);
        println!("{:?} {:?} {:?}", img.color(), img.height(), img.width());
        let img = rescale_and_normalize(img);
        println!("{:?} {:?} {:?}", img.color(), img.height(), img.width());
    }

    #[test]
    fn test_result() {
        let img_buf = preprocess("test.png");
        println!("{}  {}"
                 , include_str!("../onnx/image.json")
                     .split(",").count()
                 , img_buf.len());
        // println!("{:#?}", img_buf);
        include_str!("../onnx/image.json")
            .split(",")
            .enumerate()
            .for_each(|(i, x)| {
                let x = x.trim().parse::<f32>().unwrap();
                if (img_buf[i] - x).abs() > 0.02_f32 {
                    println!("{i} {} {}", img_buf[i], x);
                }
            });
        // let img = rescale_and_normalize(resize(
        //     ImageReader::open("test.png")
        //         .unwrap()
        //         .decode()
        //         .unwrap()
        // ));
        // slow in stdout capture
        // println!("{:#?}",img.as_rgb32f().unwrap().clone().into_vec())
    }
    #[test]
    fn test_precision() {
        // let img_buf = preprocess("test.png");
        let img = rescale_and_normalize(resize(
            ImageReader::open("test.png")
                .unwrap()
                .decode()
                .unwrap()
        ));
        println!("{:?}", img.into_rgb32f().get_pixel(224, 224));
    }


    // #[test]
    // fn test_rescale_image() {
    //     let img = DynamicImage::ImageRgb8(RgbImage::new(2, 2));
    //     let rescaled_img = rescale_image(&img);
    //
    //     // Check that each pixel has been rescaled
    //     for pixel in rescaled_img.pixels() {
    //         assert_eq!(pixel[0], 0);
    //         assert_eq!(pixel[1], 0);
    //         assert_eq!(pixel[2], 0);
    //     }
    // }
    //
    // #[test]
    // fn test_resize_image() {
    //     let img = DynamicImage::ImageRgb8(RgbImage::new(800, 600));
    //     let resized_img = img.resize_exact(448, 448, FilterType::CatmullRom);
    //
    //     assert_eq!(resized_img.dimensions(), (448, 448));
    // }
    //
    // #[test]
    // fn test_process_image() {
    //     // Create a small test image
    //     let img = DynamicImage::ImageRgb8(RgbImage::new(800, 600));
    //     img.save("test_image.png").unwrap();
    //
    //     let processed_image = process_image("test_image.png");
    //
    //     // Check that the processed image has the correct dimensions
    //     assert_eq!(processed_image.dimensions(), (448, 448));
    //
    //     // Clean up the test image
    //     std::fs::remove_file("test_image.png").unwrap();
    // }
}
