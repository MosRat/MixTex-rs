# import cv2
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor

feature_extractor = AutoImageProcessor.from_pretrained("../onnx")
image = Image.open("../test.png").convert("RGB")

image_resize = image.resize((448, 448), resample=Image.BICUBIC)
print(np.asarray(image_resize).shape)

image_rescale = np.asarray(image_resize) * 0.00392156862745098
print(image_rescale)

image_normalize = (image_rescale - 0.5) / 0.5
print(image_normalize)
#
# u8_image = np.asarray(image)
# print(u8_image * 0.00392156862745098)

inputs: np.ndarray = feature_extractor(image, return_tensors="np").pixel_values
fake = inputs.squeeze(0).transpose((1, 2, 0))
print("-----------------------------------")
print(fake)
print("^" * 40)
print(image_normalize)

print(np.allclose(inputs.squeeze(0).transpose((1, 2, 0)), image_normalize))

for i in range(448):
    for j in range(448):
        if not np.allclose(image_normalize[i][j], fake[i][j]):
            print(f"({i},{j}):{image_normalize[i][j]} <=> {fake[i][j]}")
