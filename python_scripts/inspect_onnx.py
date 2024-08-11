import time

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

feature_extractor = AutoImageProcessor.from_pretrained("../onnx")
tokenizer = AutoTokenizer.from_pretrained("../onnx")
image = Image.open("../test.png").convert("RGB")
inputs: np.ndarray = feature_extractor(image, return_tensors="np").pixel_values

# 加载ONNX模型
model_path = r"E:\WorkSpace\RustProjects\MixTex-rs\onnx\decoder_model_merged.onnx"  # 替换为你的模型路径
# model = onnx.load(model_path)
#
# # 打印模型的基本信息
# print("模型的输入信息:")
# for input_tensor in model.graph.input:
#     print(f"Name: {input_tensor.name}")
#     print(f"Shape: {[dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]}")
#     print(f"Type: {input_tensor.type.tensor_type.elem_type}")
#     print()
#
# print("模型的输出信息:")
# for output_tensor in model.graph.output:
#     print(f"Name: {output_tensor.name}")
#     print(f"Shape: {[dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]}")
#     print(f"Type: {output_tensor.type.tensor_type.elem_type}")
#     print()

# 使用 onnxruntime 运行模型并显示更详细的信息
encoder_session = ort.InferenceSession(
    r"E:\WorkSpace\RustProjects\MixTex-rs\onnx\encoder_model.onnx",
    providers=["CPUExecutionProvider"],
)
print(encoder_session.get_providers())
decoder_session = ort.InferenceSession(
    r"E:\WorkSpace\RustProjects\MixTex-rs\onnx\decoder_model_merged.onnx",
    providers=['DmlExecutionProvider'],
)
print(decoder_session.get_providers())

# print("ONNX模型输入:")
# for input_tensor in session.get_inputs():
#     print(f"Name: {input_tensor.name}")
#     print(f"Shape: {input_tensor.shape}")
#     print(f"Type: {input_tensor.type}")
#     print()
#
# print("ONNX模型输出:")
# for output_tensor in session.get_outputs():
#     print(f"Name: {output_tensor.name}")
#     print(f"Shape: {output_tensor.shape}")
#     print(f"Type: {output_tensor.type}")
#     print()

# print(inputs.shape, inputs.dtype)
# print(inputs.flatten())
# json.dump(inputs.flatten().tolist(), open("../image.json", "w"))
#
# res = session.run(None, {"pixel_values": inputs})[0]
# print("result:", res.shape, res.flatten())

if __name__ == '__main__':
    start = time.perf_counter()
    max_length = 512
    generated_text = ""

    encoder_outputs = encoder_session.run(None, {"pixel_values": inputs})[0]
    print(f"Encoder Time cost: {time.perf_counter() - start:.6f}s")

    decoder_input_ids = tokenizer("<s>", return_tensors="np").input_ids.astype(np.int64)
    print(decoder_input_ids)
    for i in range(max_length):
        start_loop = time.perf_counter()
        decoder_outputs = decoder_session.run(None, {
            "input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_outputs,
            "use_cache_branch": np.array([False]),
            'past_key_values.0.key': np.zeros((1, 12, 224, 64), dtype=np.float32),
            'past_key_values.0.value': np.zeros((1, 12, 224, 64), dtype=np.float32),
            'past_key_values.1.key': np.zeros((1, 12, 224, 64), dtype=np.float32),
            'past_key_values.1.value': np.zeros((1, 12, 224, 64), dtype=np.float32),
            'past_key_values.2.key': np.zeros((1, 12, 224, 64), dtype=np.float32),
            'past_key_values.2.value': np.zeros((1, 12, 224, 64), dtype=np.float32)
        })[0]
        # print(f"loop {i} Inference time cost:{time.perf_counter() - start_loop:.6f}s")
        # print("decoder_outputs", decoder_outputs.shape)
        next_token_id = np.argmax(decoder_outputs[:, -1, :], axis=-1)
        # print("next_token_id", next_token_id.shape)
        decoder_input_ids = np.concatenate([decoder_input_ids, next_token_id[:, None]], axis=-1)
        # print("decoder_input_ids", decoder_input_ids.shape)
        generated_text += tokenizer.decode(next_token_id, skip_special_tokens=True)
        # print(generated_text)
        # self.log(tokenizer.decode(next_token_id, skip_special_tokens=True), end="")
        # if self.check_repetition(generated_text, 12):
        #     self.log('\n===?!重复重复重复!?===\n')
        #     break
        if next_token_id == tokenizer.eos_token_id:
            print('\n===成功复制到剪切板===\n')
            break
        # print(f"Decoder loop {i} Time cost: {time.perf_counter() - start_loop:.6f}s")

    print(generated_text)
    print(f"Time cost: {time.perf_counter() - start:.6f}s")
