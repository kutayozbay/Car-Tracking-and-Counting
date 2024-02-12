import torch.onnx
import main
# Export the PyTorch model to ONNX format
input_shape = (1, 3, 3840, 2160)  # Specify the input shape of your model
dummy_input = torch.randn(input_shape)  # Create a dummy input tensor
output_path = "yolov8x.onnx"  # Specify the output path for the ONNX model
print(main.model)
torch.onnx.export(main.model, dummy_input, output_path, opset_version=12)
