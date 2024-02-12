import onnxruntime

# Load the ONNX model
onnx_session = onnxruntime.InferenceSession("yolov8x.onnx")

# Prepare input data
input_name = onnx_session.get_inputs()[0].name
input_data = np.random.randn(1, 3, height, width).astype(np.float32)  # Use random input data for testing

# Measure inference time for ONNX model
num_iterations = 100  # You can adjust the number of iterations for more accurate measurement
onnx_total_time = 0
for _ in range(num_iterations):
    start_time = time.time()
    output = onnx_session.run(None, {input_name: input_data})
    end_time = time.time()
    onnx_total_time += end_time - start_time

onnx_average_time = onnx_total_time / num_iterations
print("Average inference time for ONNX model:", onnx_average_time, "seconds")

# Now, you can also measure the inference time for the original PyTorch model using similar code
# Compare the average inference times to evaluate the performance difference
