param(
  [Parameter(Mandatory = $true)][string]$InputOnnx,
  [Parameter(Mandatory = $true)][string]$OutputOnnx
)

# TensorRT-RTX is strongly typed.  Do not use a builder kFP16 flag; create a
# typed ONNX graph first, then parse it with compile-trt-pipeline.  This uses
# NVIDIA TensorRT Model Optimizer when it is installed in the project venv.
rtk .venv/Scripts/python.exe -m modelopt.onnx.autocast `
  --onnx_path $InputOnnx `
  --output_path $OutputOnnx
