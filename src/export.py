"""
ONNX export functionality for medical segmentation models.
Supports dynamic input shapes for deployment on mobile/edge devices.
"""

from typing import Any, cast

import torch
import torch.onnx

# Optional dependencies with graceful fallback
try:
    import onnx
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    onnx = cast(Any, None)
    ort = cast(Any, None)
    ONNX_AVAILABLE = False


def export_to_onnx(
    model,
    export_path,
    in_channels,
    input_shape,
    input_shape_override=None,
    dynamic_axes=True,
    opset_version=11,
):
    """
    Export PyTorch model to ONNX format.

    Args:
        model: Trained PyTorch model
        export_path: Path to save ONNX model
        in_channels: Number of input channels
        input_shape: Spatial dimensions tuple (D, H, W) or (H, W)
        input_shape_override: Input tensor shape override (B, C, *spatial_dims)
        dynamic_axes: Whether to enable dynamic batch/spatial dimensions
        opset_version: ONNX opset version for compatibility

    Returns:
        bool: True if export successful, False if ONNX not available
    """
    if not ONNX_AVAILABLE:
        print("ERROR: ONNX export not available. Install dependencies:")
        print("uv sync --extra onnx")
        return False

    model.eval()

    # Use provided input shape if not overridden
    if input_shape_override is None:
        input_shape_override = (1, in_channels) + input_shape

    # Create dummy input tensor
    dummy_input = torch.randn(input_shape_override)

    # Define dynamic axes for flexible input sizes
    dynamic_axes_dict = None
    if dynamic_axes:
        spatial_dims = len(input_shape)
        dynamic_axes_dict = {
            "input": {0: "batch_size"},  # Dynamic batch
            "output": {0: "batch_size"},  # Dynamic batch
        }

        # Add dynamic spatial dimensions
        for i in range(spatial_dims):
            dim_idx = i + 2  # Skip batch and channel dimensions
            dim_name = f"dim_{i}"
            dynamic_axes_dict["input"][dim_idx] = dim_name
            dynamic_axes_dict["output"][dim_idx] = dim_name

    # Export model
    torch.onnx.export(
        model,
        (dummy_input,),
        export_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes_dict,
        verbose=False,
    )

    print(f"Model exported to: {export_path}")

    # Verify export
    if verify_onnx_export(export_path, dummy_input):
        print("ONNX export completed successfully")
        return True
    else:
        return False


def verify_onnx_export(onnx_path, test_input):
    """
    Verify ONNX model can be loaded and produces valid output.

    Args:
        onnx_path: Path to ONNX model
        test_input: Test input tensor

    Returns:
        bool: True if verification successful, False otherwise
    """
    if not ONNX_AVAILABLE:
        print("Warning: Cannot verify ONNX export - ONNX not available")
        return False

    try:
        # Load and check ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        # Test inference
        ort_session = ort.InferenceSession(onnx_path)
        input_name = ort_session.get_inputs()[0].name

        ort_inputs = {input_name: test_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)

        output_shape = getattr(ort_outputs[0], "shape", "unknown")
        print(f"ONNX export verified. Output shape: {output_shape}")
        return True

    except Exception as e:
        print(f"Warning: ONNX export verification failed: {e}")
        return False


def get_model_info(onnx_path):
    """
    Get information about exported ONNX model.

    Args:
        onnx_path: Path to ONNX model

    Returns:
        Dictionary with model information, None if ONNX not available
    """
    if not ONNX_AVAILABLE:
        print("Error: Cannot get model info - ONNX not available")
        return None

    try:
        model = onnx.load(onnx_path)

        # Get input/output info
        input_info = []
        for input_tensor in model.graph.input:
            shape = [
                dim.dim_value if dim.dim_value > 0 else "dynamic"
                for dim in input_tensor.type.tensor_type.shape.dim
            ]
            input_info.append(
                {
                    "name": input_tensor.name,
                    "shape": shape,
                    "type": input_tensor.type.tensor_type.elem_type,
                }
            )

        output_info = []
        for output_tensor in model.graph.output:
            shape = [
                dim.dim_value if dim.dim_value > 0 else "dynamic"
                for dim in output_tensor.type.tensor_type.shape.dim
            ]
            output_info.append(
                {
                    "name": output_tensor.name,
                    "shape": shape,
                    "type": output_tensor.type.tensor_type.elem_type,
                }
            )

        # Calculate model size
        model_size_mb = len(model.SerializeToString()) / (1024 * 1024)

        return {
            "inputs": input_info,
            "outputs": output_info,
            "size_mb": round(model_size_mb, 2),
            "opset_version": model.opset_import[0].version,
        }

    except Exception as e:
        print(f"Error loading ONNX model info: {e}")
        return None
