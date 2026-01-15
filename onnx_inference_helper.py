"""onnx_inference_helper.py

Python helper module for ONNX inference from MATLAB.

Functions:
    - predict_hcom: Run ResidualMLP ONNX model
    - predict_transformer_uncertainty: Run Transformer model with uncertainty output

Usage from MATLAB:
    pymod = py.importlib.import_module('onnx_inference_helper');
    y = pymod.predict_hcom(x_numpy, 'model.onnx');
"""

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("Please install onnxruntime: pip install onnxruntime")


def predict_hcom(x: np.ndarray, onnx_path: str) -> list:
    """
    Run ResidualMLP ONNX model inference.
    
    Args:
        x: Input features as numpy array, shape (1, in_dim) or (in_dim,)
        onnx_path: Path to ONNX model file
    
    Returns:
        List containing h8_pred as numpy array, shape (8,)
    """
    # Ensure 2D input
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    # Ensure float32
    x = x.astype(np.float32)
    
    # Create ONNX runtime session
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # Get input name
    input_name = sess.get_inputs()[0].name
    
    # Run inference
    outputs = sess.run(None, {input_name: x})
    
    # Return as list for MATLAB compatibility
    h8_pred = outputs[0].flatten()
    return h8_pred.tolist()


def predict_transformer_uncertainty(x: np.ndarray, onnx_path: str) -> tuple:
    """
    Run Transformer Uncertainty ONNX model inference.
    
    Args:
        x: Input features as numpy array, shape (1, in_dim) or (in_dim,)
        onnx_path: Path to ONNX model file
    
    Returns:
        Tuple of (h8_pred, variance), each as list of floats
    """
    # Ensure 2D input
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    # Ensure float32
    x = x.astype(np.float32)
    
    # Create ONNX runtime session
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # Get input name
    input_name = sess.get_inputs()[0].name
    
    # Run inference
    outputs = sess.run(None, {input_name: x})
    
    # Return as tuple of lists for MATLAB compatibility
    h8_pred = outputs[0].flatten().tolist()
    variance = outputs[1].flatten().tolist()
    
    return (h8_pred, variance)


def batch_predict_hcom(x: np.ndarray, onnx_path: str) -> np.ndarray:
    """
    Batch inference for ResidualMLP.
    
    Args:
        x: Input features, shape (batch_size, in_dim)
        onnx_path: Path to ONNX model
    
    Returns:
        h8_pred, shape (batch_size, 8)
    """
    x = x.astype(np.float32)
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: x})
    return outputs[0]


def batch_predict_transformer_uncertainty(x: np.ndarray, onnx_path: str) -> tuple:
    """
    Batch inference for Transformer with uncertainty.
    
    Args:
        x: Input features, shape (batch_size, in_dim)
        onnx_path: Path to ONNX model
    
    Returns:
        Tuple of (h8_pred, variance), each shape (batch_size, 8)
    """
    x = x.astype(np.float32)
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: x})
    return (outputs[0], outputs[1])


if __name__ == "__main__":
    # Quick test
    print("ONNX Inference Helper loaded successfully")
    print(f"onnxruntime version: {ort.__version__}")
