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


def predict_v7_hybrid(x_spectral: np.ndarray, x_2d: np.ndarray, h_rls: np.ndarray, onnx_path: str) -> list:
    """
    Run V7 Hybrid ONNX model inference.
    
    Args:
        x_spectral: Spectral features, shape (1, 210)
        x_2d: 2D E matrix, shape (1, 2, 64, 64)
        h_rls: Ridge LS estimate, shape (1, 8)
        onnx_path: Path to ONNX model file
    
    Returns:
        List containing h8_pred as numpy array, shape (8,)
    """
    # Ensure 2D inputs
    if x_spectral.ndim == 1:
        x_spectral = x_spectral.reshape(1, -1)
    if h_rls.ndim == 1:
        h_rls = h_rls.reshape(1, -1)
    if x_2d.ndim == 3:
        x_2d = x_2d.reshape(1, x_2d.shape[0], x_2d.shape[1], x_2d.shape[2])
    
    # Ensure float32
    x_spectral = x_spectral.astype(np.float32)
    x_2d = x_2d.astype(np.float32)
    h_rls = h_rls.astype(np.float32)
    
    # Create ONNX runtime session
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # Get input names
    input_names = [inp.name for inp in sess.get_inputs()]
    
    # Run inference
    outputs = sess.run(None, {
        input_names[0]: x_spectral,
        input_names[1]: x_2d,
        input_names[2]: h_rls
    })
    
    # Return as list for MATLAB compatibility
    h8_pred = outputs[0].flatten()
    return h8_pred.tolist()
