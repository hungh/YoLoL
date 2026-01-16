import torch
import os
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, Union
from src.load.read_produce_dataset import process_dataset
from src.load.read_produce_dataset import load_yaml_classes
from src.train.mini_batch_train import DeepNet
from src.load.read_produce_dataset import scale_data

# Constants
MODEL_DIR = Path(__file__).parents[2] / 'saved_models'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path: Union[str, Path]) -> Tuple[torch.nn.Module, int, Optional[float]]:
    """Load model from checkpoint with error handling."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model = DeepNet(
            input_size=checkpoint.get('input_size', 12288),
            num_classes=checkpoint.get('num_classes', 63)
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return (
            model,
            checkpoint.get('epoch', 0),
            checkpoint.get('loss', None)
        )
    except Exception as e:
        raise RuntimeError(f"Error loading model from {model_path}: {str(e)}")

def predict(
    model: torch.nn.Module, 
    input_tensor: torch.Tensor,
    threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Make predictions with proper batching and memory management."""
    model.eval()
    with torch.no_grad():
        if len(input_tensor) > 1000:  # Process in batches if large input
            preds = []
            probs = []
            for i in range(0, len(input_tensor), 1000):
                batch = input_tensor[i:i+1000].to(device)
                out = model(batch)
                prob = torch.sigmoid(out)
                preds.append((prob > threshold).float())
                probs.append(prob)
            return torch.cat(preds), torch.cat(probs)
        else:
            input_tensor = input_tensor.to(device)
            output = model(input_tensor)
            probabilities = torch.sigmoid(output)
            return (probabilities > threshold).float(), probabilities

"""
Print predictions and labels with class names.
Return :
    array of printed strings
"""
def predictions_results(
    predictions: np.ndarray, 
    labels: np.ndarray, 
    class_names: Dict[int, str]
) -> None:
    """Print predictions and labels with class names."""
    results = []
    results.append("\n===== PREDICTIONS =====")
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        pred_classes = [class_names[j] for j, p in enumerate(pred) if p == 1]
        true_classes = [class_names[j] for j, l in enumerate(label) if l == 1]
        results.append(f"\nSample {i+1}:")
        results.append(f"\n  Predicted: {', '.join(pred_classes) or 'None'}")
        results.append(f"\n  Actual:    {', '.join(true_classes) or 'None'}")
        if set(np.where(pred == 1)[0]) == set(np.where(label == 1)[0]):
            results.append("\n  ✓ Correct")
        else:
            results.append("\n  ✗ Incorrect")
    return results

def load_test_data() -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load and preprocess test data consistently with training."""
    project_root = Path(__file__).parents[2]
    data_dir = project_root / "assets/produce_dataset/LVIS_Fruits_And_Vegetables"
    
    # Load data
    data = load_yaml_classes(data_dir / "data.yaml")
    test_images = data_dir / data["test"]
    test_labels = str(test_images).replace("images", "labels")

    X_test, Y_test = process_dataset(test_images, test_labels, target_size=64)
    X_test = scale_data(X_test, method='minmax')  # Same as training
    return X_test.T, Y_test, data  # Transpose to match training format

def evaluate_model(
    model_path: Union[str, Path],
    threshold: float = 0.5
) -> Dict[str, float]:
    """Evaluate model on test set and return metrics."""

    # for prediction output
    out_dir = Path('out')
    out_dir.mkdir(exist_ok=True)
    
    output_file = out_dir / 'evaluation_results.txt'

    # Load data and model
    X_test, Y_test, data = load_test_data()
    model, _, _ = load_model(model_path)
    
    # Convert to tensors
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_true = torch.tensor(Y_test, dtype=torch.float32)
    
    # Get predictions
    y_pred, y_prob = predict(model, X_tensor, threshold)
    
    # Calculate metrics
    accuracy = (y_pred == y_true.to(device)).float().mean().item()
    
     # Prepare detailed results
    results = []
    results.append("===== EVALUATION RESULTS =====")
    results.append(f"Model: {model_path}")
    results.append(f"Accuracy: {accuracy:.4f}")
    results.append(f"Threshold: {threshold}")
    results.append(f"Test samples: {len(X_test)}")
    results.append("\n===== DETAILED PREDICTIONS =====")

    
    results.extend(predictions_results(
        y_pred.cpu().numpy(),
        y_true.numpy(),
        data['names']
    ))
    # save results to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))

    print(f"Results saved to {output_file}")

    return {
        'accuracy': accuracy,
        'predictions': y_pred.cpu().numpy(),
        'probabilities': y_prob.cpu().numpy(),
        'output_file': output_file
    }

if __name__ == "__main__":
    model_path = MODEL_DIR / 'produce_classifier_final.pth'
    evaluate_model(model_path)