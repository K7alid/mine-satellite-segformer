import torch
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import numpy as np
from typing import Dict, Tuple
from utils import *


class Segformer:
    def __init__(self, model_name: str = 'satellite_mine_model_best', device: str = 'cuda'):
        self.device = device
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name).to(device)
        self.model.eval()
    def predict_mask(self, image):
      """Predict mask for a single image chunk"""
      inputs = self.processor(image, return_tensors="pt")
      inputs = {k: v.to(self.device) for k, v in inputs.items()}

      with torch.no_grad():
          outputs = self.model(**inputs)
          logits = outputs.logits
          logits = torch.nn.functional.interpolate(
              logits,
              size=image.size[::-1],
              mode="bilinear",
              align_corners=False,
          )
          pred_mask = logits.argmax(dim=1)[0].cpu().numpy()

      return pred_mask


class SegformerManager:
    def __init__(self, segformer: Segformer):
        self.segformer = segformer

    def process_large_image(self, image):
      """Process large image by splitting into optimal chunks"""
      # Split image into optimal chunks
      chunks, positions = split_image_dynamic(image)

      # Create empty mask with original image dimensions
      final_mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)

      # Process each chunk
      for chunk, (left, top, right, bottom) in zip(chunks, positions):
          # Predict mask for chunk
          chunk_mask = self.segformer.predict_mask(chunk)

          # Place chunk mask in final mask
          final_mask[top:bottom, left:right] = chunk_mask

      return final_mask



class SegmentationMetrics:
  def __init__(self, threshold: float = 0.5):
      self.threshold = threshold

  def _prepare_inputs(self, pred: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
      # Ensure binary values
      pred = (pred > self.threshold).astype(np.int32)
      target = (target > 0).astype(np.int32)
      return pred, target

  def compute_iou(self, pred: np.ndarray, target: np.ndarray) -> float:
      """Calculate IoU with zero division handling"""
      pred, target = self._prepare_inputs(pred, target)
      intersection = np.logical_and(pred, target).sum()
      union = np.logical_or(pred, target).sum()

      # Handle the case where both pred and target are empty
      if union == 0:
          return 1.0 if intersection == 0 else 0.0
      return float(intersection / union)

  def compute_precision(self, pred: np.ndarray, target: np.ndarray) -> float:
      """Calculate Precision with zero division handling"""
      pred, target = self._prepare_inputs(pred, target)
      true_positive = np.logical_and(pred, target).sum()
      predicted_positive = pred.sum()

      # Handle the case where no positive predictions
      if predicted_positive == 0:
          return 1.0 if true_positive == 0 else 0.0
      return float(true_positive / predicted_positive)

  def compute_recall(self, pred: np.ndarray, target: np.ndarray) -> float:
      """Calculate Recall with zero division handling"""
      pred, target = self._prepare_inputs(pred, target)
      true_positive = np.logical_and(pred, target).sum()
      actual_positive = target.sum()

      # Handle the case where no actual positives
      if actual_positive == 0:
          return 1.0 if true_positive == 0 else 0.0
      return float(true_positive / actual_positive)

  def compute_f1_score(self, precision: float, recall: float) -> float:
      """Calculate F1-score with zero division handling"""
      if precision + recall == 0:
          return 0.0
      return 2 * (precision * recall) / (precision + recall)

  def compute_accuracy(self, pred: np.ndarray, target: np.ndarray) -> float:
      """Calculate Accuracy"""
      pred, target = self._prepare_inputs(pred, target)
      total_pixels = target.size
      if total_pixels == 0:
          return 0.0
      return float(np.sum(pred == target) / total_pixels)

  def compute_all_metrics(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
      """Compute all metrics at once"""
      precision = self.compute_precision(pred, target)
      recall = self.compute_recall(pred, target)

      return {
          'iou': self.compute_iou(pred, target),
          'precision': precision,
          'recall': recall,
          'f1_score': self.compute_f1_score(precision, recall),
          'accuracy': self.compute_accuracy(pred, target)
      }


# Example usage for testing
if __name__ == '__main__':
    segformer = Segformer()
    mask = segformer.predict_mask(Image.open('segformer_model/examples/AREA_A_2023-12-01_1_21NTF_1_0_visual.png'))
    print(mask.shape)
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)  # Convert to 8-bit grayscale
    mask_image.save("predicted_mask.png")