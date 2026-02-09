import pytest
import torch
import numpy as np
from src.torch_train.cnn.utils.yolo_utils import yolo_non_max_suppression, iou

class TestYoloNonMaxSuppression:
    """Test suite for yolo_non_max_suppression function"""
    
    def test_no_boxes(self):
        """Test 1: No boxes - should not throw errors"""
        scores = torch.tensor([])
        boxes = torch.tensor([]).view(0, 4)
        classes = torch.tensor([])
        
        result_scores, result_boxes, result_classes = yolo_non_max_suppression(
            scores, boxes, classes, max_boxes=10, iou_threshold=0.5
        )
        
        # Should return empty tensors
        assert result_scores.shape[0] == 0
        assert result_boxes.shape[0] == 0
        assert result_boxes.shape[1] == 4
        assert result_classes.shape[0] == 0
    
    def test_two_overlapping_boxes(self):
        """Test 2: Two overlapping boxes - should keep highest score"""
        scores = torch.tensor([0.9, 0.7])
        boxes = torch.tensor([
            [10, 10, 50, 50],   # Box 1 (higher score)
            [15, 15, 55, 55],   # Box 2 (overlaps, lower score)
        ], dtype=torch.float32)
        classes = torch.tensor([0, 0])  # Same class
        
        result_scores, result_boxes, result_classes = yolo_non_max_suppression(
            scores, boxes, classes, max_boxes=10, iou_threshold=0.5
        )
        
        # Should keep only the highest score box
        assert len(result_scores) == 1
        assert result_scores[0] == 0.9
        assert torch.equal(result_boxes[0], torch.tensor([10, 10, 50, 50], dtype=torch.float32))
        assert result_classes[0] == 0
    
    def test_three_overlapping_boxes(self):
        """Test 3: Three overlapping boxes - should keep highest score only"""
        scores = torch.tensor([0.9, 0.7, 0.8])
        boxes = torch.tensor([
            [10, 10, 50, 50],   # Box 1 (highest score)
            [15, 15, 55, 55],   # Box 2 (overlaps, lowest score)
            [12, 12, 52, 52],   # Box 3 (overlaps, medium score)
        ], dtype=torch.float32)
        classes = torch.tensor([0, 0, 0])  # Same class
        
        result_scores, result_boxes, result_classes = yolo_non_max_suppression(
            scores, boxes, classes, max_boxes=10, iou_threshold=0.5
        )
        
        # Should keep only the highest score box
        assert len(result_scores) == 1
        assert result_scores[0] == 0.9
        assert torch.equal(result_boxes[0], torch.tensor([10, 10, 50, 50], dtype=torch.float32))
        assert result_classes[0] == 0
    
    def test_two_non_overlapping_boxes(self):
        """Test 4: Two non-overlapping boxes - should keep both"""
        scores = torch.tensor([0.9, 0.8])
        boxes = torch.tensor([
            [10, 10, 50, 50],   # Box 1
            [100, 100, 150, 150],  # Box 2 (no overlap)
        ], dtype=torch.float32)
        classes = torch.tensor([0, 0])  # Same class
        
        result_scores, result_boxes, result_classes = yolo_non_max_suppression(
            scores, boxes, classes, max_boxes=10, iou_threshold=0.5
        )
        
        # Should keep both boxes
        assert len(result_scores) == 2
        assert torch.allclose(result_scores, torch.tensor([0.9, 0.8]))
        
        # Should be sorted by score (descending)
        assert torch.equal(result_boxes[0], torch.tensor([10, 10, 50, 50], dtype=torch.float32))
        assert torch.equal(result_boxes[1], torch.tensor([100, 100, 150, 150], dtype=torch.float32))
    
    def test_multiple_classes(self):
        """Test 5: Multiple classes - should process each class separately"""
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
        boxes = torch.tensor([
            [10, 10, 50, 50],   # Class 0 (high score)
            [15, 15, 55, 55],   # Class 0 (overlaps with box 1)
            [100, 100, 150, 150],  # Class 1
            [200, 200, 250, 250],  # Class 1
        ], dtype=torch.float32)
        classes = torch.tensor([0, 0, 1, 1])
        
        result_scores, result_boxes, result_classes = yolo_non_max_suppression(
            scores, boxes, classes, max_boxes=10, iou_threshold=0.5
        )
        
        # Should keep 3 boxes (1 from class 0, 2 from class 1)
        assert len(result_scores) == 3
        
        # Highest score should be first (class 0, box 1)
        assert result_scores[0] == 0.9
        assert result_classes[0] == 0
    
    def test_max_boxes_limit(self):
        """Test 6: Max boxes limit - should return only max_boxes"""
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
        boxes = torch.tensor([
            [10, 10, 50, 50],   # Box 1
            [100, 100, 150, 150],  # Box 2
            [200, 200, 250, 250],  # Box 3
            [300, 300, 350, 350],  # Box 4
            [400, 400, 450, 450],  # Box 5
        ], dtype=torch.float32)
        classes = torch.tensor([0, 1, 2, 3, 4])  # Different classes
        
        result_scores, result_boxes, result_classes = yolo_non_max_suppression(
            scores, boxes, classes, max_boxes=3, iou_threshold=0.5
        )
        
        # Should return only 3 boxes (max_boxes limit)
        assert len(result_scores) == 3
        assert len(result_boxes) == 3
        assert len(result_classes) == 3
    
    def test_iou_threshold_filtering(self):
        """Test 7: IoU threshold - should filter overlapping boxes above threshold"""
        scores = torch.tensor([0.9, 0.8])
        boxes = torch.tensor([
            [10, 10, 50, 50],   # Box 1
            [12, 12, 52, 52],   # Box 2 (high overlap, IoU > 0.5)
        ], dtype=torch.float32)
        classes = torch.tensor([0, 0])
        
        # Test with strict threshold
        result_scores, result_boxes, result_classes = yolo_non_max_suppression(
            scores, boxes, classes, max_boxes=10, iou_threshold=0.3  # Low threshold
        )
        
        # Should keep only one box (high IoU > 0.3)
        assert len(result_scores) == 1
        assert result_scores[0] == 0.9


class TestIoU:
    """Test suite for IoU function"""
    
    def test_iou_identical_boxes(self):
        """Test IoU of identical boxes"""
        box1 = [10, 10, 50, 50]
        box2 = [10, 10, 50, 50]
        
        result = iou(box1, box2)
        assert result == 1.0  # Perfect overlap
    
    def test_iou_no_overlap(self):
        """Test IoU of non-overlapping boxes"""
        box1 = [10, 10, 50, 50]
        box2 = [100, 100, 150, 150]
        
        result = iou(box1, box2)
        assert result == 0.0  # No overlap
    
    def test_iou_partial_overlap(self):
        """Test IoU of partially overlapping boxes"""
        box1 = [10, 10, 50, 50]  # Area: 40*40 = 1600
        box2 = [30, 30, 70, 70]  # Area: 40*40 = 1600
        
        result = iou(box1, box2)
        # Intersection: 20*20 = 400
        # Union: 1600 + 1600 - 400 = 2800
        # IoU: 400/2800 = 0.142857...
        assert abs(result - 0.142857) < 0.001


if __name__ == "__main__":
    # Run tests manually
    test_suite = TestYoloNonMaxSuppression()
    
    print("Running yolo_non_max_suppression tests...")
    
    try:
        test_suite.test_no_boxes()
        print("✅ Test 1: No boxes - PASSED")
    except Exception as e:
        print(f"❌ Test 1: No boxes - FAILED: {e}")
    
    try:
        test_suite.test_two_overlapping_boxes()
        print("✅ Test 2: Two overlapping boxes - PASSED")
    except Exception as e:
        print(f"❌ Test 2: Two overlapping boxes - FAILED: {e}")
    
    try:
        test_suite.test_three_overlapping_boxes()
        print("✅ Test 3: Three overlapping boxes - PASSED")
    except Exception as e:
        print(f"❌ Test 3: Three overlapping boxes - FAILED: {e}")
    
    try:
        test_suite.test_two_non_overlapping_boxes()
        print("✅ Test 4: Two non-overlapping boxes - PASSED")
    except Exception as e:
        print(f"❌ Test 4: Two non-overlapping boxes - FAILED: {e}")
    
    try:
        test_suite.test_multiple_classes()
        print("✅ Test 5: Multiple classes - PASSED")
    except Exception as e:
        print(f"❌ Test 5: Multiple classes - FAILED: {e}")
    
    try:
        test_suite.test_max_boxes_limit()
        print("✅ Test 6: Max boxes limit - PASSED")
    except Exception as e:
        print(f"❌ Test 6: Max boxes limit - FAILED: {e}")
    
    try:
        test_suite.test_iou_threshold_filtering()
        print("✅ Test 7: IoU threshold filtering - PASSED")
    except Exception as e:
        print(f"❌ Test 7: IoU threshold filtering - FAILED: {e}")
    
    print("\nRunning IoU tests...")
    
    iou_suite = TestIoU()
    
    try:
        iou_suite.test_iou_identical_boxes()
        print("✅ IoU Test 1: Identical boxes - PASSED")
    except Exception as e:
        print(f"❌ IoU Test 1: Identical boxes - FAILED: {e}")
    
    try:
        iou_suite.test_iou_no_overlap()
        print("✅ IoU Test 2: No overlap - PASSED")
    except Exception as e:
        print(f"❌ IoU Test 2: No overlap - FAILED: {e}")
    
    try:
        iou_suite.test_iou_partial_overlap()
        print("✅ IoU Test 3: Partial overlap - PASSED")
    except Exception as e:
        print(f"❌ IoU Test 3: Partial overlap - FAILED: {e}")
    
    print("\nAll tests completed!")