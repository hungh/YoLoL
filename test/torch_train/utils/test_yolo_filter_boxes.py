import pytest
import torch
import numpy as np
from src.torch_train.cnn.utils.yolo_utils import yolo_filter_boxes

class TestYoloFilterBoxes:
    """Test suite for yolo_filter_boxes function"""
    
    def test_three_boxes_select_highest_confidence(self):
        """Test 1: Three boxes - should select highest confidence"""
        # Create test data for 3 boxes
        grid_size = 19
        num_anchors = 3
        num_classes = 63
        
        # Boxes: (19, 19, 3, 4) - simplified to just 3 boxes at different positions
        boxes = torch.zeros(grid_size, grid_size, num_anchors, 4)
        boxes[0, 0, 0] = torch.tensor([10, 10, 50, 50])  # Box 1
        boxes[0, 0, 1] = torch.tensor([20, 20, 60, 60])  # Box 2
        boxes[0, 0, 2] = torch.tensor([30, 30, 70, 70])  # Box 3
        
        # Box confidence: (19, 19, 3, 1) - different confidence scores
        box_confidence = torch.zeros(grid_size, grid_size, num_anchors, 1)
        box_confidence[0, 0, 0] = torch.tensor([0.7])  # Medium confidence
        box_confidence[0, 0, 1] = torch.tensor([0.9])  # High confidence (should be selected)
        box_confidence[0, 0, 2] = torch.tensor([0.5])  # Low confidence
        
        # Box class probabilities: (19, 19, 3, 63) - all boxes predict class 0 with high prob
        box_class_probs = torch.zeros(grid_size, grid_size, num_anchors, num_classes)
        box_class_probs[0, 0, 0, 0] = 0.8  # Box 1 predicts class 0
        box_class_probs[0, 0, 1, 0] = 0.9  # Box 2 predicts class 0
        box_class_probs[0, 0, 2, 0] = 0.7  # Box 3 predicts class 0
        
        # Apply filter with default threshold (0.6)
        scores, filtered_boxes, classes = yolo_filter_boxes(
            boxes, box_confidence, box_class_probs, threshold=0.6
        )
        
        # Should select only box 1 (highest confidence: 0.9 * 0.9 = 0.81)
        assert len(scores) == 1
        assert len(filtered_boxes) == 1
        assert len(classes) == 1
        
        # Check dimensions
        assert scores.shape[0] == 1  # (N,)
        assert filtered_boxes.shape == (1, 4)  # (N, 4)
        assert classes.shape[0] == 1  # (N,)
        
        # Check values
        assert scores[0] == pytest.approx(0.81, rel=1e-3)  # 0.9 * 0.9
        assert torch.equal(filtered_boxes[0], torch.tensor([20, 20, 60, 60], dtype=torch.float32))
        assert classes[0] == 0  # Class 0
    
    def test_single_box(self):
        """Test 2: Single box - should return that one"""
        grid_size = 19
        num_anchors = 3
        num_classes = 63
        
        # Single box
        boxes = torch.zeros(grid_size, grid_size, num_anchors, 4)
        boxes[5, 5, 1] = torch.tensor([100, 100, 150, 150])
        
        box_confidence = torch.zeros(grid_size, grid_size, num_anchors, 1)
        box_confidence[5, 5, 1] = torch.tensor([0.8])
        
        box_class_probs = torch.zeros(grid_size, grid_size, num_anchors, num_classes)
        box_class_probs[5, 5, 1, 2] = 0.9  # Predict class 2
        
        scores, filtered_boxes, classes = yolo_filter_boxes(
            boxes, box_confidence, box_class_probs, threshold=0.6
        )
        
        # Should return the single box
        assert len(scores) == 1
        assert len(filtered_boxes) == 1
        assert len(classes) == 1
        
        # Check dimensions
        assert scores.shape[0] == 1
        assert filtered_boxes.shape == (1, 4)
        assert classes.shape[0] == 1
        
        # Check values
        assert scores[0] == pytest.approx(0.72, rel=1e-3)  # 0.8 * 0.9
        assert torch.equal(filtered_boxes[0], torch.tensor([100, 100, 150, 150], dtype=torch.float32))
        assert classes[0] == 2  # Class 2
    
    def test_no_boxes(self):
        """Test 3: No boxes - should not raise error"""
        grid_size = 19
        num_anchors = 3
        num_classes = 63
        
        # All zeros - no boxes
        boxes = torch.zeros(grid_size, grid_size, num_anchors, 4)
        box_confidence = torch.zeros(grid_size, grid_size, num_anchors, 1)
        box_class_probs = torch.zeros(grid_size, grid_size, num_anchors, num_classes)
        
        # Should not raise any error
        scores, filtered_boxes, classes = yolo_filter_boxes(
            boxes, box_confidence, box_class_probs, threshold=0.6
        )
        
        # Should return empty tensors
        assert len(scores) == 0
        assert len(filtered_boxes) == 0
        assert len(classes) == 0
        
        # Check dimensions
        assert scores.shape[0] == 0  # (0,)
        assert filtered_boxes.shape == (0, 4)  # (0, 4)
        assert classes.shape[0] == 0  # (0,)
    
    def test_multiple_boxes_above_threshold(self):
        """Test 4: Multiple boxes above threshold - should return all"""
        grid_size = 19
        num_anchors = 3
        num_classes = 63
        
        # Two boxes with high confidence
        boxes = torch.zeros(grid_size, grid_size, num_anchors, 4)
        boxes[0, 0, 0] = torch.tensor([10, 10, 50, 50])
        boxes[10, 10, 1] = torch.tensor([100, 100, 150, 150])
        
        box_confidence = torch.zeros(grid_size, grid_size, num_anchors, 1)
        box_confidence[0, 0, 0] = torch.tensor([0.9])  # High confidence
        box_confidence[10, 10, 1] = torch.tensor([0.8])  # High confidence
        
        box_class_probs = torch.zeros(grid_size, grid_size, num_anchors, num_classes)
        box_class_probs[0, 0, 0, 1] = 0.9  # Predict class 1
        box_class_probs[10, 10, 1, 3] = 0.8  # Predict class 3
        
        scores, filtered_boxes, classes = yolo_filter_boxes(
            boxes, box_confidence, box_class_probs, threshold=0.5
        )
        
        # Should return both boxes
        assert len(scores) == 2
        assert len(filtered_boxes) == 2
        assert len(classes) == 2
        
        # Check dimensions
        assert scores.shape[0] == 2
        assert filtered_boxes.shape == (2, 4)
        assert classes.shape[0] == 2
        
        # Check that both boxes are present
        expected_boxes = [
            torch.tensor([10, 10, 50, 50], dtype=torch.float32),
            torch.tensor([100, 100, 150, 150], dtype=torch.float32)
        ]
        
        # Should contain both expected boxes (order may vary)
        for expected_box in expected_boxes:
            assert any(torch.equal(filtered_boxes[i], expected_box) for i in range(len(filtered_boxes)))
    
    def test_threshold_filtering(self):
        """Test 5: Threshold filtering - should filter low confidence boxes"""
        grid_size = 19
        num_anchors = 3
        num_classes = 63
        
        # Two boxes, one above threshold, one below
        boxes = torch.zeros(grid_size, grid_size, num_anchors, 4)
        boxes[0, 0, 0] = torch.tensor([10, 10, 50, 50])  # High confidence
        boxes[0, 0, 1] = torch.tensor([20, 20, 60, 60])  # Low confidence
        
        box_confidence = torch.zeros(grid_size, grid_size, num_anchors, 1)
        box_confidence[0, 0, 0] = torch.tensor([0.9])  # High confidence
        box_confidence[0, 0, 1] = torch.tensor([0.3])  # Low confidence
        
        box_class_probs = torch.zeros(grid_size, grid_size, num_anchors, num_classes)
        box_class_probs[0, 0, 0, 0] = 0.9  # High confidence box
        box_class_probs[0, 0, 1, 0] = 0.8  # Low confidence box
        
        # Use high threshold
        scores, filtered_boxes, classes = yolo_filter_boxes(
            boxes, box_confidence, box_class_probs, threshold=0.7
        )
        
        # Should only return high confidence box
        assert len(scores) == 1
        assert scores[0] == pytest.approx(0.81, rel=1e-3)  # 0.9 * 0.9
        assert torch.equal(filtered_boxes[0], torch.tensor([10, 10, 50, 50], dtype=torch.float32))
        assert classes[0] == 0


if __name__ == "__main__":
    # Run tests manually
    test_suite = TestYoloFilterBoxes()
    
    print("Running yolo_filter_boxes tests...")
    
    try:
        test_suite.test_three_boxes_select_highest_confidence()
        print("✅ Test 1: Three boxes select highest confidence - PASSED")
    except Exception as e:
        print(f"❌ Test 1: Three boxes select highest confidence - FAILED: {e}")
    
    try:
        test_suite.test_single_box()
        print("✅ Test 2: Single box - PASSED")
    except Exception as e:
        print(f"❌ Test 2: Single box - FAILED: {e}")
    
    try:
        test_suite.test_no_boxes()
        print("✅ Test 3: No boxes - PASSED")
    except Exception as e:
        print(f"❌ Test 3: No boxes - FAILED: {e}")
    
    try:
        test_suite.test_multiple_boxes_above_threshold()
        print("✅ Test 4: Multiple boxes above threshold - PASSED")
    except Exception as e:
        print(f"❌ Test 4: Multiple boxes above threshold - FAILED: {e}")
    
    try:
        test_suite.test_threshold_filtering()
        print("✅ Test 5: Threshold filtering - PASSED")
    except Exception as e:
        print(f"❌ Test 5: Threshold filtering - FAILED: {e}")
    
    print("\nAll yolo_filter_boxes tests completed!")