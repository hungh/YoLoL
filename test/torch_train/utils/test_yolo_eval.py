import torch
from src.torch_train.cnn.utils.yolo_utils import yolo_eval


class TestYoLoEval:
    def test_yolo_eval_24_boxes(self):
        """Test yolo_eval with 24 boxes across 19x19 grid cells"""

        
        # Create test data matching YOLO output format
        batch_size = 1
        grid_h, grid_w = 19, 19
        num_anchors = 3
        num_classes = 63
        
        # box_xy: (batch_size, grid_h, grid_w, num_anchors, 2) - center coordinates
        box_xy = torch.zeros(batch_size, grid_h, grid_w, num_anchors, 2)
        
        # box_wh: (batch_size, grid_h, grid_w, num_anchors, 2) - width/height
        box_wh = torch.zeros(batch_size, grid_h, grid_w, num_anchors, 2)
        
        # box_confidence: (batch_size, grid_h, grid_w, num_anchors, 1)
        box_confidence = torch.zeros(batch_size, grid_h, grid_w, num_anchors, 1)
        
        # box_class_probs: (batch_size, grid_h, grid_w, num_anchors, num_classes)
        box_class_probs = torch.zeros(batch_size, grid_h, grid_w, num_anchors, num_classes)
        
        # Create 24 boxes with varying confidence scores at different grid positions
        # We'll place them at different (i,j,k) positions in the 19x19x3 grid
        
        # High confidence boxes (should be selected)
        positions = [
            (2, 3, 0, 0.95, 0),   # (i,j,anchor, confidence, class)
            (5, 7, 1, 0.92, 1),
            (8, 12, 2, 0.88, 2),
            (11, 4, 0, 0.85, 3),
            (15, 16, 1, 0.82, 4),
            (18, 1, 2, 0.79, 5),
        ]
        
        # Medium confidence boxes (some may be selected)
        medium_positions = [
            (1, 8, 1, 0.65, 6),
            (4, 15, 0, 0.62, 7),
            (7, 2, 2, 0.68, 8),
            (10, 18, 1, 0.64, 9),
            (13, 9, 0, 0.67, 10),
            (16, 14, 2, 0.63, 11),
        ]
        
        # Low confidence boxes (should be filtered out by threshold)
        low_positions = [
            (0, 5, 1, 0.45, 12),
            (3, 11, 0, 0.42, 13),
            (6, 17, 2, 0.48, 14),
            (9, 6, 1, 0.44, 15),
            (12, 13, 0, 0.46, 16),
            (17, 3, 2, 0.43, 17),
        ]
        
        # Additional boxes to reach 24 total
        extra_positions = [
            (14, 8, 1, 0.55, 18),
            (2, 16, 0, 0.52, 19),
            (5, 10, 2, 0.58, 20),
            (8, 1, 1, 0.51, 21),
            (11, 14, 0, 0.56, 22),
            (18, 7, 2, 0.53, 23),
        ]
        
        all_positions = positions + medium_positions + low_positions + extra_positions
        
        # Fill the tensors with box data
        for idx, (i, j, anchor, conf, class_idx) in enumerate(all_positions):
            # Set box center coordinates (normalized between 0 and 1)
            box_xy[0, i, j, anchor, 0] = (j + 0.5) / grid_w  # x_center
            box_xy[0, i, j, anchor, 1] = (i + 0.5) / grid_h  # y_center
            
            # Set box dimensions (normalized)
            box_wh[0, i, j, anchor, 0] = 0.1  # width
            box_wh[0, i, j, anchor, 1] = 0.1  # height
            
            # Set confidence
            box_confidence[0, i, j, anchor, 0] = conf
            
            # Set class probability (high for the target class)
            box_class_probs[0, i, j, anchor, class_idx] = 0.9
        
        # Prepare yolo_outputs tuple
        yolo_outputs = (box_xy, box_wh, box_confidence, box_class_probs)
        
        # Test with default parameters
        scores, boxes, classes = yolo_eval(
            yolo_outputs=yolo_outputs,
            image_shape=(720, 1280),  # Default from docstring
            max_boxes=10,             # Default from docstring
            score_threshold=0.6,       # Default from docstring
            iou_threshold=0.5         # Default from docstring
        )
        
        # Assertions
        # Should return boxes with confidence > 0.6 (score_threshold)
        # High confidence boxes: 6 (should all pass)
        # Medium confidence boxes: 6 (some may pass depending on class probability)
        # Expected: 6-8 boxes total
        
        assert len(scores) > 0, "Should return some boxes"
        assert len(scores) <= 10, "Should not exceed max_boxes limit"
        assert len(boxes) == len(scores), "Boxes and scores should have same length"
        assert len(classes) == len(scores), "Classes and scores should have same length"
        
        # Check that scores are sorted in descending order
        if len(scores) > 1:
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], f"Scores should be sorted descending: {scores[i]} >= {scores[i+1]}"
        
        # Check that returned boxes have correct shape
        assert boxes.shape[1] == 4, "Boxes should have 4 coordinates"
        
        # Check that the highest confidence boxes are included
        # The top 6 should definitely be included (confidence > 0.6 * 0.9 = 0.54)
        expected_high_conf_classes = [0, 1, 2, 3, 4, 5]
        returned_classes = classes.tolist()
        
        # At least the top 3 highest confidence should be present
        top_classes = [0, 1, 2]  # First 3 from high confidence list
        for expected_class in top_classes:
            assert expected_class in returned_classes, f"High confidence class {expected_class} should be included"
        
        # Check that scores are reasonable (between 0 and 1)
        assert torch.all(scores >= 0), "All scores should be non-negative"
        assert torch.all(scores <= 1), "All scores should be <= 1"
        
        # Check that box coordinates are reasonable (within image bounds)
        assert torch.all(boxes >= 0), "All box coordinates should be non-negative"
        assert torch.all(boxes[:, 0] <= 1280), "x coordinates should be within image width"
        assert torch.all(boxes[:, 1] <= 720), "y coordinates should be within image height"
        assert torch.all(boxes[:, 2] <= 1280), "x coordinates should be within image width"
        assert torch.all(boxes[:, 3] <= 720), "y coordinates should be within image height"


if __name__ == "__main__":
    test_suite = TestYoLoEval()
    test_suite.test_yolo_eval_24_boxes()