import torch
import tempfile
import os
from src.torch_train.dataset.YoLoDataSet import YoLoDataSet

class TestYoLoDataSet:
    
    def test_multiple_boxes_per_grid_cell(self):
        """Test _get_yolo_encoding with multiple boxes per grid cell"""
        
        # Create temporary annotation file with 3 boxes (2 classes) all in same grid cell (0,0)
        annotation_content = """0 0.15 0.15 0.10 0.10  
0 0.25 0.25 0.20 0.15  
1 0.15 0.15 0.25 0.20  
"""
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(annotation_content)
            annotation_path = f.name
        
        # Create dataset instance with small grid (3x3)
        dataset = YoLoDataSet(
            image_dir=None,
            annotation_dir=os.path.dirname(annotation_path),
            grid_size=3,
            num_anchors=3,
            num_classes=5
        )
        
        # Test image size
        image_size = (90, 90)
        
        # Call the function
        print("=== Starting YoLo encoding test ===")
        result = dataset._get_yolo_encoding(annotation_path, image_size)
        print("=== YoLo encoding complete ===")

        # Check the actual values:
        print("Grid cell (0,0) full data:")
        for anchor_idx in range(3):
            confidence = result[0, 0, anchor_idx, 4]
            classes = result[0, 0, anchor_idx, 5:]
            print(f"  Anchor {anchor_idx}: confidence={confidence:.1f}, classes={classes}")
        
        # Verify result shape
        assert result.shape == (3, 3, 3, 10), f"Expected shape (3, 3, 3, 10), got {result.shape}"
        
        # Test grid cell (0,0) - should have 3 boxes, 2 classes, 3 active anchors
        grid_0_0 = result[0, 0, :, :]  # All 3 anchor boxes for grid cell (0,0)
        
        # Check that exactly 3 anchor boxes are activated (one per box)
        active_anchors = grid_0_0[:, 4]  # Confidence values
        active_count = torch.sum(active_anchors > 0).item()
        assert active_count == 3, f"Grid cell (0,0) should have exactly 3 active anchors, got {active_count}"
        
        # Find which anchors are active
        active_anchor_indices = torch.where(active_anchors > 0)[0]
        assert len(active_anchor_indices) == 3, f"Should have 3 active anchor indices, got {len(active_anchor_indices)}"
        
        # Test class encoding - should have both class 0 and class 1
        class_encodings = grid_0_0[:, 5:]  # All class probabilities for all anchors
        
        # Count how many anchors have each class
        class_0_count = 0
        class_1_count = 0
        anchors_for_class_0 = []
        anchors_for_class_1 = []
        
        for anchor_idx in active_anchor_indices:
            if class_encodings[anchor_idx, 0] == 1.0:  # Class 0
                class_0_count += 1
                anchors_for_class_0.append(anchor_idx.item())
            if class_encodings[anchor_idx, 1] == 1.0:  # Class 1
                class_1_count += 1
                anchors_for_class_1.append(anchor_idx.item())
        
        assert class_0_count == 2, f"Should have 2 anchors with class 0, got {class_0_count}"
        assert class_1_count == 1, f"Should have 1 anchor with class 1, got {class_1_count}"
        assert len(anchors_for_class_0) == 2, "Should have 2 different anchors for class 0"
        assert len(anchors_for_class_1) == 1, "Should have 1 anchor for class 1"
        
        # Test that class 0 uses different anchors (no overlap)
        assert len(set(anchors_for_class_0)) == 2, "Class 0 should use 2 different anchors"
        
        # Test offset calculations are within valid range
        for anchor_idx in active_anchor_indices:
            x_offset = grid_0_0[anchor_idx, 0]
            y_offset = grid_0_0[anchor_idx, 1]
            assert 0 <= x_offset <= 1, f"x_offset should be in [0,1], got {x_offset}"
            assert 0 <= y_offset <= 1, f"y_offset should be in [0,1], got {y_offset}"
        
        print("✅ All tests passed!")
        
        # Cleanup
        os.unlink(annotation_path)

if __name__ == "__main__":
    test_suite = TestYoLoDataSet()
    
    print("Running YoLoDataSet tests...")
    
    try:
        test_suite.test_multiple_boxes_per_grid_cell()
        print("✅ Test 1: Multiple boxes per grid cell - PASSED")
    except Exception as e:
        print(f"❌ Test 1: Multiple boxes per grid cell - FAILED: {e}")
    
    try:
        test_suite.test_single_box_per_grid_cell()
        print("✅ Test 2: Single box per grid cell - PASSED")
    except Exception as e:
        print(f"❌ Test 2: Single box per grid cell - PASSED: {e}")
    
    try:
        test_suite.test_anchor_selection_accuracy()
        print("✅ Test 3: Anchor selection accuracy - PASSED")
    except Exception as e:
        print(f"❌ Test 3: Anchor selection accuracy - FAILED: {e}")
    
    print("\nAll tests completed!")