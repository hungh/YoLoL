"""
This file is for processing the yolo model.
1. Run the preprocessing model to get the encoding
2. Run the yolo model to get the predictions
"""

import torch
from torchvision import transforms
import cv2
from ..architectures.all_models import PreYoloCNN32
from ..utils.yolo_utils import yolo_eval

# tech debt: use environment config instead of hardcoded paths
class PreTrainedYOLO:
    def __init__(self, preprocess_model_path="saved_models/PREPROCESS_YOLO.pt", max_boxes=10, score_threshold=.260, iou_threshold=.4):
        # Load model architecture
        self.preprocess_model = PreYoloCNN32()
        
        # Load trained weights
        state_dict = torch.load(preprocess_model_path, map_location='cpu')
        self.preprocess_model.load_state_dict(state_dict)
        
        # Set to evaluation mode
        self.preprocess_model.eval()
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_boxes = max_boxes
        
    
    def predict(self, image_file):
        # Load the image
        image = cv2.imread(image_file)
        
        if image is None:
            raise ValueError(f"Image not found at {image_file}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transform = transforms.Compose([
            transforms.Lambda(lambda x: cv2.resize(x, (64, 64))),
            transforms.Lambda(lambda x: x.transpose(2, 0, 1) / 255.0),
            transforms.Lambda(lambda x: torch.FloatTensor(x))
        ])
        
        image_tensor = transform(image_rgb).unsqueeze(0)

        with torch.no_grad():
            # get the encoding from the preprocessing model
            yolo_encoding = self.preprocess_model(image_tensor)
            print(f"Model output shape: {yolo_encoding.shape}")
            print(f"Model output range: [{yolo_encoding.min():.3f}, {yolo_encoding.max():.3f}]")
            print(f"Object confidence range: [{yolo_encoding[0, :, :, :, 4].min():.3f}, {yolo_encoding[0, :, :, :, 4].max():.3f}]")

            yolo_outputs =  self.get_yolo_outputs(yolo_encoding)
        
        out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, image_shape=image.shape[:2], max_boxes=self.max_boxes, score_threshold=self.score_threshold, iou_threshold=self.iou_threshold)
        
        # print the results
        print("out_scores:", out_scores)
        print("out_boxes:", out_boxes)
        print("out_classes:", out_classes)

    def get_yolo_outputs(self, encoding):
        """
        Parse the encoding from the YOLO model.
        Args:
            encoding: The encoding from the YOLO model. With shape (batch_size, 19, 19, 3, 68)
                    height, width, the number of anchor boxes, 3, 68 (5 bounding box parameters + 63 classes)
        Returns:
            The parsed encoding.
                box_xy : tensor of shape (batch_size, 19, 19, 3, 2)
                box_wh : tensor of shape (batch_size, 19, 19, 3, 2)
                box_confidence : tensor of shape (batch_size, 19, 19, 3, 1)
                box_class_probs : tensor of shape (batch_size, 19, 19, 3, 63)
        """        
        box_xy = encoding[:, :, :, :, :2]
        box_wh = encoding[:, :, :, :, 2:4]
        box_confidence = encoding[:, :, :, :, 4:5]
        class_probs = encoding[:, :, :, :, 5:]

        # Apply sigmoid to confidence to get [0,1] range
        box_confidence = torch.sigmoid(box_confidence)
        
        # Apply sigmoid to class probabilities
        class_probs = torch.sigmoid(class_probs)

        return [box_xy, box_wh, box_confidence, class_probs]
    
    if __name__ == "__main__":
        pre_trained_yolo = PreTrainedYOLO()
        image_file = "assets/produce_dataset/LVIS_Fruits_And_Vegetables/images/val/val/000000555239.jpg"
        pre_trained_yolo.predict(image_file)
        
