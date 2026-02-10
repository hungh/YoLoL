"""
This file is for processing the yolo model.
1. Run the preprocessing model to get the encoding
2. Load a pre-trained YOLO model
3. Run the yolo model to get the predictions
"""
# from preprocess_yolo import PreprocessYOLO

import torch
import torchinfo

class PreTrainedYOLO:
    def __init__(self, preprocess_model_path):
        # self.preprocess_yolo = PreprocessYOLO()
        self.preprocess_model = torch.load(preprocess_model_path)
    
    def process(self, image_file):
        # Load the image
        image = cv2.imread(image_file)

        # get the encoding from the preprocessing model
        encoding = self.preprocess_model(image)
        
        
        # Load pre-trained YOLO model
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        # print model summary
        print(torchinfo.summary(model, (1, 3, 640, 640)))

        # Process the image
        results = model(encoding)
        
        # Get the predictions
        predictions = results.pandas().xyxy[0]
        
        return predictions

    def parse_yolo_encoding(self, encoding):
        """
        Parse the encoding from the YOLO model.
        Args:
            encoding: The encoding from the YOLO model. With shape (batch_size, 19, 19, 3, 68)
                    height, width, the number of anchor boxes, 3, 68 (63 classes + 5 bounding box parameters)
        Returns:
            The parsed encoding.
        """
        # Parse the encoding
        pass
        
