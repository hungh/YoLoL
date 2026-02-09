from torch.utils.data import Dataset
import glob
import os
import cv2
import numpy as np
import torch

class YoLoDataSet(Dataset):
    def __init__(self, image_dir, annotation_dir, grid_size=19, num_anchors=3, num_classes=68, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.grid_size = grid_size
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.transform = transform

        self.image_files = glob(os.path.join(self.image_dir, "*.jpg")) + glob(os.path.join(self.image_dir, "*.png"))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # for image
        image_path = self.image_files[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # for annotation
        annotation_path = os.path.join(self.annotation_dir, os.path.basename(image_path).replace(".jpg", ".txt"))
        target_encoding = self._get_yolo_encoding(annotation_path, image.shape)

        # transform image if needed
        if self.transform:
            image = self.transform(image)

        return image, target_encoding
        
    def _get_yolo_encoding(self, annotation_path, image_size):
        """
        Convert YOLO annotation to YOLO encoding (19, 19, 3, 68) # 68 = 63 classes + 5 bounding box parameters (1 Pc, 4 Bb)
        Args:
            annotation_path (str): path to the annotation file
            image_size (tuple): (width, height) of the image
        Returns:
            torch.Tensor: YOLO encoding (19, 19, 3, 68) # 3 anchor boxes, 63 classes + 5 bounding box parameters (1 Pc, 4 Bb)
        """
        target_encoding = np.zeros((self.grid_size, self.grid_size, self.num_anchors, self.num_classes + 5))

        if not os.path.exists(annotation_path):            
            return target_encoding
        
        # calculate the image height width and the grid size based on the image size
        img_width, img_height = image_size
        grid_width = img_width / self.grid_size
        grid_height = img_height / self.grid_size

        # read annotation file
        with open(annotation_path, 'r') as f:
            for line in f:
                anchor_box_idx = 0
                try:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split()) # bounding box parameters 

                    # calculate the fractions in the annotation file into the image's scale
                    x_center = x_center * img_width
                    y_center = y_center * img_height
                    width = width * img_width
                    height = height * img_height
                    
                    # calculate the grid cell and anchor box to the scale of the grid size (a.k.a grid coordinates)
                    grid_x = int(x_center / grid_width)
                    grid_y = int(y_center / grid_height)

                    # the grid x, y should be less than the grid size
                    if grid_x >= self.grid_size or grid_y >= self.grid_size:
                        continue

                    # calculate the relative coordinates within grid cell
                    x_offset = (x_center / grid_width) - grid_x # in float if the offet is zero, the x, y should be at the center of the grid cell
                    y_offset = (y_center / grid_height) - grid_y # in float

                    # set bounding box parameters
                    target_encoding[grid_y, grid_x, anchor_box_idx, 0] = x_offset
                    target_encoding[grid_y, grid_x, anchor_box_idx, 1] = y_offset
                    target_encoding[grid_y, grid_x, anchor_box_idx, 2] = width / grid_width # to the scale of the grid size
                    target_encoding[grid_y, grid_x, anchor_box_idx, 3] = height / grid_height
                    target_encoding[grid_y, grid_x, anchor_box_idx, 4] = 1.0 # confidence (is the class there)
                    
                    # set class using hot encoding
                    if class_id < self.num_classes:
                        target_encoding[grid_y, grid_x, anchor_box_idx, 5 + class_id] = 1.0
                    else:
                        # log the class id
                        print(f"Class id {class_id} is out of range, skipping")

                    # TODO: will run the Non-Maximum Suppression (NMS) using iou to remove duplicate bounding boxes later  
                    

                except Exception as e:
                    print(f"Failed to process annotation line/continue :{line}. Exception : {e}")
                    continue
        
        return torch.tensor(target_encoding)


        

        