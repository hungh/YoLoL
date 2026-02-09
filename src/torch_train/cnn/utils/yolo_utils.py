"""
YoLo Utility functions
"""

import torchvision
import torch


def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (box1_x1, box1_y1, box1_x2, box_1_y2)
    box2 -- second box, list object with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)
    """


    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2

    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    inter_width = xi2 - xi1
    inter_height =  yi2 - yi1
    inter_area = max(inter_height, 0) * max(inter_width, 0)

    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area
    
    # compute the IoU
    iou = inter_area / union_area 
    
    return iou

def scale_boxes(boxes, image_shape):
    """Scale boxes back to original image shape."""
    height, width = image_shape
    box_coords = boxes.clone()
    
    # Convert from normalized to pixel coordinates
    box_coords[..., 0] *= width   # x_min
    box_coords[..., 1] *= height  # y_min  
    box_coords[..., 2] *= width   # x_max
    box_coords[..., 3] *= height  # y_max
    
    # Clamp to image bounds
    box_coords[..., [0, 2]] = torch.clamp(box_coords[..., [0, 2]], 0, width)   # x coordinates
    box_coords[..., [1, 3]] = torch.clamp(box_coords[..., [1, 3]], 0, height)  # y coordinates
    
    return box_coords
    

def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold = .6):
    """Filters YOLO boxes by thresholding on object and class confidence.
    
    Arguments:
        boxes -- tensor of shape (19, 19, 3, 4)
        box_confidence -- tensor of shape (19, 19, 3, 1)
        box_class_probs -- tensor of shape (19, 19, 3, 63)
        threshold -- real value, if [ highest class probability score < threshold],
                     then get rid of the corresponding box

    Returns:
        scores -- tensor of shape (None,), containing the class probability score for selected boxes
        boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
        classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """
    box_scores = box_confidence * box_class_probs # (19, 19, 3, 63)

    if len(box_scores.shape) > 3:
        box_scores = box_scores.squeeze(0)
        boxes = boxes.squeeze(0)
        box_confidence = box_confidence.squeeze(0)
        box_class_probs = box_class_probs.squeeze(0)

    # set dim to -1 to get the index of the max value in the last dimension
    box_classes = torch.argmax(box_scores, dim=-1) # (19, 19, 3) where data is index of class with max prob
    box_class_scores = torch.max(box_scores, dim=-1)[0] # (19, 19, 3) where data is the score of a class with prob
    
    # Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    filtering_mask = box_class_scores >= threshold # (19, 19, 3) where data is Bool
    
    # Apply the mask to box_class_scores, boxes and box_classes
    scores = box_class_scores[filtering_mask] #  for example (1454, ) selected boxes with prob, prob>=threshold
    boxes = boxes[filtering_mask] ## like (1454, 4)
    classes = box_classes[filtering_mask] # something like (1454, )
    
    return scores, boxes, classes


def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes
    
    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None, ), predicted class for each box
    
    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes, dtype=torch.float32)
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, dtype=torch.float32)
    if not isinstance(classes, torch.Tensor):
        classes = torch.tensor(classes, dtype=torch.int64)

    nms_indices = []
    classes_labels = torch.unique(classes) # Get unique classes
    
    for label in classes_labels:
        filtering_mask = classes == label
    
        # Get boxes for this class    
        boxes_label = boxes[filtering_mask]
        
        # Get scores for this class
        scores_label = scores[filtering_mask] # shape=(small_num, ); scores that match with one label (i.e, car)
        
        if scores_label.shape[0] > 0:  # Check if there are any boxes to process

            nms_indices_label = torchvision.ops.nms(boxes_label, scores_label, iou_threshold)

            # Get original indices of the selected boxes
            selected_indices = torch.where(filtering_mask)[0] # indices of selected (labeled) boxes;(not via NSM)
            
            # Append the resulting boxes into the partial result
            nms_indices.append(selected_indices[nms_indices_label])   # since box_labels is trimmed in size,
            # it has a different index value with 'score'. tf.gather puts the original inddex back to 'nms_indices'
            

    # Flatten the list of indices and concatenate
    if nms_indices:
        nms_indices = torch.cat(nms_indices, axis=0)
    else:
        nms_indices = torch.tensor([], dtype=torch.int64)
        print("[WARN]No boxes found. Return empty tensors.")

    scores = scores[nms_indices]
    boxes = boxes[nms_indices]
    classes = classes[nms_indices]
    
    # Sort by scores and return the top max_boxes
    sort_order = torch.argsort(scores, descending=True)
    scores = scores[sort_order[:max_boxes]]
    boxes = boxes[sort_order[:max_boxes]]
    classes = classes[sort_order[:max_boxes]]

    return scores, boxes, classes


def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return torch.cat([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ], dim = -1)


def yolo_eval(yolo_outputs, image_shape = (720, 1280), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.
    
    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (64, 64, 3)), contains 4 tensors:
                    box_xy: tensor of shape (None, 19, 19, 3, 2)
                    box_wh: tensor of shape (None, 19, 19, 3, 2)
                    box_confidence: tensor of shape (None, 19, 19, 3, 1)
                    box_class_probs: tensor of shape (None, 19, 19, 3, 63)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (64, 64) (has to be int dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
    
    # Retrieve outputs of the YOLO model (≈1 line)
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    
    # Convert boxes to be ready for filtering functions (convert boxes box_xy and box_wh to corner coordinates)
    boxes = yolo_boxes_to_corners(box_xy, box_wh) # (19, 19, 5, 4) bx, by, bh, bw
    
    # Use the function `yolo_filter_boxes` you've implemented to perform Score-filtering with a threshold of score_threshold
    scores, boxes, classes = yolo_filter_boxes(boxes, # Use boxes
                                  box_confidence, # Use box confidence
                                  box_class_probs, # Use box class probability
                                  threshold=score_threshold  # Use threshold=score_threshold
                                 )
    
    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape) # (720, 1280, 5, 4)
    
    # Use the function `yolo_non_max_suppression` you've implemented to perform Non-max suppression with 
    # maximum number of boxes set to max_boxes and a threshold of iou_threshold
    scores, boxes, classes = yolo_non_max_suppression(scores, # Use scores
                                  boxes, # Use boxes
                                  classes, # Use classes
                                  max_boxes, # Use max boxes
                                  iou_threshold=iou_threshold  # Use iou_threshold=iou_threshold
                                 )
    
    return scores, boxes, classes