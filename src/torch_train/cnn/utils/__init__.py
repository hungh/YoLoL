"""
Utility functions for training CNN models
"""

import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch

def imshow_with_unnormalize(img):
    """
    Show a single image. Unnormalize the image first.
    NOTE: the image was expected to be normalized to [-1, 1]
    """
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_images_in_grid(loader, classes, batch_size = 4):
    """
    Show images in a grid.
    
    Args:
        loader: DataLoader
        classes: list of class names
        batch_size: batch size
    Returns:
        images: images
        labels: labels
    """
    # get some random training images
    dataiter = iter(loader)
    images, labels = next(dataiter)

    # show images
    imshow_with_unnormalize(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    return images, labels


def predict_image(model, image):
    """
    Predict the class of an image
    Args:
        model: the model to predict
        image: the image to predict
    Returns:
        predicted: the predicted class
    """
    _, predicted = torch.max(model(image).data, 1)
    return predicted.cpu()


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
    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)
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
    sort_order = torch.argsort(scores, descending=True).numpy()
    scores = scores[sort_order[:max_boxes]]
    boxes = boxes[sort_order[:max_boxes]]
    classes = classes[sort_order[:max_boxes]]

    return scores, boxes, classes