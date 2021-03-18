import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE
    area_pred = (prediction_box[2] - prediction_box[0])*(prediction_box[3] - prediction_box[1])
    area_truth = (gt_box[2] - gt_box[0])*(gt_box[3] - gt_box[1])
    # Compute intersection
    # Find location of intersection box
    intersection_xmin = max([prediction_box[0], gt_box[0]])
    intersection_xmax = min([prediction_box[2], gt_box[2]])
    intersection_ymin = max([prediction_box[1], gt_box[1]])
    intersection_ymax = min([prediction_box[3], gt_box[3]])
    # Check if there is no overlap
    if (intersection_xmax <= intersection_xmin) or (intersection_ymax <= intersection_ymin):
        return 0
    # Compute area of intersection
    intersection_area = (intersection_xmax - intersection_xmin)*(intersection_ymax - intersection_ymin)
    # Compute union
    union = area_pred + area_truth - intersection_area
    iou = intersection_area/union
    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if (num_tp + num_fp) == 0:
        return 1
    return num_tp/(num_tp + num_fp)


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if (num_tp + num_fn) == 0:
        return 0
    return num_tp/(num_tp + num_fn)


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold
    box_matches = []
    for p_box in prediction_boxes:
        for gt_box in gt_boxes:
            iou = calculate_iou(p_box, gt_box)
            if iou >= iou_threshold:
                box_matches.append((p_box, gt_box))
    # Sort all matches on IoU in descending order
    box_matches.sort(reverse=True, key=lambda x: calculate_iou(x[0], x[1]))
    # Find all matches with the highest IoU threshold
    prediction_matches = []
    gt_matches = []
    for pair in box_matches:
        if not(any(np.array_equal(pair[0], arr) for arr in gt_matches)) and not(any(np.array_equal(pair[1], arr) for arr in prediction_matches)):
            prediction_matches.append(pair[1])
            gt_matches.append(pair[0])
    return np.array(prediction_matches), np.array(gt_matches)


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    prediction_matches, gt_matches = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
    result = {"true_pos": 0, "false_pos": 0, "false_neg": 0}
    result["true_pos"] = len(prediction_matches)
    result["false_pos"] = len(prediction_boxes) - len(prediction_matches)
    result["false_neg"] = len(gt_boxes) - len(gt_matches)
    return result


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    true_pos = 0
    false_pos = 0
    false_neg = 0
    for i in range(len(all_prediction_boxes)):
        prediction_boxes = all_prediction_boxes[i]
        gt_boxes = all_gt_boxes[i]
        res = calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold)
        true_pos += res["true_pos"]
        false_pos += res["false_pos"]
        false_neg += res["false_neg"]
    precision = calculate_precision(true_pos, false_pos, false_neg)
    recall = calculate_recall(true_pos, false_pos, false_neg)
    return (precision, recall)


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    # YOUR CODE HERE
    precisions = []
    recalls = []
    for confidence_threshold in confidence_thresholds:
        confident_prediction_boxes = []
        for img_index in range(len(confidence_scores)):
            img_prediction_boxes = []
            img_gt_boxes = []
            for box_index in range(len(confidence_scores[img_index])):
                if confidence_scores[img_index][box_index] >= confidence_threshold:
                    img_prediction_boxes.append(all_prediction_boxes[img_index][box_index])
            confident_prediction_boxes.append(img_prediction_boxes)
        precision, recall = calculate_precision_recall_all_images(
            confident_prediction_boxes, all_gt_boxes, iou_threshold)
        precisions.append(precision)
        recalls.append(recall)
    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE
    average_precision = 0
    for recall_level in recall_levels:
        valid_precisions = []
        for i in range(len(recalls)):
            if recalls[i] >= recall_level:
                valid_precisions.append(precisions[i])
        if valid_precisions:
            average_precision += max(valid_precisions)
    average_precision = average_precision/len(recall_levels)
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
