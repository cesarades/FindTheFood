import os
import cv2
import torch
import csv
import numpy as np
import time
import argparse
from collections import defaultdict
from constants import FOODS, MODELS
from tqdm import tqdm


# Args.
parser = argparse.ArgumentParser()

parser.add_argument('--model', default=None, help='model')
parser.add_argument('--threshold', default=None, help='threshold')
parser.add_argument('--nms', default=None, help='nms')
parser.add_argument('--poly', default=None, help='polygon refinement')
parser.add_argument('--topk', default=None, help='top k')
parser.add_argument('--dataset', default='FoodSeg103', help='dataset')

args = parser.parse_args()

model = MODELS[int(args.model)] if args.model else None
threshold = float(args.threshold) if args.threshold else None
nms = float(args.nms) if args.nms else None
poly = args.poly.lower() in ('t', 'true', 'y', 'yes') if args.poly else None
topk = float(args.topk) if args.topk else None


# Set the device to GPU if available.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def inlcude_dir(dir_name):
    """
    Check if a directory should be included based on the provided arguments.

    Args:
        dir_name: The name of the directory to check.
    
    Returns:
        bool: True if the directory should be included, False otherwise.
    """
    if model and model.replace('/', '-') != '-'.join(dir_name.split('-')[:-4]):
        return False
    if threshold and threshold != dir_name.split('-')[-4]:
        return False
    if nms and nms != dir_name.split('-')[-3]:
        return False
    if poly and poly != dir_name.split('-')[-2]:
        return False
    if topk and topk != dir_name.split('-')[-1]:
        return False
    return True
        


def calculate_scores_by_class(predicted_mask, ground_truth_mask, labels):
    """
    Calculate true positives, false positives, and false negatives for each class label.

    Args:
        predicted_mask: The predicted mask tensor.
        ground_truth_mask: The ground truth mask tensor.
        labels: The unique labels present in the ground truth mask.
    
    Returns:
        dict: A dictionary containing the computed scores for each class.
    """
    scores = {}
    total_pixels = ground_truth_mask.numel()

    for label in labels:
        # Create binary masks for the current label.
        pred_binary = (predicted_mask == label)
        gt_binary = (ground_truth_mask == label)

        # Calculate TP, FP, FN.
        tp = (pred_binary & gt_binary).sum().item()
        fp = (pred_binary & ~gt_binary).sum().item()
        fn = (~pred_binary & gt_binary).sum().item()

        # Store the scores.
        label_name = FOODS[label.item()]
        scores[label_name] = (tp, fp, fn, total_pixels)

    return scores


def get_directory_results(dataset, directory=None):
    """
    Compute per-image scores for each model by comparing predicted masks with ground truth masks.

    Args:
        dataset: The dataset to process.
        directory: The subdirectory to process. If None, all subdirectories are processed.

    Returns:
        dict: A dictionary containing the computed metrics for each model.
    """
    per_model_scores = defaultdict(list)
    base_directory = f'data/{dataset}/Images/inf_dir'
    ground_truth_directory = f'data/{dataset}/Images/ann_dir/test'

    # Get the desired subdirectories.
    subdir_names = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    subdir_names = list(filter(inlcude_dir, subdir_names))
    if directory:
        subdir_names = [directory]

    # Iterate through each subdirectory in the base directory.
    for subdir_name in tqdm(subdir_names, desc='Models'):

        subdir_path = os.path.join(base_directory, subdir_name)
        image_names = os.listdir(subdir_path)

        # Process each image individually.
        for image_name in tqdm(image_names, desc=subdir_name, leave=False):
            pred_mask_path = os.path.join(subdir_path, image_name)
            gt_mask_path = os.path.join(ground_truth_directory, image_name)

            # Read masks using OpenCV.
            pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

            # Check if images were read successfully.
            if pred_mask is None or gt_mask is None:
                print(f'Warning: Failed to read masks for image {image_name}. Skipping.')
                continue  # Skip this image if reading failed.

            # Convert masks to torch tensors and move to device.
            pred_mask_tensor = torch.tensor(pred_mask, dtype=torch.int64, device=device)
            gt_mask_tensor = torch.tensor(gt_mask, dtype=torch.int64, device=device)

            # Get unique labels in the ground truth mask.
            labels = torch.unique(gt_mask_tensor)

            # Calculate scores by class.
            image_scores = calculate_scores_by_class(pred_mask_tensor, gt_mask_tensor, labels)

            # Append the image scores to the per_model_scores dictionary.
            per_model_scores[subdir_name].append(image_scores)

        # Aggregate the scores for each model.
        per_model_scores[subdir_name] = calculate_metrics(subdir_name, per_model_scores[subdir_name], dataset)

    return per_model_scores


def calculate_metrics(model_name, images_metrics, dataset):
    """
    Calculate mean Intersection over Union (mIoU), mean Accuracy (mAcc), and overall Accuracy (aAcc).

    Args:
        model_name: The name of the model.
        images_metrics: A list of dictionaries containing the metrics for each image.
        dataset: The dataset to process. If None, the default dataset is used.
    
    Returns:
        dict: A dictionary containing the computed metrics.
    """

    total_tp = 0
    total_pixels = 0
    label_ious = defaultdict(list)
    label_accs = defaultdict(list)

    for image_metrics in images_metrics:
        for label, metrics in image_metrics.items():
            # Extract metrics.
            tp, fp, fn, pixels = metrics

            # Calculate IoU and Accuracy.
            iou_denominator = tp + fp + fn
            acc_denominator = tp + fn

            iou = tp / iou_denominator if iou_denominator > 0 else 0
            acc = tp / acc_denominator if acc_denominator > 0 else 0

            # Append metrics to the label-specific lists.
            label_ious[label].append(iou)
            label_accs[label].append(acc)

            # Accumulate total true positives and pixels.
            total_tp += tp
            total_pixels += pixels

    # Calculate mean IoU and mean Accuracy for each label.
    mIoU = np.mean([np.mean(scores) for scores in label_ious.values()]) if label_ious else 0
    mAcc = np.mean([np.mean(scores) for scores in label_accs.values()]) if label_accs else 0

    # Calculate overall Accuracy.
    aAcc = (total_tp / total_pixels) if total_pixels > 0 else 0

    # Write the metrics to a CSV file.
    with open(f'data/{dataset}/Images/inf_dir/stored_metrics.csv', 'a', newline='') as f:
        writer = csv.writer(f)

        # Write headers if this is the first time writing.
        if f.tell() == 0:  # Use the correct file handle 'f' instead of 'file'
            writer.writerow(['time', 'model_name', 'threshold', 'nms', 'poly', 'topk', 'mIoU', 'mAcc', 'aAcc'])
        
        # Assuming model_name is a string like 'model-threshold-nms-poly-topk'
        model_name_parts = model_name.split('-')  # Correct the split operation
        writer.writerow([
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),  # Current timestamp
            '-'.join(model_name_parts[:-4]),  # Reconstruct the base model name
            model_name_parts[-4],  # Threshold
            model_name_parts[-3],  # NMS
            model_name_parts[-2],  # Poly
            model_name_parts[-1],  # Top-k
            mIoU * 100,  # mIoU percentage
            mAcc * 100,  # mAcc percentage
            aAcc * 100   # aAcc percentage
        ])

    return {'mIoU': mIoU * 100, 'mAcc': mAcc * 100, 'aAcc': aAcc * 100}


if __name__ == '__main__':
    # Get the per-model results.
    results = get_directory_results(dataset=args.dataset)

    # Print the results.
    print('\nSummary of Metrics:')
    for model, metrics in sorted(results.items()):
        print(f'Model: {model}')
        print(f'    mIoU: {metrics['mIoU']:.2f}')
        print(f'    mAcc: {metrics['mAcc']:.2f}')
        print(f'    aAcc: {metrics['aAcc']:.2f}\n')
