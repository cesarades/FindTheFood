import os
import cv2
import numpy as np
import torch


def get_filenames(dataset='FoodSeg103'):
    """
    Get the list of filenames in the dataset.
    """
    return [f[:-4] for f in os.listdir(f'data/{dataset}/Images/img_dir/test')]


def mask_to_polygon(mask: np.ndarray):
    """
    Convert a binary mask to a polygon.

        1. cv2.RETR_EXTERNAL retrieves only the outermost contours, ignoring any nested contours.
        2. cv2.CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments, leaving only their endpoints, which simplifies the contour.
        3. The max function with cv2.contourArea as the key selects the contour with the largest area. This assumes that the largest contour represents the main object of interest.
    """
    # Find contours in the binary mask.
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Select the largest contour based on area.
    largest_contour = max(contours, key=cv2.contourArea)

    # Reshape the contour to a 2D array of (x, y) coordinates and convert to a list.
    return largest_contour.reshape(-1, 2).tolist()


def polygon_to_mask(polygon, image_shape):
    """
    Convert a polygon to a binary mask.
    """
    # Create an empty mask with the given image shape.
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert the polygon points to a NumPy array of type int32.
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon on the mask with the color 1.
    cv2.fillPoly(mask, [pts], color=(1,))

    return mask


def refine_masks(masks, poly_refin=False):
    """
    Process SAM output and optionally refine binary masks by converting them to polygons and back to masks.
    """
    # Move masks to CPU and convert to float type.
    masks = masks.cpu().float()

    # Permute dimensions to (batch_size, height, width, channels).
    masks = masks.permute(0, 2, 3, 1)

    # Compute the mean across the channel dimension.
    masks = masks.mean(axis=-1)

    # Threshold the masks to obtain binary masks.
    masks = (masks > 0).int()

    # Convert masks to NumPy arrays of type uint8.
    masks = masks.numpy().astype(np.uint8)

    # Convert masks to a list for iteration.
    masks = list(masks)

    if poly_refin:
        # Refine each mask by converting to a polygon and back.
        for idx, mask in enumerate(masks):
            if mask.sum() > 0:
                shape = mask.shape
                polygon = mask_to_polygon(mask)
                mask = polygon_to_mask(polygon, shape)
                masks[idx] = mask

    return masks



def combine_masks(scores, masks, ingredients, device='cuda'):
    """
    Combine multiple detection masks into one final mask based on the following rules:
    - At each pixel, assign the label (ingredient ID) that appears most frequently.
    - In case of a tie, assign the label from the detection with the highest confidence score.

    Args:
        scores (Tensor or list): Confidence scores for each detection (num_detections).
        masks (Tensor or list): Binary masks for each detection (num_detections, height, width).
        ingredients (Tensor or list): Ingredient IDs for each detection (num_detections).
        device (str): Device to perform computations on ('cuda' or 'cpu').

    Returns:
        Tensor: The combined mask (height, width) with resolved labels.
    """

    # Convert inputs to torch tensors.
    scores = torch.tensor(scores, dtype=torch.float32) if not isinstance(scores, torch.Tensor) else scores
    ingredients = torch.tensor(ingredients, dtype=torch.long) if not isinstance(ingredients, torch.Tensor) else ingredients
    masks = torch.stack([torch.tensor(mask, dtype=torch.long) if not isinstance(mask, torch.Tensor) else mask for mask in masks])

    # Move tensors to the specified device.
    scores = scores.to(device)
    ingredients = ingredients.to(device)
    masks = masks.to(device)

    num_detections, height, width = masks.shape
    num_pixels = height * width

    # Test for edge cases.
    assert len(masks) == len(scores) == len(ingredients), 'Mismatch in parameter lengths'
    if len(masks) == 0:
        return torch.zeros((height, width), dtype=torch.long, device=device)

    # Assign ingredient IDs to masks.
    masks = masks * ingredients.view(-1, 1, 1)  # Now masks contain zeros and label IDs

    # Flatten masks to shape (num_detections, num_pixels).
    masks_flat = masks.view(num_detections, num_pixels)

    # Get the maximum label ID.
    max_label_id = masks_flat.max().item()

    # Initialize tensors for counts and label scores.
    counts = torch.zeros((max_label_id + 1, num_pixels), dtype=torch.long, device=device)
    label_scores = torch.zeros((max_label_id + 1, num_pixels), dtype=torch.float32, device=device)
    positions = torch.arange(num_pixels, device=device)

    # Loop over each detection to update counts and label scores.
    for i in range(num_detections):
        labels_i = masks_flat[i]  # Shape: (num_pixels,)
        scores_i = scores[i]
        valid = labels_i > 0  # Exclude background pixels
        labels_valid = labels_i[valid]
        positions_valid = positions[valid]
        indices = (labels_valid, positions_valid)

        # Update counts for each label ID at each pixel.
        counts.index_put_(indices, torch.ones_like(labels_valid), accumulate=True)

        # Update label scores by keeping the maximum score for each label ID at each pixel.
        current_scores = label_scores[indices]
        scores_i_expanded = scores_i.expand_as(current_scores)
        label_scores[indices] = torch.max(current_scores, scores_i_expanded)

    # Exclude background label ID (0) from counts.
    counts[0, :] = 0

    # Find the maximum count of label IDs at each pixel.
    max_counts, _ = counts.max(dim=0)  # Shape: (num_pixels,)

    # Identify label IDs that have the maximum count at each pixel (handle ties).
    ties = counts == max_counts.unsqueeze(0)  # Shape: (max_label_id + 1, num_pixels)

    # For tied labels, keep the one with the highest confidence score.
    tied_label_scores = label_scores.clone()
    tied_label_scores[~ties] = float('-inf')  # Exclude non-tied labels by setting their scores to -inf

    # Select the best label ID based on the highest score among tied labels.
    _, best_labels = tied_label_scores.max(dim=0)  # Shape: (num_pixels,)

    # Reshape the best labels back to the original image dimensions.
    final_mask = best_labels.view(height, width)

    return final_mask

