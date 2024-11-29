import torch

def SamBase(model, processor, image, device, boxes):
    """
    Segment objects in an image using the SamBase model based on provided bounding boxes.

    Args:
        model: The SamBase model.
        processor: The processor for preparing inputs and post-processing outputs.
        image: The input image.
        device: The device to run computations on ('cuda' or 'cpu').
        boxes: List of bounding boxes for the objects to segment.
    
    Returns:
        Tensor: The segmentation mask for the objects.
    """
    inputs = processor(images=image, input_boxes=boxes, return_tensors='pt').to(device)
    
    with torch.no_grad():
            outputs = model(**inputs)

    return processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]
