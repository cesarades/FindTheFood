import torch


def GroundingDino(model, processor, image, device, all_ingredient_labels, threshold, topk):
    """
    Detect objects in an image using the GroundingDino model based on provided ingredient labels.
    The input is split into chunks since the number of labels exceeds the model size.

    Args:
        model: The GroundingDino model.
        processor: The processor for preparing inputs and post-processing outputs.
        image: The input image.
        device: The device to run computations on ('cuda' or 'cpu').
        all_ingredient_labels: List of ingredient labels to detect in the image.
        threshold: The confidence threshold for filtering detections.
        topk: The number of top detections to return.

    Returns:
        Tuple containing:
            - scores: Tensor of confidence scores for the detections.
            - boxes: Tensor of bounding boxes for the detections.
            - labels: List of labels corresponding to the detections.
    """
    n = len(all_ingredient_labels)

    # Divide the ingredient list into two chunks.
    chunk_1 = '. '.join(all_ingredient_labels[:n // 2]) + '.'
    chunk_2 = '. '.join(all_ingredient_labels[n // 2:]) + '.'

    # Initialize empty lists for scores, boxes, and labels.
    scores = []
    boxes = []
    labels = []

    # Process each chunk separately.
    for chunk in [chunk_1, chunk_2]:
        inputs = processor(images=image, text=chunk, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process the results.
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=threshold,
            text_threshold=threshold,
            target_sizes=[image.size[::-1]]
        )[0]

        # Append results to the respective lists.
        scores.append(results['scores'])
        boxes.append(results['boxes'])
        labels.extend([x.replace('.', '') for x in results['labels']])

    scores = torch.cat(scores, dim=0)
    boxes = torch.cat(boxes, dim=0)

    if topk and len(scores) > 0:
        topk = min(topk, len(scores))
        scores, indices = torch.topk(scores, topk)
        boxes = boxes[indices]
        labels = [labels[i] for i in indices]

    return scores, boxes, labels



def Owl(model, processor, image, device, all_ingredient_labels, threshold, topk):
    """
    Detect objects in an image using the Owl model based on provided ingredient labels.

    Args:
        model: The Owl model.
        processor: The processor for preparing inputs and post-processing outputs.
        image: The input image.
        device: The device to run computations on ('cuda' or 'cpu').
        all_ingredient_labels: List of ingredient labels to detect in the image.
        threshold: The confidence threshold for filtering detections.
        topk: The number of top detections to return.
    
    Returns:
        Tuple containing:
            - scores: Tensor of confidence scores for the detections.
            - boxes: Tensor of bounding boxes for the detections.
            - labels: List of labels corresponding to the detections.
    """
    inputs = processor(text=all_ingredient_labels, images=image, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs=outputs,
        target_sizes=torch.Tensor([image.size[::-1]]),
        threshold=threshold
        )[0]

    scores = results['scores']
    boxes = results['boxes']
    labels = [all_ingredient_labels[i] for i in results['labels'].tolist()]
    
    if topk and len(scores) > 0:
        topk = min(topk, len(scores))
        scores, indices = torch.topk(scores, topk)
        boxes = boxes[indices]
        labels = [labels[i] for i in indices]

    return scores, boxes, labels



def OmdetTurboSwinTinyHf(model, processor, image, device, all_ingredient_labels, threshold, topk, nms):
    """
    Detect objects in an image using the OmdetTurboSwinTinyHf model based on provided ingredient labels.

    Args:
        model: The OmdetTurboSwinTinyHf model.
        processor: The processor for preparing inputs and post-processing outputs.
        image: The input image.
        device: The device to run computations on ('cuda' or 'cpu').
        all_ingredient_labels: List of ingredient labels to detect in the image.
        threshold: The confidence threshold for filtering detections.
        topk: The number of top detections to return.
        nms: The non-maximum suppression threshold.
    
    Returns:
        Tuple containing:
            - scores: Tensor of confidence scores for the detections.
            - boxes: Tensor of bounding boxes for the detections.
            - labels: List of labels corresponding to the detections.
    """
    inputs = processor(image, text=all_ingredient_labels, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        classes=all_ingredient_labels,
        target_sizes=[image.size[::-1]],
        score_threshold=threshold,
        nms_threshold=nms,
        )[0]

    scores = results['scores']
    boxes = results['boxes']
    labels = results['classes']

    if topk and len(scores) > 0:
        topk = min(topk, len(scores))
        scores, indices = torch.topk(scores, topk)
        boxes = boxes[indices]
        labels = [labels[i] for i in indices]

    return scores, boxes, labels
