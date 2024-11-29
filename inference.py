from PIL import Image
import torch
import transformers as T
import argparse
from utils import get_filenames, refine_masks, combine_masks
from constants import FOODS, MODELS, NICKNAMES
from segmentation_models import *
from detection_models import *
from data_analysis import *
import os
from tqdm import tqdm
import cv2

# Args.
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=int, default=0, help='model')
parser.add_argument('--threshold', default=None, help='threshold')
parser.add_argument('--topk', default=None, help='top k')
parser.add_argument('--nms', type=float, default=.3, help='nms')
parser.add_argument('--poly', type=str, default='t', help='polygon refinement')
parser.add_argument('--dataset', type=str, default='FoodSeg103', help='dataset')

args = parser.parse_args()

model_id, nms, dataset = args.model, args.nms, args.dataset
polygon_refinement = args.poly.lower() in ('t', 'true', 'y', 'yes')

if not args.topk and not args.threshold:
    threshold = 0.0
    topk = 6
else:
    threshold = float(args.threshold) if args.threshold else 0.0
    topk = int(args.topk) if args.topk else None


# Globals.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = MODELS[model_id]

base_dir = f'data/{dataset}/Images/inf_dir'
model_dir = f'{'-'.join(model_name.split('/'))}-{threshold}-{nms}-{polygon_refinement}-{topk if topk else 'None'}'
output_dir = os.path.join(base_dir, model_dir)
os.makedirs(output_dir, exist_ok=True)


# Load models.
if model_id == 0 or model_id == 1:
    detector_model = T.Owlv2ForObjectDetection.from_pretrained(model_name).to(device)
    detector_processor = T.Owlv2Processor.from_pretrained(model_name)
elif model_id == 2 or model_id == 3:
    detector_model = T.OwlViTForObjectDetection.from_pretrained(model_name).to(device)
    detector_processor = T.OwlViTProcessor.from_pretrained(model_name)
elif model_id == 4:
    detector_model = T.OmDetTurboForObjectDetection.from_pretrained(model_name).to(device)
    detector_processor = T.AutoProcessor.from_pretrained(model_name)
elif model_id == 5 or model_id == 6:
    detector_model = T.AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device)
    detector_processor = T.AutoProcessor.from_pretrained(model_name)

segmentor_model = T.AutoModelForMaskGeneration.from_pretrained('facebook/sam-vit-base').to(device)
segmentor_processor = T.AutoProcessor.from_pretrained('facebook/sam-vit-base')



# Inference.
filenames = [filename for filename in get_filenames(dataset) if not os.path.exists(os.path.join(output_dir, filename + '.png'))]
for filename in tqdm(filenames, desc='Inferencing'):

    image = Image.open(os.path.join(f'data/{dataset}/Images/img_dir/test', filename + '.jpg'))
        
    # Detect.
    if model_id in (0, 1, 2, 3):
        scores, boxes, ingredients_names = Owl(detector_model, detector_processor, image, device, FOODS, threshold, topk)
    elif model_id == 4:
        scores, boxes, ingredients_names = OmdetTurboSwinTinyHf(detector_model, detector_processor, image, device, FOODS, threshold, topk, nms)
    elif model_id in (5, 6):
        scores, boxes, ingredients_names = GroundingDino(detector_model, detector_processor, image, device, FOODS, threshold, topk)

    # Filter out unsensible return labels and map to respective ingredient id.
    filtered_results = [
        (s, FOODS.index(i), b) 
        for s, i, b in zip(scores, ingredients_names, boxes) 
        if i in FOODS
    ]

    if not filtered_results:
        mask = torch.zeros((image.size[1], image.size[0]), dtype=torch.long, device=device)
    else:
        # Unpack filtered results into separate lists.
        scores, ingredients_ids, boxes = zip(*filtered_results)
        scores = torch.stack(scores)
        ingredients_ids = torch.tensor(ingredients_ids).to(device)
        boxes = torch.stack(boxes)

        # Segment.
        masks = SamBase(segmentor_model, segmentor_processor, image, device, [boxes.tolist()])
        masks = refine_masks(masks, polygon_refinement)
        mask = combine_masks(scores, masks, ingredients_ids, device)

    # Save.
    cv2.imwrite(os.path.join(output_dir, filename + '.png'), mask.cpu().numpy())
    

# Metrics
metrics = get_directory_results(dataset, model_dir)
print(
    f"Dataset: {dataset}\n"
    f"Model: {NICKNAMES[model_name]}\n"
    f"Threshold: {threshold}\n"
    f"Top K: {topk}\n"
    f"Nms: {nms}\n"
    f"Poly: {polygon_refinement}\n"
    f"Metrics: {metrics['mIoU']:.2f}, {metrics['mAcc']:.2f}, {metrics['aAcc']:.2f}"
)
