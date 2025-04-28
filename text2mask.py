from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os 
import torch 
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from torchvision.ops import box_convert
import argparse


"""device selection"""
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")


"""define the image """
IMAGE_PATH = "/home/anranli/Documents/DeepL/Final/Final Project Demo/frames/00036.jpg"



"""sam2 model selection"""
sam2_checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

"""DINO model selection"""
DINO_model = load_model("../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "../GroundingDINO/weights/groundingdino_swint_ogc.pth")
TEXT_PROMPT = "white square frame with handle"
BOX_TRESHOLD = 0.2
TEXT_TRESHOLD = 0.2



"""helper functions"""


def get_center(bbox, H, W):
    
    x_min, y_min, x_max, y_max = bbox
 
    cx = int(((x_min + x_max) / 2) * W )
    cy = int(((y_min + y_max) / 2) * H )
    
    return cx, cy

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

  
# def text_to_mask(IMAGE_PATH, DINO_model, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD):
#     image_source, image = load_image(IMAGE_PATH)
#     boxes, logits, phrases = predict(
#         model=DINO_model,
#         image=image,
#         caption=TEXT_PROMPT,
#         box_threshold=BOX_TRESHOLD,
#         text_threshold=TEXT_TRESHOLD
#     )

#     predictor.set_image(image_source)
    
#     xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
#     print("the xyxy is: ", xyxy)
    
#     if len(xyxy) == 0:
#         print("No objects detected with the given prompt and thresholds.")
#         return
    
#     H, W = image_source.shape[:2]
#     print(f"Image dimensions: H={H}, W={W}")
    
#     for box in xyxy:
#         print("Normalized box:", box)
        
#         x_min_px = int(box[0] * W)
#         y_min_px = int(box[1] * H)
#         x_max_px = int(box[2] * W)
#         y_max_px = int(box[3] * H)
        
#         input_box = np.array([x_min_px, y_min_px, x_max_px, y_max_px])
#         print("Pixel box:", input_box)
        
#         input_box = input_box[None, :]
        
#         masks, scores, logits = predictor.predict(
#             point_coords=None,
#             point_labels=None,
#             box=input_box,
#             multimask_output=True
#         )
        
#         print(f"Mask scores: {scores}")
        
#         sorted_ind = np.argsort(scores)[::-1]
#         masks = masks[sorted_ind]
#         scores = scores[sorted_ind]
#         logits = logits[sorted_ind]

#         print(f"Masks shape: {masks.shape}")
        
#         show_masks(image_source, masks, scores, box_coords=input_box[0], borders=True)

# if __name__ == "__main__":
#     text_to_mask(IMAGE_PATH, DINO_model, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD)



# def save_mask_output(image_path, binary_mask_path, DINO_model, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD):
#     image_tensor_input, image_tensor = load_image(image_path)  # <- Fix here!

#     boxes, logits, phrases = predict(
#         model=DINO_model,
#         image=image_tensor,
#         caption=TEXT_PROMPT,
#         box_threshold=BOX_TRESHOLD,
#         text_threshold=TEXT_TRESHOLD
#     )

#     predictor.set_image(image_tensor_input)
#     xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

#     if len(xyxy) == 0:
#         print("No objects detected.")
#         return

#     H, W = image_tensor_input.shape[:2]
#     for box in xyxy:
#         x_min_px = int(box[0] * W)
#         y_min_px = int(box[1] * H)
#         x_max_px = int(box[2] * W)
#         y_max_px = int(box[3] * H)
#         input_box = np.array([x_min_px, y_min_px, x_max_px, y_max_px])[None, :]

#         masks, scores, _ = predictor.predict(
#             point_coords=None,
#             point_labels=None,
#             box=input_box,
#             multimask_output=True
#         )

#         sorted_ind = np.argsort(scores)[::-1]
#         masks = masks[sorted_ind]
#         scores = scores[sorted_ind]

#         binary_mask = masks[0].astype(np.uint8) * 255  # Convert to 8-bit format
#         mask_img = Image.fromarray(binary_mask)
#         mask_img.save(binary_mask_path)
        
        
#         # # Save top-1 mask visualization
#         # plt.figure(figsize=(10, 10))
#         # plt.imshow(image_tensor_input)
#         # show_mask(masks[0], plt.gca())
#         # show_box(input_box[0], plt.gca())
#         # plt.axis("off")
#         # plt.savefig(output_path)
#         # plt.close()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input", required=True, help="Path to input image")
#     # parser.add_argument("--output", required=True, help="Path to save output mask image")
#     parser.add_argument("--binary_mask", required=True, help="Path to save binary mask image")
#     args = parser.parse_args()

#     input_path = args.input
#     # output_path = args.output
#     binary_mask_path = args.binary_mask


#     image = cv2.imread(input_path)
#     if image is None:
#         print(f"❌ Failed to load image: {input_path}")
#         exit(1)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     save_mask_output(input_path, binary_mask_path, DINO_model, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD)


def save_mask_output(image_path, binary_mask_path, DINO_model, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD):
    image_tensor_input, image_tensor = load_image(image_path)

    boxes, logits, phrases = predict(
        model=DINO_model,
        image=image_tensor,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    predictor.set_image(image_tensor_input)
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    if len(xyxy) == 0:
        print("No objects detected.")
        return False

    H, W = image_tensor_input.shape[:2]
    for box in xyxy:
        x_min_px = int(box[0] * W)
        y_min_px = int(box[1] * H)
        x_max_px = int(box[2] * W)
        y_max_px = int(box[3] * H)
        input_box = np.array([x_min_px, y_min_px, x_max_px, y_max_px])[None, :]

        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=True
        )

        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        binary_mask = masks[0].astype(bool)
        np.save(binary_mask_path, binary_mask)
        return True  # Successfully saved

    return False  # Shouldn't reach here if we found at least one box

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--binary_mask", required=True, help="Path to save binary mask image")
    args = parser.parse_args()

    input_path = args.input
    binary_mask_path = args.binary_mask

    image = cv2.imread(input_path)
    if image is None:
        print(f"❌ Failed to load image: {input_path}")
        exit(1)
    
    # Call the function with correct parameters
    success = save_mask_output(input_path, binary_mask_path, DINO_model, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD)
    
    if success:
        print(f"✅ Successfully saved binary mask to: {binary_mask_path}")
    else:
        print(f"❌ Failed to create mask for: {input_path}")
        exit(1)