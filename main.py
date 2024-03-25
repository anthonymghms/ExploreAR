import time
import os
import cv2
import numpy as np
import supervision as sv
import MEP as MyMep
import torch
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from contextlib import asynccontextmanager

def set_globals(OutputPath):
    global OUTPUT_PATH, HOME, CHECKPOINT_PATH, DETECTION_PATH, DEVICE, MODEL_TYPE
    HOME = os.getcwd()
    OUTPUT_PATH = os.path.join(HOME,OutputPath)
    CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
    DETECTION_PATH = os.path.join(HOME, "weights", "best.pt")
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"
    
def load_models(MODEL_TYPE, CHECKPOINT_PATH, DETECTION_PATH, DEVICE):
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    mask_predictor = SamPredictor(sam)
    model = YOLO(DETECTION_PATH)
    return mask_predictor, model
    
def preprocess_image(buffer, model):
    open_cv_image = np.array(Image.open(BytesIO(buffer)))
    results = model.predict(open_cv_image)
    xywh, label = results[0].boxes.xywh.tolist()[0], results[0].names.get(0)
    return xywh, label, open_cv_image

def segment_image(open_cv_image, xywh, label, maskPredictor, showImage = False):
    global IM_WIDTH, IM_HEIGHT
    existing_box = { "x": (xywh[0] - (xywh[2]/2)), "y": (xywh[1] - (xywh[3]/2)), "width": xywh[2], "height": xywh[3], 'label': label}
    box = np.array([
        existing_box['x'],
        existing_box['y'],
        existing_box['x'] + existing_box['width'],
        existing_box['y'] + existing_box['height']
    ])
    image_rgb = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
    IM_WIDTH, IM_HEIGHT = image_rgb.shape[1], image_rgb.shape[0]
    maskPredictor.set_image(image_rgb)

    masks, _, _ = maskPredictor.predict(
        box=box,
        multimask_output=True
    )
    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks=masks),
        mask=masks
    )
    detections = detections[detections.area == np.max(detections.area)]
    if showImage:
        box_annotator = sv.BoxAnnotator(color=sv.Color.RED)
        mask_annotator = sv.MaskAnnotator(color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX)
        source_image = box_annotator.annotate(scene=image_rgb.copy(), detections=detections, skip_label=True)
        segmented_image = mask_annotator.annotate(scene=image_rgb.copy(), detections=detections)
        sv.plot_images_grid(
            images=[source_image, segmented_image],
            grid_size=(1, 2),
            titles=['source image', 'segmented image']
        )
    return masks
    
def save_image(image):
    plt.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.imshow(image)
    plt.savefig(OUTPUT_PATH)
    
def find_cleaned_hull(masks):
    opencv_mask_image = np.uint8(masks) * 255
    gray = np.float32(opencv_mask_image)
    dst = cv2.dilate(cv2.cornerHarris(gray,2,3,0.04),None)
    _, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    _, _, _, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    hull = cv2.convexHull(corners)
    cleaned_hull = []
    for point in hull:
        cleaned_point = np.array(point[0].tolist(), dtype=np.float32)
        cleaned_hull.append(cleaned_point)
    cleaned_hull = np.array(cleaned_hull)
    
    return cleaned_hull

def get_closest_points(cleaned_hull):
    _, v1, v2, v3, v4, _, _ = MyMep.mep(cleaned_hull)
    points = []
    points.append(v1)
    points.append(v2)
    points.append(v3)
    points.append(v4)
    mep_points = np.array(points)
    def find_closest_hull_point(mep_point, cleaned_hull):
        distances = np.linalg.norm(cleaned_hull - mep_point, axis=1)
        closest_point_index = np.argmin(distances)
        return cleaned_hull[closest_point_index]

    closest_points = []

    for point in mep_points:
        closest_point = find_closest_hull_point(point, cleaned_hull)
        closest_points.append(closest_point)

    closest_points = np.array(closest_points)
    
    return closest_points

def fit_points_to_image(closest_points, openCvImage, saveImage = False):
    image = cv2.resize(openCvImage,(IM_WIDTH,IM_HEIGHT))

    for point in closest_points:
        x1,y1 = point.astype(int)
        cv2.circle(image, (x1, y1), radius=5, color=(0, 0, 255), thickness=-1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.show()
    
    if saveImage:
        save_image(image_rgb)
        
@asynccontextmanager
async def lifespan(app: FastAPI):
    global mask_predictor, model
    set_globals("\output.jpg")
    mask_predictor, model = load_models(MODEL_TYPE, CHECKPOINT_PATH, DETECTION_PATH, DEVICE)
    print("Models loaded")
    yield
    del mask_predictor, model

app = FastAPI(lifespan=lifespan)
    
@app.post('/run/')
async def main(file: UploadFile = File(...)):
    if file.filename:
            start_time = time.time()
            buffer = await file.read()
            xywh, label, opencv_image = preprocess_image(buffer, model)
            masks = segment_image(opencv_image,xywh, label,mask_predictor, False)
            cleaned_hull = find_cleaned_hull(masks[2])
            closest_points = get_closest_points(cleaned_hull)
            fit_points_to_image(closest_points, opencv_image, True)
            end_time = time.time()
            print(f"Time taken: {end_time - start_time}")
            return {"Success": "Image processed successfully", "OutputPath": OUTPUT_PATH, "closePoints": closest_points.tolist()}
    else:
        return {"Error": "File name is empty"}