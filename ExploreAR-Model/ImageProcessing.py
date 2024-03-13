
import os
import base64
import cv2
import numpy as np
import supervision as sv
import MEP as MyMep
import torch
from segment_anything import sam_model_registry, SamPredictor
from jupyter_bbox_widget import BBoxWidget
import matplotlib.pyplot as plt


HOME = os.getcwd()
print("HOME:", HOME)
# %mkdir -p {HOME}/weights
# wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P ./weights

CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
data_dir = os.path.join(os.path.expanduser('.'), 'data')
os.makedirs(data_dir, exist_ok=True)
print(f"Directory '{data_dir}' has been created.")
## Load Model
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_predictor = SamPredictor(sam)

IMAGE_NAME = "Baalbek.jpg"
IMAGE_PATH = os.path.join(HOME, "data", IMAGE_NAME)


def encode_image(filepath):
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    encoded = str(base64.b64encode(image_bytes), 'utf-8')
    return "data:image/jpg;base64,"+encoded

widget = BBoxWidget()
widget.image = encode_image(IMAGE_PATH)
widget
widget.bboxes

# default_box is going to be used if you will not draw any box on image above
default_box = {'x': 68, 'y': 247, 'width': 555, 'height': 678, 'label': ''}

box = widget.bboxes[0] if widget.bboxes else default_box
box = np.array([
    box['x'],
    box['y'],
    box['x'] + box['width'],
    box['y'] + box['height']
])

image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

mask_predictor.set_image(image_rgb)

masks, scores, logits = mask_predictor.predict(
    box=box,
    multimask_output=True
)
box_annotator = sv.BoxAnnotator(color=sv.Color.RED)
mask_annotator = sv.MaskAnnotator(color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX)

detections = sv.Detections(
    xyxy=sv.mask_to_xyxy(masks=masks),
    mask=masks
)
detections = detections[detections.area == np.max(detections.area)]

source_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections, skip_label=True)
segmented_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

sv.plot_images_grid(
    images=[source_image, segmented_image],
    grid_size=(1, 2),
    titles=['source image', 'segmented image']
)

plt.axis('off')
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.imshow(masks[2], interpolation='nearest', cmap='gray', aspect='auto')
plt.savefig('./data/Mask.png')
filename = './data/Mask.png'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# find Harris corners
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None)
ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)
# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
# define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
hull = cv2.convexHull(corners)
cleaned_hull = []

# Iterate through the convex hull points
for point in hull:
    # Convert the point to a list and then back to a numpy array with a float data type
    cleaned_point = np.array(point[0].tolist(), dtype=np.float32)
    # Append the cleaned point to the cleaned_hull list
    cleaned_hull.append(cleaned_point)

cleaned_hull = np.array(cleaned_hull)

plt.imshow(img)
plt.show()


area, v1, v2, v3, v4, z1o, z2o = MyMep.mep(cleaned_hull)

points = []
points.append(v1)
points.append(v2)
points.append(v3)
points.append(v4)
mep_points = np.array(points)
def find_closest_hull_point(mep_point, cleaned_hull):
    # Calculate the distance from the mep point to each point in the cleaned hull
    distances = np.linalg.norm(cleaned_hull - mep_point, axis=1)
    # Find the index of the closest point
    closest_point_index = np.argmin(distances)
    # Return the closest point
    return cleaned_hull[closest_point_index]

closest_points = []

for point in mep_points:
  closest_point = find_closest_hull_point(point, cleaned_hull)
  closest_points.append(closest_point)

closest_points = np.array(closest_points)
newfilename = './data/Baalbek.jpg'
image = cv2.resize(cv2.imread(newfilename),(640,480))

for point in closest_points:
  x1,y1 = point.astype(int)
  cv2.circle(image, (x1, y1), radius=5, color=(0, 0, 255), thickness=-1)  # Red color

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.show()