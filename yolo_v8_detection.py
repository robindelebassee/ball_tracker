import cv2
import torch
import os
import sys
import queue
import argparse
import warnings
import time
import copy
from ultralytics import YOLO
from yolo_utils import get_bbox, get_circle
from tqdm import tqdm
from datetime import datetime


###   supress warnings  ###
warnings.filterwarnings("ignore")
###################


### Parse Arguments ###
parser = argparse.ArgumentParser()

parser.add_argument("--input_video_path",
                    type=str,
                    default="videos/rally1.mp4",
                    help="path to the input video")
parser.add_argument("--output_video_path",
                    type=str,
                    default="",
                    help="path for the output video (.mp4)")
parser.add_argument("--confidence",
                    type=float,
                    default=0.2,
                    help="prediction confidence")
parser.add_argument("--show",
                    action='store_true',
                    help="watch preview")
parser.add_argument("--bbox_type",
                    type=str,
                    default='circle',
                    choices=['circle', 'box'],
                    help="form of bounding boxes")
parser.add_argument("--color",
                    type=str,
                    default='green',
                    choices=['black', 'white', 'red', 'green', 'purple',
                             'blue', 'yellow', 'cyan', 'gray', 'navy'],
                    help="color for highlighting the ball")
parser.add_argument("--no_trace",
                    action='store_true',
                    help="don't draw trajectory of the ball")

args = parser.parse_args()
input_video = args.input_video_path
output_video = args.output_video_path
confidence = args.confidence
show = args.show
bbox_type = args.bbox_type
no_trace = args.no_trace
color = args.color

if color == 'yellow':
    color = [0, 255, 255]
elif color == 'black':
    color = [0, 0, 0]
elif color == 'white':
    color = [255, 255, 255]
elif color == 'red':
    color = [0, 0, 255]
elif color == 'green':
    color = [0, 255, 0]
elif color == 'blue':
    color = [255, 0, 0]
elif color == 'cyan':
    color = [255, 255, 0]
###################


###    Start Time    ###
t1 = datetime.now()
#####################


### Capture Video ###
video_in = cv2.VideoCapture(input_video)

if (video_in.isOpened() == False):
    print("Error reading video file")
###################


### Video Writer ###
basename = os.path.basename(input_video)
extension = os.path.splitext(output_video)[1]

if output_video == "":  #
    os.makedirs('video_output', exist_ok=True)
    output_video = os.path.join(
        "video_output", 'yolo_v8_' + 'detect' + '_' + basename)
else:
    f = os.path.split(output_video)[0]
    if not os.path.isdir(f):
        os.makedirs(f)

if (extension != '.mp4') and (extension != ''):
    raise Exception(
        f"Extention for output video should be `.mp4`")

file_name = output_video
fps = video_in.get(cv2.CAP_PROP_FPS)
frame_width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
dims = (frame_width, frame_height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
nb_frames = video_in.get(cv2.CAP_PROP_FRAME_COUNT)
video_out = cv2.VideoWriter(file_name, fourcc, fps, dims)
###################


model = YOLO('best.pt')
model.conf = confidence
###################


### Trajectory of volleyball ###
q = queue.deque()  # we save the coordinates of the 7 previous frames
for i in range(0, 8):
    q.appendleft(None)
###################

pbar = tqdm(total=int(nb_frames))

### Process Video & Write Frames ###
while video_in.isOpened():

    ret, image = video_in.read()
    if not ret:
        break
    img_copy = copy.deepcopy(image)

    # Update Progress Bar
    pbar.update(1)

    pred = model.predict(image)
    bbox = get_bbox(pred)

    print(bbox)

    if bbox != (0, 0, 0, 0):
        q.appendleft(bbox)
        q.pop()
    else:
        q.appendleft(None)
        q.pop()

    ### add color, bbox and trace ###
    for i in range(0, 8):
        if q[i] is not None:

            if i == 0:  # current detection
                if bbox_type == 'box':
                    cv2.rectangle(img_copy, q[i], color, thickness=2)
                elif bbox_type == 'circle':
                    *center, r = get_circle(q[i])
                    cv2.circle(img_copy, center, r, color, 5)

            elif (i != 0) and (no_trace is False):  # past detections
                if bbox_type == 'box':
                    cv2.rectangle(img_copy, q[i], color, thickness=2)
                elif bbox_type == 'circle':
                    *center, r = get_circle(q[i])
                    try:
                        cv2.circle(img_copy, center, r-10, color, -1)
                    except:
                        cv2.circle(img_copy, center, r, color, -1)

    video_out.write(img_copy)

    if show:
        cv2.imshow('"p" - PAUSE, "Esc" - EXIT', img_copy)

    k = cv2.waitKey(1)
    if k == ord('p'):
        cv2.waitKey(-1)  # PAUSE
    if k == 27:  # ESC
        break

video_in.release()
video_out.release()
cv2.destroyAllWindows()
pbar.close()


###     End Time     ###
t2 = datetime.now()
dt = t2 - t1
###################
print(f'Video processed in - {dt.seconds/60:.2f} minutes')
