# ball_tracker

Tracker tool to follow the ball in volleyball video tapes

Link to Google Drive data repo for the project
https://drive.google.com/drive/folders/1EhpmRTL258UyIT3EJpO_CIbpX_SZIpX9?usp=sharing

Link to the final report of our project :
https://www.overleaf.com/read/bxpgmqcxtztf

The main experiments using image processing techniques to detect and track the ball are in the `image_processing_methods.ipynb` notebook. The graphs resulting from these experiments are in the `images` folder.

The fine tuning of YOLOv8 model on a Roboflow dataset containing volleyball annotations (https://universe.roboflow.com/volleyball-tracking/volleyball-tracking/dataset/13) is done in the `yolo_v8_train.ipynb` notebook.

Finally, the annotation of videos with bounding boxes detected with our fine tuned YOLO model is done in `yolo_v8_detection.py`. The annotated videos are stored in the `video_output` directory.

Some examples of input videos that we used can be found in the `videos` repository. These videos are taken from https://vball.io/game/Volleibol_test/.
