import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt


class VideoProcessor:

    def __init__(self, video_path, limit_output_size=None, offset_between=0):
        self.video_capture = cv2.VideoCapture(video_path)
        if not self.video_capture.isOpened():
            print("Error reading video file")
        self.limit_output_size = limit_output_size if limit_output_size is not None else self.video_capture.get(
            cv2.CAP_PROP_FRAME_COUNT)
        self.offset_between = offset_between
        self.make_frames()

    def make_frames(self):
        '''
        Split the video into frames
        '''
        self.frames = []
        iter = 0
        offset = self.offset_between
        pbar = tqdm(total=int(self.limit_output_size),
                    bar_format='Processing: {desc}{percentage:3.0f}%|{bar:10}')
        while self.video_capture.isOpened() and iter < self.limit_output_size:
            try:
                ret, frame = self.video_capture.read()
                if offset == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.frames.append(rgb)
                    iter += 1
                    offset = self.offset_between
                else:
                    offset -= 1
                pbar.update(1)
            except ValueError as err:
                print('Frame loading failed : ', err)
                break
        self.video_capture.release()
        pbar.close()
        print('Frames processing completed')

    def show_frame(self, index):
        plt.imshow(self.frames[index])

    def show_video(self):
        for frame in self.frames:
            cv2.imshow('"p" - PAUSE, "Esc" - EXIT', frame)

            k = cv2.waitKey(1000)
            if k == ord('p'):
                cv2.waitKey(-1)  # PAUSE
            if k == 27:  # ESC
                break

        cv2.destroyAllWindows()
