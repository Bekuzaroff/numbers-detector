import cv2 as cv


class VideoProcessing:
    def __init__(self):
        pass
    
    def track_video(self, model, video_path):
        for result in model.track(video_path, persist=True):
            cv.imshow("Tracking", result.plot())
            if cv.waitKey(10) & 0xFF == 27:  # ESC для выхода
                break
