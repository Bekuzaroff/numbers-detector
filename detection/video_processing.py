import cv2 as cv


class VideoProcessing:
    def __init__(self):
        pass


    def process_video(video_path):
        cap = cv.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            cv.imshow('Video', frame)

            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
