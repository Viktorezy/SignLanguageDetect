import cv2
from predict import predict_sign_language

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        return frame if ret else None

def gen_frames(camera, model):
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        sign = predict_sign_language(frame, model)
        # Add the predicted sign text on the frame
        cv2.putText(frame, f'Sign: {sign}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
