from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import cv2
from video_feed import VideoCamera, gen_frames
import tensorflow as tf

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = tf.keras.models.load_model('actionModel.keras')

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(VideoCamera(), model), media_type="multipart/x-mixed-replace; boundary=frame")
