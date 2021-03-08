
import asyncio
import json
import logging
import os
import ssl
import uuid
import numpy as np
import cv2
from aiohttp import web
from av import VideoFrame
import sys
import os
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
import tensorflow as tf
ROOT = os.path.dirname(__file__)
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier(os.path.join(ROOT, "facedectaction.xml"))
model =tf.keras.models.load_model(os.path.join(ROOT, "fer-self-made.h5"))
logger = logging.getLogger("pc")
pcs = set()
def detectFaces(img):
        # Convertinto grayscale since it works with grayscale images
        gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, 1.3, 8)
        if len(faces):
                for f in faces:
                         frame = cv2.rectangle(img, (f[0], f[1]), (f[0]+f[2], f[1]+f[3]), (0, 255, 0), 2) 
                         img = img[f[1]:f[1]+f[3], f[0]:f[0]+f[2]]      
                         img = img[32:256, 32:256]
                         img = cv2.resize(img, (imgXdim, imgYdim)) 
                         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                         image = img.reshape(48,48,1)
                         image = image.astype(np.float32)/225
                         image = np.array(image,ndmin=4)
                         pri = model.predict(image)
                         argmax = np.argmax(pri)
                         frame = cv2.putText(frame, labels[argmax], (f[0] + 5, f[1] + 15), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 0, 0) , 1)
                return frame
        else:
                return img

class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")
        img = detectFaces(img)
        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)
    recorder = MediaBlackhole()

   
    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        log_info("ICE connection state is %s", pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)
        if track.kind == "video":
            local_video = VideoTransformTrack(
                track, transform=params["video_transform"]
            )
            pc.addTrack(local_video)

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":


    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(app, access_log=None, port=8080, ssl_context=None)
