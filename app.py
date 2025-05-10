from flask import Flask, render_template, Response
import cv2
import dlib
from imutils.video import VideoStream
from imutils import face_utils
from scipy.spatial import distance as dist
import numpy as np
import time
from threading import Thread
from pygame import mixer

# Initialize Flask app
app = Flask(__name__)

# Initialize Pygame Mixer for alerts
mixer.init()
sound1 = mixer.Sound('wake_up.mp3')
sound2 = mixer.Sound('alert.mp3')

# Global variables for drowsiness detection
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 35
YAWN_THRESH = 30
YAWN_CONSEC_FRAMES = 30
COUNTER = 0
YAWN_COUNTER = 0  # Renamed from YARN_FRAME for clarity
alarm_status = False
alarm_status2 = False
saying = False # To prevent overlapping sounds, though this might need a more robust solution
fps = 0 # 添加帧率变量

# Load models
print("-> Loading the predictor and detector...")
# It's generally better to use absolute paths or paths relative to the app's root for model files
# For now, assuming they are in the same directory as app.py
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Start video stream
print("-> Starting Video Stream")
vs = VideoStream(src=0).start() # Assuming default webcam
time.sleep(2.0) # Allow camera to warm up

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = abs(top_mean[1] - low_mean[1])
    return distance

def play_alarm(sound_type):
    global saying
    if sound_type == "drowsy" and not saying:
        saying = True
        sound1.play()
        saying = False
    elif sound_type == "yawn" and not saying:
        # For yawn, we might not need the 'saying' guard if it's a different alert
        # or if we want it to be able to interrupt/play alongside the drowsy alert.
        # For simplicity, keeping a similar pattern for now.
        saying = True # This logic might need refinement based on desired alert behavior
        sound2.play()
        saying = False


def generate_frames():
    global COUNTER, YAWN_COUNTER, alarm_status, alarm_status2, saying, fps
    
    # 帧率计算相关变量
    frame_count = 0
    start_time = time.time()

    while True:
        frame = vs.read()
        frame = cv2.resize(frame, (640, 480)) # Standardized resize
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 更新帧率计算
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:  # 每秒更新一次帧率
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

        drowsy_alert = False
        yawn_alert = False

        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            eye = final_ear(shape)
            ear = eye[0]
            leftEye = eye[1]
            rightEye = eye[2]
            distance = lip_distance(shape)

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            lip = shape[48:60] # Points for lips
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)


            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not alarm_status:
                        alarm_status = True
                        # In a web app, directly starting a thread to play sound might be problematic
                        # if multiple users access it. For a dev board, it might be fine.
                        # Consider sending event to frontend or a more robust sound management.
                        thread = Thread(target=play_alarm, args=("drowsy",))
                        thread.daemon = True
                        thread.start()
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    drowsy_alert = True
            else:
                COUNTER = 0
                alarm_status = False

            if distance > YAWN_THRESH:
                YAWN_COUNTER += 1
                if YAWN_COUNTER >= YAWN_CONSEC_FRAMES:
                    if not alarm_status2:
                        alarm_status2 = True
                        thread = Thread(target=play_alarm, args=("yawn",))
                        thread.daemon = True
                        thread.start()
                    cv2.putText(frame, "YAWN ALERT!", (10, 60), # Adjusted position
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    yawn_alert = True
            else:
                YAWN_COUNTER = 0
                alarm_status2 = False

            cv2.putText(frame, f"EAR: {ear:.2f}", (frame.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"YAWN: {distance:.2f}", (frame.shape[1] - 150, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 在视频帧上显示FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
        # Encode frame as JPEG
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if not flag:
            continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    global fps, alarm_status, alarm_status2
    return {
        'fps': round(fps, 2),
        'drowsy_alert': alarm_status,
        'yawn_alert': alarm_status2
    }

if __name__ == '__main__':
    # The host '0.0.0.0' makes the server accessible from other devices on the network,
    # which is useful for a development board.
    # Debug should be False in a "production" or testing environment on a dev board for performance.
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True, use_reloader=False)

# Cleanup function (optional, might not be called on abrupt termination)
def cleanup():
    print("-> Stopping Video Stream")
    vs.stop()
    # mixer.quit() # If you want to explicitly quit mixer

# importatexit
# atexit.register(cleanup) # Register cleanup function to be called on exit

# Note: The dlib and haarcascade files need to be in the same directory as app.py,
# or paths need to be adjusted.
# Sound files (wake_up.mp3, alert.mp3) also need to be accessible. 