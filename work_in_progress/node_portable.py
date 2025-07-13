# ako dodan još mobitel
import cv2
import numpy as np
import torch
import logging
import redis
import json
from transformers import CLIPProcessor, CLIPModel
from scipy.spatial.distance import cosine  
import mediapipe as mp
import os
import threading

# Env setup
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Redis setup
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("node.log"),
        logging.StreamHandler()
    ]
)

# Config
NODE_ID = 0
DIFF_THRESHOLD = 0.2

# Model
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model.eval()

# Face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# State
last_message = f"node {NODE_ID}: detected Unknown [--]"
last_message_lock = threading.Lock()
last_embedding = None
last_embedding_lock = threading.Lock()

def segment_face(image_rgb):
    small_rgb = cv2.resize(image_rgb, (320, 240))
    results = face_mesh.process(small_rgb)
    if not results.multi_face_landmarks:
        return None
    h_orig, w_orig = image_rgb.shape[:2]
    h_small, w_small = small_rgb.shape[:2]
    scale_x = w_orig / w_small
    scale_y = h_orig / h_small
    mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
    landmarks = results.multi_face_landmarks[0].landmark
    points = [(int(lm.x * w_small * scale_x), int(lm.y * h_small * scale_y)) for lm in landmarks]
    hull = cv2.convexHull(np.array(points))
    cv2.fillConvexPoly(mask, hull, 255)
    segmented_face = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    return segmented_face

# Camera setup
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        logging.warning("Failed to grab frame from camera.")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        if face.size == 0:
            continue

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        segmented_face = segment_face(face_rgb)
        if segmented_face is None:
            continue

        inputs = processor(images=segmented_face, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        embedding = outputs.cpu().numpy().flatten()
        embedding /= np.linalg.norm(embedding)

        send_embedding = False
        with last_embedding_lock:
            if last_embedding is None or cosine(embedding, last_embedding) > DIFF_THRESHOLD:
                send_embedding = True
                last_embedding = embedding

        if send_embedding:
            data = {
                "embedding": embedding.tolist(),
                "node_id": NODE_ID,
                "retries": 0
            }
            redis_client.lpush("embedding_queue", json.dumps(data))
            logging.info(" Embedding poslan u Redis queue.")

        # Draw
        with last_message_lock:
            display_message = last_message
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, display_message, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Distributed CV Node", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        logging.info("Prekid programa.")
        break

cap.release()
cv2.destroyAllWindows()
logging.info("Node clean shutdown.")
