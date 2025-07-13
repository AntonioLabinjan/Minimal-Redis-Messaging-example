# server.py
# to pull this: docker pull antoniolabinjan/face-rec-central_server:latest
# to run this: docker run -p 6010:6010 face-rec-central_server_6010
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import logging
from flask import Flask, request, jsonify, redirect, url_for, render_template_string
import torch
import numpy as np
import faiss
from transformers import CLIPProcessor, CLIPModel
import cv2
from datetime import datetime
import redis
import threading
from datetime import datetime, timedelta


# Logging konfiguracija
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

app = Flask(__name__)

# Inicijalizacija modela
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model.eval()

# Baza embeddinga
known_face_encodings = []
known_face_names = []
faiss_index = None

# Log i queue
detection_log = []
# Redis setup
redis_client = redis.Redis(host='localhost', port=6379, db=0)  # prilagodi port ako treba


# Threshold testiranje
thresholds_to_test = [
    0.20, 0.25, 0.30, 0.35, 0.40, 0.42, 0.45, 0.47, 0.50, 0.52, 0.55, 0.57, 0.60, 0.65, 0.70
]
threshold_stats = {th: [] for th in thresholds_to_test}


@app.route('/redis-test', methods=['GET'])
def redis_test():
    try:
        # Postavi neki ključ i vrijednost u Redis
        redis_client.set('test_key', 'Redis radi! 🚀')

        # Uzmi vrijednost nazad
        value = redis_client.get('test_key')
        if value:
            value = value.decode('utf-8')
        else:
            value = 'Nema vrijednosti'

        return jsonify({"status": "success", "value": value})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
# === Dataset i FAISS ===
def add_known_face(image_path, name):
    image = cv2.imread(image_path)
    if image is None:
        logging.warning(f"Ne mogu učitati: {image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    embedding = outputs.cpu().numpy().flatten()
    normalized = embedding / np.linalg.norm(embedding)
    known_face_encodings.append(normalized)
    known_face_names.append(name)
import os
import logging

def load_dataset():
    count = 0
    known_faces = set()
    dataset_path = "dataset"
    
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue
        for subdir in os.listdir(person_path):
            subdir_path = os.path.join(person_path, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for img_file in os.listdir(subdir_path):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                img_path = os.path.join(subdir_path, img_file)
                if os.path.isfile(img_path):
                    add_known_face(img_path, person)
                    known_faces.add(person)
                    count += 1
    
    logging.info(f"Učitano {count} poznatih lica.")
    logging.info(f"Poznata lica: {', '.join(sorted(known_faces))}")


def build_index():
    global faiss_index
    encodings_np = np.array(known_face_encodings).astype('float32')
    faiss_index = faiss.IndexFlatL2(encodings_np.shape[1])
    faiss_index.add(encodings_np)
    logging.info("FAISS indeks izgrađen.")

# === Klasifikacija ===
def classify_face(face_embedding, k=7, threshold=0.45):
    D, I = faiss_index.search(np.array([face_embedding]).astype('float32'), k)
    votes = {}
    for idx, dist in zip(I[0], D[0]):
        if dist > threshold:
            continue
        name = known_face_names[idx]
        weight = 1.0 / (dist + 1e-6)
        votes[name] = votes.get(name, 0) + weight
    if votes:
        winner = max(votes, key=votes.get)
        return winner, round(votes[winner], 2)
    return "Unknown", 0

unknown_attempts = []
intruder_alerts = []

def check_for_intruder_alert(timestamp_str, node_id, window_seconds=30, threshold_attempts=3):
    global unknown_attempts, intruder_alerts

    try:
        ts = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except:
        return

    # Dodaj trenutni pokušaj
    unknown_attempts.append({"timestamp": ts, "node_id": node_id})

    # Makni stare pokušaje
    cutoff = datetime.now() - timedelta(seconds=window_seconds)
    unknown_attempts[:] = [a for a in unknown_attempts if a["timestamp"] >= cutoff]

    if len(unknown_attempts) >= threshold_attempts:
        alert_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "count": len(unknown_attempts),
            "nodes": list(set(a["node_id"] for a in unknown_attempts))
        }
        intruder_alerts.append(alert_entry)
        logging.warning(f"[ALERT] Unknown intruder! {alert_entry}")
        unknown_attempts.clear()  # reset za novi prozor

# === Worker koji obrađuje queue ===
import redis
import json
import numpy as np

MAX_RETRIES = 3

def classify_worker():
    while True:
        try:
            result = redis_client.brpop("embedding_queue", timeout=2)
            if result is None:
                continue  # nema embeddinga trenutno

            _, message = result
            data = json.loads(message)
            embedding = np.array(data["embedding"])
            node_id = data["node_id"]
            retries = data.get("retries", 0)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            best_result = ("Unknown", 0, 0.0)  # name, votes, threshold

            for th in thresholds_to_test:
                name, score = classify_face(embedding, k=7, threshold=th)
                threshold_stats[th].append((name, score))
                if name != "Unknown" and score > best_result[1]:
                    best_result = (name, score, th)

            log_entry = {
                "timestamp": timestamp,
                "name": best_result[0],
                "votes": best_result[1],
                "node_id": node_id,
                "used_threshold": best_result[2]
            }
            detection_log.append(log_entry)

            if best_result[0] == "Unknown":
                check_for_intruder_alert(timestamp, node_id)

        except (json.JSONDecodeError, KeyError) as e:
            try:
                redis_client.lpush("embedding_dead", message)
                logging.warning(f" Nevažeći JSON ili KeyError: {e} | Poruka: {message}")
            except Exception as inner_e:
                logging.error(f"Greška kod spremanja nevažeće poruke u dead-letter: {inner_e}")

        except Exception as e:
            try:
                retries = data.get("retries", 0) if 'data' in locals() else 0
            except Exception:
                retries = 0

            if retries < MAX_RETRIES:
                try:
                    if 'data' in locals():
                        data["retries"] = retries + 1
                        redis_client.lpush("embedding_queue", json.dumps(data))
                        logging.warning(f" Retry #{retries + 1} | Node {data.get('node_id')} | Greška: {e}")
                    else:
                        logging.warning(f" Retry failed - no data variable | Greška: {e}")
                except Exception as e2:
                    logging.error(f" Retry failed. Greška: {e2}")
            else:
                try:
                    if 'data' in locals():
                        redis_client.lpush("embedding_dead", json.dumps(data))
                        logging.error(f" Previše pokušaja, šaljem u dead-letter | Poruka: {data} | Greška: {e}")
                    else:
                        logging.error(f" Dead-letter fallback failed - no data variable | Greška: {e}")
                except Exception as e2:
                    logging.error(f" Dead-letter fallback failed. Greška: {e2}")



# === Flask routes ===
@app.route("/classify", methods=["POST"])
def classify_api():
    data = request.get_json()
    embedding = np.array(data["embedding"])
    embedding /= np.linalg.norm(embedding)
    node_id = data.get("node_id", 0)

    message = json.dumps({
        "embedding": embedding.tolist(),
        "node_id": node_id,
        "retries": 0
    })
    redis_client.lpush("embedding_queue", message)

    return jsonify({"message": f"node {node_id}: embedding received"})

@app.route("/log", methods=["GET"])
def view_log():
    return jsonify(detection_log)

@app.route("/log/html")
def view_log_html():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Detekcija lica - Log</title>
        <meta charset="utf-8">
        <style>
            body { font-family: sans-serif; padding: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>Log Detekcija</h1>
        <table id="logTable">
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Ime</th>
                    <th>Score</th>
                    <th>Node ID</th>
                    <th>Threshold</th>
                </tr>
            </thead>
            <tbody></tbody>
        </table>

        <script>
            function getScoreColor(score) {
                if (score >= 6) return '#b3e5fc';
                if (score >= 4) return '#c8e6c9';
                if (score >= 2) return '#fff9c4';
                return '#ffcdd2';
            }

            async function fetchLog() {
                const res = await fetch("/log");
                const data = await res.json();
                const tbody = document.querySelector("#logTable tbody");
                tbody.innerHTML = "";
                data.forEach(entry => {
                    const row = document.createElement("tr");
                    const scoreColor = getScoreColor(entry.votes);
                    row.innerHTML = `
                        <td>${entry.timestamp}</td>
                        <td>${entry.name}</td>
                        <td style="background-color: ${scoreColor}; font-weight: bold;">${entry.votes.toFixed(2)}</td>
                        <td>${entry.node_id}</td>
                        <td>${entry.used_threshold.toFixed(2)}</td>
                    `;
                    tbody.appendChild(row);
                });
            }

            setInterval(fetchLog, 1000);
            fetchLog();
        </script>
    </body>
    </html>
    """

@app.route("/")
def home():
    return redirect(url_for("view_log_html"))


@app.route("/queue_contents", methods=["GET"])
def get_queue_contents():
    try:
        # Uzmi max 20 elemenata iz queuea (list length može biti velik)
        items = redis_client.lrange("embedding_queue", 0, 19)
        # Decode bytes u string i parse JSON
        decoded = []
        for item in items:
            try:
                decoded.append(item.decode('utf-8'))
            except Exception as e:
                decoded.append(f"<decode error: {e}>")

        return jsonify({"queue_length": redis_client.llen("embedding_queue"), "items": decoded})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/threshold_stats")
def threshold_stats_view():
    summary = {
        str(th): len([res for res in results if res[0] != "Unknown"])
        for th, results in threshold_stats.items()
    }

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Threshold Analiza</title>
        <meta charset="utf-8">
        <style>
            body { font-family: sans-serif; padding: 20px; }
            table { border-collapse: collapse; width: 50%; margin: auto; }
            th, td { border: 1px solid #ccc; padding: 10px; text-align: center; }
            th { background-color: #f2f2f2; }
            h1 { text-align: center; }
        </style>
    </head>
    <body>
        <h1>Rezultati po Thresholdu</h1>
        <table>
            <thead>
                <tr>
                    <th>Threshold</th>
                    <th>Broj pogodaka</th>
                </tr>
            </thead>
            <tbody>
                {% for threshold, count in summary.items() %}
                <tr>
                    <td>{{ threshold }}</td>
                    <td>{{ count }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </body>
    </html>
    """
    return render_template_string(html_template, summary=summary)

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "pong"}), 200

from datetime import timedelta

# da vidimo di se zadnje neki logira
@app.route("/active_nodes/html", methods=["GET"])
def active_nodes_html():
    active_threshold_seconds = 60
    now = datetime.now()

    recent_nodes = {}
    for entry in reversed(detection_log):
        try:
            entry_time = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
        except:
            continue
        if now - entry_time <= timedelta(seconds=active_threshold_seconds):
            node_id = entry["node_id"]
            if node_id not in recent_nodes:
                recent_nodes[node_id] = entry_time
        else:
            break

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Aktivni Nodesi</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            table { border-collapse: collapse; width: 50%; margin: auto; }
            th, td { border: 1px solid #ccc; padding: 10px; text-align: center; }
            th { background-color: #f2f2f2; }
            .active { background-color: #c8e6c9; }
        </style>
    </head>
    <body>
        <h1 style="text-align:center;">Aktivni Nodesi (Zadnjih {{threshold}} sekundi)</h1>
        <table>
            <thead>
                <tr>
                    <th>Node ID</th>
                    <th>Zadnje Viđen</th>
                </tr>
            </thead>
            <tbody>
    """
    for node_id, last_seen in recent_nodes.items():
        html += f"""
            <tr class="active">
                <td>{node_id}</td>
                <td>{last_seen.strftime("%Y-%m-%d %H:%M:%S")}</td>
            </tr>
        """

    if not recent_nodes:
        html += """
            <tr>
                <td colspan="2" style="color: grey;">Nema aktivnih nodova u zadnjih 60 sekundi.</td>
            </tr>
        """

    html += f"""
            </tbody>
        </table>
        <p style="text-align:center; margin-top: 20px;">Ukupno aktivnih: <b>{len(recent_nodes)}</b></p>
    </body>
    </html>
    """

    return render_template_string(html, threshold=active_threshold_seconds)


@app.route("/intruder_alerts", methods=["GET"])
def get_intruder_alerts():
    return jsonify(intruder_alerts)

@app.route("/intruder_alerts/html", methods=["GET"])
def intruder_alerts_html():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta http-equiv="refresh" content="5">
        <title>Unknown Intruder Alerts</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            table { border-collapse: collapse; width: 70%; margin: auto; }
            th, td { border: 1px solid #ccc; padding: 10px; text-align: center; }
            th { background-color: #f2f2f2; }
            .alert { background-color: #ffcdd2; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1 style="text-align:center;">Intruder Alert Log</h1>
        <table>
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Broj pokušaja</th>
                    <th>Nodeovi</th>
                </tr>
            </thead>
            <tbody>
                {% for alert in alerts %}
                <tr class="alert">
                    <td>{{ alert.timestamp }}</td>
                    <td>{{ alert.count }}</td>
                    <td>{{ alert.nodes | join(", ") }}</td>
                </tr>
                {% endfor %}
                {% if alerts|length == 0 %}
                <tr>
                    <td colspan="3" style="color: grey;">Nema intruder alertova zabilježeno.</td>
                </tr>
                {% endif %}
            </tbody>
        </table>
    </body>
    </html>
    """
    return render_template_string(html, alerts=intruder_alerts)


@app.route("/reload_dataset", methods=["GET"])
def reload_dataset():
    global faiss_index, known_face_encodings, known_face_names

    known_face_encodings.clear()
    known_face_names.clear()
    detection_log.clear()

    logging.info("Ponovno učitavanje dataset-a...")
    load_dataset()

    logging.info("Ponovno gradnja FAISS indeksa...")
    build_index()

    return "Dataset i indeks su uspješno osvježeni!"

if __name__ == "__main__":
    known_face_encodings.clear()
    known_face_names.clear()

    logging.info("Učitavanje poznatih lica...")
    load_dataset()

    logging.info("Gradnja FAISS indeksa...")
    build_index()

    logging.info("Server pokrenut na portu 6010.")
    worker_thread = threading.Thread(target=classify_worker, daemon=True)
    worker_thread.start()
    logging.info("Worker thread dela")
    app.run(host="0.0.0.0", port=6010, debug=True)