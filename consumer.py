import redis
import json
import time
import random
import logging

# Redis setup
r = redis.Redis(host='localhost', port=6379, db=0)

# Logging konfiguracija
logging.basicConfig(
    filename='consumer.log',
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

print("📥 Čekam embeddinge za klasifikaciju...\n")

def fake_classify(embedding):
    if embedding[0] > 0.5:
        return "Antonio", round(random.uniform(2.0, 5.0), 2)
    else:
        return "Unknown", round(random.uniform(0.0, 1.0), 2)

MAX_RETRIES = 3
MAX_EMPTY_CYCLES = 3  # 3*timeout = 6 sekundi
empty_cycles = 0

while True:
    try:
        result = r.brpop("embedding_queue", timeout=2)

        if result is None:
            empty_cycles += 1
            print("⏳ Nema novih embeddinga... čekam dalje.")

            if empty_cycles >= MAX_EMPTY_CYCLES:
                print("🛑 Nema aktivnosti. Gasi se consumer nakon 3 prazna pokušaja.")
                logging.info("Consumer se gasi zbog neaktivnosti.")
                break

            continue

        empty_cycles = 0  # reset broja praznih pokušaja

        _, message = result
        print(f"\n📦 Primljen raw message: {message}")

        try:
            data = json.loads(message)
            embedding = data["embedding"]
            node_id = data["node_id"]
            retries = data.get("retries", 0)

            name, score = fake_classify(embedding)
            print(f"🧠 Node {node_id} → Predikcija: {name} (score: {score})")

            logging.info(f"✔️ Obrada embeddinga uspješna (Node {node_id} → {name}, Score: {score})")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Nevažeća poruka, šaljem u dead-letter queue: {e}")
            r.lpush("embedding_dead", message)
            logging.warning(f"❌ Nevažeći JSON: {e} | Sadržaj: {message}")

        except Exception as e:
            if retries < MAX_RETRIES:
                data["retries"] = retries + 1
                r.lpush("embedding_queue", json.dumps(data))
                print(f"🔁 Retry #{retries + 1} za Node {data.get('node_id')}...")
                logging.warning(f"Retry #{retries + 1} za poruku: {message} | Greška: {e}")
            else:
                r.lpush("embedding_dead", json.dumps(data))
                print(f"💀 Poruka neuspješna nakon {MAX_RETRIES} pokušaja. Premještena u dead-letter queue.")
                logging.error(f"Dead-letter: {data} | Zadnja greška: {e}")

        time.sleep(1)

    except Exception as e:
        print(f"❌ Fatalna greška u loopu: {e}")
        logging.critical(f"Fatalna greška u loopu: {e}")
