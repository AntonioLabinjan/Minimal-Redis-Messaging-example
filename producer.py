import redis
import json
import time
import random

r = redis.Redis(host='localhost', port=6379, db=0)

print("🚀 Slanje 5 embeddinga u queue...\n")

for i in range(5):
    embedding = [round(random.uniform(0, 1), 3) for _ in range(4)]
    data = {
        "embedding": embedding,
        "node_id": i % 2,
        "retries": 0
    }
    message = json.dumps(data)
    r.lpush("embedding_queue", message)
    print(f"✅ Poslan embedding #{i+1}: {embedding}")
    time.sleep(0.5)

print("🎉 Sva 5 embeddinga poslano.")
