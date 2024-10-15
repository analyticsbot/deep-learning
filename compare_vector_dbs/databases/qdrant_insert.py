import qdrant_client
import sys

num_rows = int(sys.argv[1])
client = qdrant_client.QdrantClient(host="localhost", port=6333)

vectors = [[i]*128 for i in range(num_rows)]
points = [
    qdrant_client.PointStruct(id=i, vector=vectors[i], payload={"text": f"row {i}"})
    for i in range(num_rows)
]

client.upsert("test_collection", points)
print(f"Inserted {num_rows} rows into Qdrant.")
