import qdrant_client
import sys

num_queries = int(sys.argv[1])
client = qdrant_client.QdrantClient(host="localhost", port=6333)

for i in range(num_queries):
    vector = [i]*128
    results = client.search("test_collection", vector)
    print(f"Query result for vector {i}: {results}")
