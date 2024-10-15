import lancedb
import sys

num_queries = int(sys.argv[1])
db = lancedb.connect("/lancedb_data/lancedb.db")
collection = db.open_table("embeddings")

for i in range(num_queries):
    vector = [i]*128
    result = collection.search(vector)
    print(f"Query result for vector {i}: {result}")
