import lancedb
import sys

num_rows = int(sys.argv[1])
db = lancedb.connect("/lancedb_data/lancedb.db")
collection = db.open_table("embeddings")

data = [{"vector": [i]*128, "text": f"row {i}"} for i in range(num_rows)]
collection.add(data)

print(f"Inserted {num_rows} rows into LanceDB.")
