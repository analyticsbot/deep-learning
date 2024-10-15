import lancedb
import os
import pyarrow as pa

# Connect to or create a LanceDB database
db_path = "/lancedb_data/lancedb.db"
if not os.path.exists(db_path):
    # Create the database
    db = lancedb.connect(db_path)
    # Define a schema
    schema = schema = pa.schema([
    pa.field("id", pa.int32()),
    pa.field("text", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), list_size=128))  # Assuming vectors of size 2
])
    db.create_table("embeddings", schema)
else:
    # Connect to the existing database
    db = lancedb.connect(db_path)

print("LanceDB is running...")
# Keep the process alive
while True:
    pass  # This keeps the container running
