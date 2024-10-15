import streamlit as st
import subprocess
import pandas as pd
import json
import os

# Load existing databases info
if os.path.exists("stored_db_info.json"):
    with open("stored_db_info.json", "r") as f:
        stored_dbs = json.load(f)
else:
    stored_dbs = {}

st.title("Vector Database Performance Tester")

# Show available databases
st.subheader("Available Databases:")
if stored_dbs:
    for db_name, db_info in stored_dbs.items():
        st.write(f"- {db_name}: {db_info['docker_hub_path']}")
else:
    st.write("No databases available.")

# Add a new database
st.subheader("Add a New Database")
new_db_name = st.text_input("Database Name")
docker_hub_path = st.text_input("Docker Hub Path for the Database")
insert_script = st.text_area("Insert Script (Paste Python code)")
query_script = st.text_area("Query Script (Paste Python code)")

if st.button("Add Database"):
    # Store new DB info locally
    stored_dbs[new_db_name] = {
        "docker_hub_path": docker_hub_path,
        "insert_script": insert_script,
        "query_script": query_script
    }

    # Save the scripts to files
    with open(f"databases/{new_db_name}_insert.py", "w") as f:
        f.write(insert_script)
    with open(f"databases/{new_db_name}_query.py", "w") as f:
        f.write(query_script)

    # Save the DB info
    with open("stored_db_info.json", "w") as f:
        json.dump(stored_dbs, f)
    st.success(f"Database {new_db_name} added successfully!")

# Test Insert or Query
st.subheader("Test Insert or Query")
operation = st.selectbox("Choose Operation", ("Insert", "Query"))
db_choice = st.selectbox("Select Database", list(stored_dbs.keys()))
num_rows = st.number_input("Number of Rows", min_value=100, max_value=1000000, step=100)

if st.button(f"Run {operation}"):
    if operation == "Insert":
        # Run insert script for the chosen database
        subprocess.run(["python3", f"databases/{db_choice}_insert.py", str(num_rows)])
    elif operation == "Query":
        # Run query script for the chosen database
        subprocess.run(["python3", f"databases/{db_choice}_query.py", str(num_rows)])

    st.write(f"{operation} operation completed for {db_choice}.")
