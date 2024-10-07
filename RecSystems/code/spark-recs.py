import os
import zipfile
import requests
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.fpm import FPGrowth

# Step 1: Download the dataset
url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
zip_file = "ml-latest-small.zip"
extract_folder = "ml-latest-small"

# Download zip file
if not os.path.exists(zip_file):
    response = requests.get(url)
    with open(zip_file, "wb") as file:
        file.write(response.content)

# Extract the zip file
if not os.path.exists(extract_folder):
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_folder)

# Step 2: Initialize Spark session
spark = SparkSession.builder.appName("MovieLens-AssociationRuleMining").getOrCreate()

# Step 3: Load data into DataFrame
ratings_file = os.path.join(extract_folder, "ratings.csv")
movies_file = os.path.join(extract_folder, "movies.csv")

# Load the ratings data (userId, movieId, rating)
ratings_df = spark.read.csv(ratings_file, header=True, inferSchema=True)

# Step 4: Preprocess the data for FPGrowth
# Create transactions where each user is a transaction and movies are items bought together
transactions_df = ratings_df.groupBy("userId").agg(F.collect_list("movieId").alias("items"))

# Step 5: Apply FPGrowth (Association Rule Mining)
fp_growth = FPGrowth(itemsCol="items", minSupport=0.01, minConfidence=0.5)
model = fp_growth.fit(transactions_df)

# Get frequent itemsets
freq_itemsets = model.freqItemsets
freq_itemsets.show(10)

# Get association rules
rules = model.associationRules
rules.show(10)

# Example of filtering rules with lift > 2
filtered_rules = rules.filter(F.col("lift") > 2)
filtered_rules.show(10)

# Step 6: Stop Spark session
spark.stop()
