# import the required libraries
import time  
import pyspark  
from pyspark.sql import SparkSession  
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col, pow
import mlflow
import mlflow.spark
import psycopg2

# Set the tracking URI (change this to your server URL)
mlflow.set_tracking_uri("http://mlflow:5000")  # Change to your MLflow tracking server URL
mlflow.set_experiment("Spark_Training")  

# create spark session
spark = SparkSession.builder.appName('recommendation').getOrCreate()

# load the datasets using pyspark
movies = spark.read.load("/sparkdata/ml-latest-small/movies.csv", format='csv', header=True)
ratings = spark.read.load('/sparkdata/ml-latest-small/ratings.csv', format='csv', header=True)
links = spark.read.load("/sparkdata/ml-latest-small/links.csv", format='csv', header=True)
tags = spark.read.load("/sparkdata/ml-latest-small/tags.csv", format='csv', header=True)
ratings = ratings.withColumn("userId", col("userId").cast("int"))
ratings = ratings.withColumn("movieId", col("movieId").cast("int"))
ratings = ratings.withColumn("rating", col("rating").cast("float"))

# show loaded ratings data
ratings.show()

# split the data into train, validation and test sets
train, validation, test = ratings.randomSplit([0.6, 0.2, 0.2], seed=0)
print("The number of ratings in each set: {}, {}, {}".format(train.count(), validation.count(), test.count()))

# RMSE calculation
def RMSE(predictions):
    squared_diff = predictions.withColumn("squared_diff", pow(col("rating") - col("prediction"), 2))
    mse = squared_diff.selectExpr("mean(squared_diff) as mse").first().mse
    return mse ** 0.5

# Grid Search implementation
def GridSearch(train, valid, num_iterations, reg_param, n_factors):
    min_rmse = float('inf')
    best_n = -1
    best_reg = 0
    best_model = None
    # run Grid Search for all the parameters defined in the range in a loop
    for n in n_factors:
        for reg in reg_param:
            als = ALS(rank=n, 
                      maxIter=num_iterations, 
                      seed=0, 
                      regParam=reg,
                      userCol="userId", 
                      itemCol="movieId", 
                      ratingCol="rating", 
                      coldStartStrategy="drop")            
            model = als.fit(train)
            predictions = model.transform(valid)
            rmse = RMSE(predictions)     
            print('{} latent factors and regularization = {}: validation RMSE is {}'.format(n, reg, rmse))
            # track the best model using RMSE
            if rmse < min_rmse:
                min_rmse = rmse
                best_n = n
                best_reg = reg
                best_model = model
    
    pred = best_model.transform(train)
    train_rmse = RMSE(pred)
    # best model and its metrics
    print('\nThe best model has {} latent factors and regularization = {}:'.format(best_n, best_reg))
    print('Training RMSE is {}; validation RMSE is {}'.format(train_rmse, min_rmse))
    return best_model

# build the model using different ranges for Grid Search
num_iterations = 10
ranks = [6, 8, 10, 12]
reg_params = [0.05, 0.1, 0.2, 0.4, 0.8]

# Start MLflow run to log parameters and metrics
with mlflow.start_run():
    start_time = time.time()
    
    # Log parameters
    mlflow.log_param("iterations", num_iterations)
    mlflow.log_param("ranks", ranks)
    mlflow.log_param("reg_params", reg_params)
    
    # Run grid search
    final_model = GridSearch(train, validation, num_iterations, reg_params, ranks)
    
    # Log final model
    mlflow.spark.log_model(final_model, "ALSModel")

    print('Total Runtime: {:.2f} seconds'.format(time.time() - start_time))

    # Test the accuracy of the model on the test set using RMSE
    pred_test = final_model.transform(test)
    test_rmse = RMSE(pred_test)
    
    # Log test RMSE
    mlflow.log_metric("test_rmse", test_rmse)
    
    print('The testing RMSE is ' + str(test_rmse))

# Save the model in the required format
final_model.save("/sparkdata/models/als_model")
mlflow.spark.log_model(final_model, "/sparkdata/models/als_model")


# Generate Top 5 Recommendations for each user
user_recs = final_model.recommendForAllUsers(5)
user_recs.show(truncate=False)

# Convert recommendations into a more readable format
recommendations = user_recs.withColumn("recommendations", col("recommendations.movieId"))

# Set up PostgreSQL connection and write recommendations to DB
recommendations.write \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://postgres:5432/airflow") \
    .option("dbtable", "recommendations") \
    .option("user", "airflow") \
    .option("password", "airflow") \
    .option("driver", "org.postgresql.Driver") \
    .mode("overwrite") \
    .save()
