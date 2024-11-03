from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_list, col, array_distinct, size
from pyspark.ml.fpm import FPGrowth

# Start Spark session
spark = SparkSession.builder \
    .appName("AssociationRuleLearning") \
    .getOrCreate()

# Load dataset
# Replace 'groceries.csv' with the path to your dataset
data = spark.read.csv('../dataset/Groceries_dataset.csv', header=True)

# Prepare data for FP-Growth
# Group items by transaction, and remove duplicates by converting items to a set
transactions = data.groupBy("Member_number").agg(collect_list("itemDescription").alias("items"))

# Remove duplicates in each transaction and filter out empty transactions
transactions = transactions \
    .withColumn("items", array_distinct(col("items"))) \
    .filter(size(col("items")) > 0)

# Create FP-Growth model
fp_growth = FPGrowth(itemsCol="items", minSupport=0.01, minConfidence=0.2)

# Fit the model
model = fp_growth.fit(transactions)

# Display frequent itemsets
print("Frequent Itemsets:")
model.freqItemsets.show(truncate=False)

# +---------------------------------------------+----+
# |items                                        |freq|
# +---------------------------------------------+----+
# |[specialty cheese]                           |71  |
# |[chocolate marshmallow]                      |60  |
# |[pet care]                                   |85  |
# |[pet care, rolls/buns]                       |40  |
# |[pet care, other vegetables]                 |40  |
# |[house keeping products]                     |45  |
# |[flower (seeds)]                             |67  |
# |[curd]                                       |471 |
# |[curd, sausage]                              |125 |
# |[curd, sausage, rolls/buns]                  |59  |
# |[curd, sausage, rolls/buns, whole milk]      |39  |
# |[curd, sausage, yogurt]                      |58  |
# |[curd, sausage, yogurt, whole milk]          |39  |
# |[curd, sausage, other vegetables]            |58  |
# |[curd, sausage, other vegetables, whole milk]|40  |
# |[curd, sausage, soda]                        |52  |
# |[curd, sausage, whole milk]                  |74  |
# |[curd, frankfurter]                          |77  |
# |[curd, frankfurter, rolls/buns]              |45  |
# |[curd, frankfurter, whole milk]              |48  |
# +---------------------------------------------+----+
# only showing top 20 rows

# Display association rules
print("Association Rules:")
model.associationRules.show(truncate=False)

# +----------------------------------------+------------------+-------------------+------------------+--------------------+
# |antecedent                              |consequent        |confidence         |lift              |support             |
# +----------------------------------------+------------------+-------------------+------------------+--------------------+
# |[bottled beer, rolls/buns, whole milk]  |[sausage]         |0.3221476510067114 |1.5638001788594782|0.012314007183170857|
# |[bottled beer, rolls/buns, whole milk]  |[other vegetables]|0.5033557046979866 |1.3365671232375693|0.019240636223704463|
# |[bottled beer, rolls/buns, whole milk]  |[tropical fruit]  |0.3288590604026846 |1.4071269126780073|0.012570548999486916|
# |[bottled beer, rolls/buns, whole milk]  |[yogurt]          |0.3624161073825503 |1.2807778663437726|0.013853258081067214|
# |[bottled beer, rolls/buns, whole milk]  |[bottled water]   |0.2751677852348993 |1.287639888170033 |0.01051821446895844 |
# |[bottled beer, rolls/buns, whole milk]  |[soda]            |0.3288590604026846 |1.0490119619064358|0.012570548999486916|
# |[bottled beer, rolls/buns, whole milk]  |[root vegetables] |0.28187919463087246|1.222208120880023 |0.0107747562852745  |
# |[frankfurter, rolls/buns, whole milk]   |[other vegetables]|0.525              |1.3940395095367848|0.01616213442791175 |
# |[frankfurter, rolls/buns, whole milk]   |[yogurt]          |0.35               |1.2368993653671805|0.0107747562852745  |
# |[frankfurter, rolls/buns, whole milk]   |[bottled water]   |0.325              |1.520828331332533 |0.010005130836326322|
# |[frankfurter, rolls/buns, whole milk]   |[soda]            |0.425              |1.3556873977086743|0.013083632632119035|
# |[frankfurter, rolls/buns, whole milk]   |[root vegetables] |0.3416666666666667 |1.4814423433444568|0.01051821446895844 |
# |[bottled beer, yogurt, other vegetables]|[whole milk]      |0.6385542168674698 |1.3936642426368406|0.013596716264751155|
# |[margarine, soda]                       |[sausage]         |0.2468354430379747 |1.198212399703643 |0.010005130836326322|
# |[margarine, soda]                       |[yogurt]          |0.3227848101265823 |1.1407209337021014|0.013083632632119035|
# |[margarine, soda]                       |[bottled water]   |0.25316455696202533|1.1846764022064522|0.01026167265264238 |
# |[margarine, soda]                       |[rolls/buns]      |0.43670886075949367|1.2489296692887062|0.017701385325808106|
# |[margarine, soda]                       |[other vegetables]|0.44936708860759494|1.1932104301038182|0.018214468958440224|
# |[margarine, soda]                       |[whole milk]      |0.5379746835443038 |1.174146313804981 |0.02180605438686506 |
# |[margarine, soda]                       |[shopping bags]   |0.25949367088607594|1.5419303797468356|0.01051821446895844 |
# +----------------------------------------+------------------+-------------------+------------------+--------------------+
# only showing top 20 rows

# Stop the Spark session
spark.stop()

## To run the code
# docker run -it --rm \
#   -v $(pwd)/dataset:/dataset \
#   -v $(pwd)/notebooks/association_rule_learning_pyspark.py:/app/association_rule_learning_pyspark.py \
#   bitnami/spark:latest /bin/bash

# spark-submit association_rule_learning_pyspark.py