## A. Feature Engineering
#### 1. Bucketizing
Bucketizing, in the context of machine learning, refers to the process of dividing a continuous feature or variable into discrete intervals, or "buckets." It is often used when dealing with continuous numerical features but can also be applied to categorical features in certain scenarios.

For categorical features, bucketizing typically involves grouping similar categories together to reduce the dimensionality or complexity of the feature. This is useful when you have many distinct categories, and combining related ones into broader groups can make the model more generalizable or easier to train.

For example:

Original Categorical Feature: Types of fruits (apple, banana, orange, grape, mango)
Bucketized Feature: Grouped into "Citrus" (orange) and "Non-citrus" (apple, banana, grape, mango)
In the case of continuous features (like age or income), bucketizing involves converting ranges of values into categories (e.g., grouping ages 0-20, 21-40, etc.).

Bucketizing simplifies data, but it should be used carefully to avoid losing important nuances.

#### 2. Feature Hashing ####

Feature hashing, or hashing trick, converts text data, or categorical attributes with high cardinalities, into a feature vector of arbitrary dimensionality. In some AdTech companies (Twitter, Pinterest, etc.), it’s not uncommon for a model to have thousands of raw features.


In AdTech, consider a model that predicts user click behavior based on categorical features like "ad campaign ID" or "user location." These features can have thousands of unique values, which would make one-hot encoding inefficient. Feature hashing compresses these high-cardinality categorical features into a fixed-length feature vector by applying a hash function, allowing the model to handle a large number of features efficiently without exploding the feature space. For example, "ad campaign ID" might be hashed into a vector of length 1,000 instead of having individual binary columns for each ID.

##### Feature Hashing Example with Python and `sklearn`

This example demonstrates how to use **feature hashing** (also known as the hashing trick) with categorical data in Python using `sklearn`'s `FeatureHasher`. Feature hashing is useful when dealing with high cardinality categorical features, such as those found in AdTech companies (Twitter, Pinterest, etc.).

##### Example Data

We have the following sample data containing categorical features like `ad_campaign_id`, `user_location`, and `device`:

```python
from sklearn.feature_extraction import FeatureHasher

# Sample data: each row is a dictionary of categorical features
data = [
    {'ad_campaign_id': 'campaign_1', 'user_location': 'NY', 'device': 'mobile'},
    {'ad_campaign_id': 'campaign_2', 'user_location': 'CA', 'device': 'desktop'},
    {'ad_campaign_id': 'campaign_3', 'user_location': 'TX', 'device': 'tablet'},
    {'ad_campaign_id': 'campaign_1', 'user_location': 'NY', 'device': 'desktop'},
]

# Feature hashing: converts categorical features into a fixed-size feature vector
hasher = FeatureHasher(n_features=10, input_type='dict')  # Set to 10 features for simplicity
hashed_features = hasher.transform(data)

# Convert to an array for easier visualization
hashed_array = hashed_features.toarray()

# Display the hashed feature matrix
print(hashed_array)
```

**Explanation**
The data consists of 3 categorical features: ad_campaign_id, user_location, and device.
We use FeatureHasher to convert these categorical features into a fixed-length feature vector of size 10 (n_features=10).
The hashed features reduce the memory usage compared to one-hot encoding while preserving some information about the original features.

**Output**
```
The hashed feature matrix will look like this:
[[ 0.  1. -1.  0.  1.  0.  0.  0. -1.  0.]
 [ 1.  0.  1.  1.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  0.  1.  0.  0.  0.  1.  0.  0.]
 [ 0.  2.  0.  0.  1.  0.  0.  0. -1.  0.]]
```
Each row represents the hashed features for a row of categorical data, which can be used as input to a machine learning model.

**Notes**

FeatureHasher is particularly useful when working with datasets that have high cardinality categorical features. By converting categorical features into fixed-length vectors, you can reduce the memory footprint and still retain useful information for machine learning models.

| **Aspect**             | **One-Hot Encoding**                                                                                 | **Feature Hashing**                                                                                  |
|-------------------------|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| **Memory Usage**       | High memory usage for high-cardinality features due to large sparse matrix.                         | Fixed memory usage regardless of cardinality, making it efficient for large datasets.               |
| **Handling New Categories** | Fails if unseen categories are encountered during inference.                                       | Seamlessly handles new categories by hashing them into the predefined space.                        |
| **Hash Collisions**    | No collisions; unique mapping ensures no information loss.                                          | May introduce noise due to hash collisions where different categories map to the same index.        |
| **Model Performance**  | Precise and interpretable, works well for small to medium cardinality with models like logistic regression. | Slightly noisy due to collisions but works well for high cardinality, especially with robust models. |
| **Training Speed**     | Slower due to high dimensionality for large cardinality.                                            | Faster due to reduced dimensionality.                                                              |
| **Best Use Case**      | Small to medium cardinality features for interpretable models.                                      | High-cardinality features or large-scale datasets.                                                 |

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Simulate data
np.random.seed(42)
ad_ids = [f"ad_{np.random.randint(1, 1000000)}" for _ in range(1000)]
labels = np.random.randint(0, 2, size=1000)

# Split data
X_train, X_test, y_train, y_test = train_test_split(ad_ids, labels, test_size=0.2, random_state=42)

# One-Hot Encoding
ohe = OneHotEncoder(handle_unknown='ignore')
X_train_ohe = ohe.fit_transform(np.array(X_train).reshape(-1, 1))
X_test_ohe = ohe.transform(np.array(X_test).reshape(-1, 1))

# Feature Hashing
hv = HashingVectorizer(n_features=100, alternate_sign=False)
X_train_hash = hv.transform(X_train)
X_test_hash = hv.transform(X_test)

# Logistic Regression
lr_ohe = LogisticRegression(max_iter=1000).fit(X_train_ohe, y_train)
lr_hash = LogisticRegression(max_iter=1000).fit(X_train_hash, y_train)

# Calculate accuracies
ohe_accuracy = lr_ohe.score(X_test_ohe, y_test)
hash_accuracy = lr_hash.score(X_test_hash, y_test)

ohe_accuracy, hash_accuracy
```

The accuracies for the logistic regression models using different encoding techniques are as follows:

- One-Hot Encoding Accuracy: 50.0%
- Feature Hashing Accuracy: 55.5%

This shows that for this dataset, the model using feature hashing performed slightly better than the one using one-hot encoding.


##### Cross Feature and Hashing Trick Example in Python

This example demonstrates **cross features** and how we can use the **hashing trick** to manage high-dimensional categorical data in Python using `sklearn`.

##### What is a Cross Feature?

A **cross feature** is simply a new feature created by combining two or more categorical features. For example, if we have the Uber pickup data containing `latitude` and `longitude` of locations, we can combine them (or cross them) to create a new feature that represents the pickup location more uniquely.

- Let's say we have two categorical features, `latitude` and `longitude`.
- If `latitude` has 1,000 possible values and `longitude` has 1,000 possible values, their cross feature would have 1,000 × 1,000 = 1,000,000 possible values. This makes the data very high-dimensional.

To handle this large number of possible combinations (or high-dimensional data), we use the **hashing trick**, which reduces the number of dimensions while still preserving useful information.

##### Example: Predict Uber Demand Using Cross Features

Let's assume we have some Uber pickup data containing `latitude` and `longitude` information:

```python
from sklearn.feature_extraction import FeatureHasher

# Sample data: latitude and longitude are categorical features
data = [
    {'latitude': 'lat_40', 'longitude': 'long_73'},
    {'latitude': 'lat_41', 'longitude': 'long_74'},
    {'latitude': 'lat_42', 'longitude': 'long_75'},
    {'latitude': 'lat_40', 'longitude': 'long_73'},
]

# Cross features: combine latitude and longitude for each record
cross_features = [{'pickup_location': f"{d['latitude']}_{d['longitude']}"} for d in data]

# Apply FeatureHasher to reduce the dimensions of the cross features
hasher = FeatureHasher(n_features=10, input_type='dict')  # Using 10 dimensions
hashed_features = hasher.transform(cross_features)

# Convert to an array for easier visualization
hashed_array = hashed_features.toarray()

# Display the hashed feature matrix
print(hashed_array)
```

**Explanation**
Cross Features: We combine the latitude and longitude values into a single feature called pickup_location.
Feature Hashing: Since this cross feature could have a large number of possible values (high cardinality), we use the hashing trick to convert it into a fixed-size feature vector of length 10 (n_features=10).
This helps reduce the complexity while keeping enough information to feed into a machine learning model.

**Output**
This will print a hashed feature matrix like this:
```[[ 0.  1. -1.  0.  1.  0.  0.  0. -1.  0.]
 [ 1.  0.  1.  1.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  0.  1.  0.  0.  0.  1.  0.  0.]
 [ 0.  2.  0.  0.  1.  0.  0.  0. -1.  0.]]
```

##### Why Use the Hashing Trick here?
Cross features often create many new categories when you combine existing features, making the data too large or "high-dimensional." Using the hashing trick helps keep the data manageable while still using the important information in your machine learning model.


#### 3. Scaling/Normalization:
- Min-Max Scaling: Rescales features to a range (usually [0, 1]).
- Standardization (Z-score Normalization): Rescales features to have zero mean and unit variance.
- Log Transformation: Helps reduce skewness in data by applying the log function.
- Square Root / Cube Root: Similar to log transformation, helps with skewness for positive data.
- Power Transformation: Helps stabilize variance and make the data more Gaussian-like.

#### 4. Feature Encoding
- Label Encoding: Ordinal categories like education levels.  

Example:

Input:  
| Education Level |  
|-----------------|  
| High School     |  
| Bachelor's      |  
| Master's        |  

Output:  
| Education Level | Encoded Value |  
|-----------------|---------------|  
| High School     | 0             |  
| Bachelor's      | 1             |  
| Master's        | 2             |  


- One-Hot Encoding : Nominal categories like colors.  

Example:

Input:  
| Color   |  
|---------|  
| Red     |  
| Green   |  
| Blue    |  

Output:  
| Color   | Red | Green | Blue |  
|---------|-----|-------|------|  
| Red     | 1   | 0     | 0    |  
| Green   | 0   | 1     | 0    |  
| Blue    | 0   | 0     | 1    |  



- Binary Encoding: High-cardinality categories like city names.  

Example:

Input:  
| City       |  
|------------|  
| New York   |  
| San Diego  |  
| Los Angeles|  

Output:  
| City        | Binary Encoding |  
|-------------|-----------------|  
| New York    | 0001            |  
| San Diego   | 0010            |  
| Los Angeles | 0011            |  

---

- Target Encoding: Categorical data correlated with a target variable (e.g., predicting income).  

Example:

Input:  
| Job Title     | Target (Income) |  
|---------------|-----------------|  
| Engineer      | 80,000          |  
| Teacher       | 50,000          |  
| Engineer      | 90,000          |  
| Teacher       | 45,000          |  

Output:  
| Job Title     | Encoded Value   |  
|---------------|-----------------|  
| Engineer      | 85,000          |  
| Teacher       | 47,500          |  

---

- Frequency Encoding: Encoding by the frequency of occurrence in the dataset.  

Example:

Input:  
| Product    |  
|------------|  
| Apple      |  
| Banana     |  
| Apple      |  
| Orange     |  
| Banana     |  

Output:  
| Product    | Frequency Encoding |  
|------------|--------------------|  
| Apple      | 2                  |  
| Banana     | 2                  |  
| Orange     | 1                  |  

#### 5. Handling Missing Data
- Imputation: Filling missing values using strategies such as:
- Mean/Median/Mode Imputation: Replacing missing values with the column’s mean, median, or mode.
- K-Nearest Neighbors (KNN) Imputation: Predict missing values based on the closest neighbors.
- MICE (Multiple Imputation by Chained Equations): Imputing missing data multiple times and averaging the results.
- Forward/Backward Fill: Using preceding or following values for imputation (for time series data).

#### 6. Binning
- Discretization/Binning: Convert continuous features into discrete bins.
- Equal-Width Binning: Divides the data range into equal-width bins.
- Equal-Frequency Binning: Divides data so that each bin has an equal number of observations.
- Quantile Binning: Bins based on quantiles to ensure the distribution of values is uniform across bins.

#### 7. Polynomial Features

- Polynomial Expansion: Create new features by combining existing ones using polynomial combinations.

Generate new features like:

* $x_1^2$
* $x_1 \cdot x_2$
* ...

This technique can help capture non-linear relationships between features and the target variable.

#### 8. Interaction Features
- Interaction Terms: Create new features by multiplying or interacting features with each other.
Example: If you have age and income, you might create a feature age * income.

#### 9. Feature Extraction
- Principal Component Analysis (PCA): Reduces dimensionality by projecting data onto the principal components.
- Linear Discriminant Analysis (LDA): A supervised dimensionality reduction technique used to maximize class separability.
- t-SNE/UMAP: Non-linear techniques for dimensionality reduction and visualization.

#### 10. Datetime Features
- Extract Date/Time Features: Extract meaningful features from datetime data, such as:
    - Day, Month, Year
    - Day of Week, Day of Year
    - Hour, Minute
    - Is Weekend, Weekday, Is Holiday


#### 11. Domain-Specific Transformations
- Text Data:

    - Bag of Words (BoW): Convert text into a frequency matrix of words.
    - TF-IDF: Weighs terms by their importance in the document and reduces the impact of commonly used words.
    - Word Embeddings: Dense vector representations (e.g., Word2Vec, GloVe).
    - N-grams: Capture sequences of n words to extract context from text.
    - Sentence Embeddings

- Image Data:

    - Resizing/Scaling: Adjust image size to a common resolution.
    - Data Augmentation: Apply transformations like rotation, flipping, cropping, etc.
    - Color Histogram: Extract color distribution as a feature.
    - Image Embedding


#### 12. Outlier Detection and Treatment
- Clipping: Limit the range of values by capping outliers at a maximum or minimum threshold.
- Winsorizing: Replace extreme values with percentile values to reduce the impact of outliers.

#### 13. Aggregation
- Aggregating Features: Combine or aggregate features at different levels.
    -  Example: Sum, mean, count, max, min operations applied to group data.


## B. Feature Selection
- Filter Methods: Select features based on statistical tests (e.g., correlation, chi-square test, ANOVA).
- Wrapper Methods: Use models like forward selection, backward elimination, or recursive feature elimination (RFE).
- Embedded Methods: Feature selection during model training (e.g., Lasso, Ridge, tree-based models like Random Forest).

Some of the methods available in sklearn

##### Filter Methods
1. **Variance Threshold**
   - Removes features with low variance.
   - Example: `sklearn.feature_selection.VarianceThreshold`

2. **SelectKBest**
   - Selects the top k features based on a scoring function.
   - Example: `sklearn.feature_selection.SelectKBest`

3. **SelectPercentile**
   - Selects features based on the top percentile of the highest scores.
   - Example: `sklearn.feature_selection.SelectPercentile`

##### Wrapper Methods
4. **Recursive Feature Elimination (RFE)**
   - Recursively removes features and builds a model to identify important features.
   - Example: `sklearn.feature_selection.RFE`

5. **RFECV**
   - RFE combined with cross-validation to select the best number of features.
   - Example: `sklearn.feature_selection.RFECV`

##### Embedded Methods
6. **Lasso (L1 Regularization)**
   - Uses L1 regularization to shrink coefficients of less important features to zero.
   - Example: `sklearn.linear_model.Lasso`

7. **Tree-Based Feature Selection**
   - Uses tree-based estimators to determine feature importance.
   - Example: `sklearn.ensemble.RandomForestClassifier` with `.feature_importances_`

8. **SelectFromModel**
   - Selects features based on the importance weights from an estimator.
   - Example: `sklearn.feature_selection.SelectFromModel`

##### Feature Selection for Sparse Data
9. **chi2**
   - Selects features using the Chi-squared statistic for non-negative features.
   - Example: `sklearn.feature_selection.chi2`

10. **mutual_info_classif / mutual_info_regression**
    - Selects features based on mutual information for classification or regression tasks.
    - Example: `sklearn.feature_selection.mutual_info_classif`


## C. Word Embeddings
#### CBOW and Skip-gram Models in Word2Vec

Word2Vec is a popular technique used to create word embeddings, where words are represented as dense vectors in a continuous vector space. There are two primary models used in Word2Vec: **CBOW (Continuous Bag of Words)** and **Skip-gram**.

#### 1. CBOW (Continuous Bag of Words)

- **Goal**: Predict the target word using the context words (neighboring words).
- In this model, the context is used to predict the center word.
  
For example, in the sentence "The cat sits on the mat," if "cat" is the target word, the context might be ["The", "sits"].

##### CBOW Python Example

```python
import numpy as np
from collections import defaultdict

# Example sentence
sentences = [["the", "cat", "sits", "on", "the", "mat"]]

# Vocabulary and word to index mapping
vocab = {word for sentence in sentences for word in sentence}
word2idx = {word: i for i, word in enumerate(vocab)}
idx2word = {i: word for word, i in word2idx.items()}

# Parameters
window_size = 2  # Context window size (before and after the target word)
embedding_dim = 10  # Embedding size

# Generate training data (CBOW style: context -> target)
def generate_cbow_data(sentences, window_size):
    data = []
    for sentence in sentences:
        for i, word in enumerate(sentence):
            target = word2idx[word]
            context = []
            for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
                if i != j:
                    context.append(word2idx[sentence[j]])
            data.append((context, target))
    return data

training_data = generate_cbow_data(sentences, window_size)
print(f"Training data (CBOW): {training_data}")
```


The output represents the context-target pairs used in CBOW training:
```Training data (CBOW): [([word1_idx, word3_idx], target_word_idx), ...]```

#### 2. Skip-gram Model
Goal: Predict the context words (neighboring words) given the target word.
In this model, the center word is used to predict its context.
For example, in the sentence "The cat sits on the mat," if "cat" is the target word, the model tries to predict its neighboring words "The" and "sits." based on a variable called window size.

##### Skip-gram Python Example

```python
# Generate training data (Skip-gram style: target -> context)
def generate_skipgram_data(sentences, window_size):
    data = []
    for sentence in sentences:
        for i, word in enumerate(sentence):
            target = word2idx[word]
            for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
                if i != j:
                    context = word2idx[sentence[j]]
                    data.append((target, context))
    return data

training_data_skipgram = generate_skipgram_data(sentences, window_size)
print(f"Training data (Skip-gram): {training_data_skipgram}")
The output represents the target-context pairs used in Skip-gram training:

```
```Training data (Skip-gram): [(target_word_idx, context_word_idx), ...]```

#### How Do CBOW and Skip-gram Differ?
CBOW: Predicts the center word using context words.
Skip-gram: Predicts context words using the center word.

#### 3. GloVe (Global Vectors for Word Representation)
- **How it Works:**
    - Uses word co-occurrence statistics across a corpus to capture relationships between words.
    - Constructs a word-word co-occurrence matrix and factorizes it to find word embeddings.
    - Balances local (contextual) and global statistics for better word representation.
- **Use Case:** Efficient for finding semantic similarities and analogies.

#### 3. FastText

- **How it Works:**
    - Extends Word2Vec by breaking words into character n-grams.
    - Captures subword information, allowing it to represent rare and out-of-vocabulary words.
    - Models can learn embeddings even for words not in the training corpus.
- **Use Case:** Suitable for morphologically rich languages and handling misspelled words.

#### 4. ELMo (Embeddings from Language Models)

- **How it Works:**
    - Uses deep bidirectional LSTMs to generate context-dependent embeddings.
    - Word embeddings depend on the sentence context, capturing polysemy.
    - Pretrained on large corpora and fine-tuned for specific tasks.
- **Use Case:** Effective for NLP tasks requiring contextual understanding.

#### 5. BERT (Bidirectional Encoder Representations from Transformers)

- **How it Works:**
    - Uses transformer architecture to generate contextual embeddings.
    - Words are represented differently based on their context in a sentence.
    - Pretrained on large corpora with unsupervised objectives like masked language modeling.
- **Use Case:** Best for NLP tasks like classification, question answering, and named entity recognition.

#### 6. TF-IDF (Term Frequency-Inverse Document Frequency)

- **How it Works:**
    - Computes weights for words based on their frequency in a document and their rarity across the corpus.
    - Does not produce dense embeddings but is useful for sparse representation of text.
- **Use Case:** Useful for simpler NLP tasks or feature extraction for traditional ML models.

#### 7. Doc2Vec

- **How it Works:**
    - Extends Word2Vec to learn embeddings for entire documents or paragraphs instead of individual words.
    - Uses distributed memory (DM) and distributed bag of words (DBOW) frameworks.
- **Use Case:** Suitable for document similarity and classification tasks.

#### 8. Transformer-Based Models (GPT, RoBERTa, T5)

- **How it Works:**
    - Use self-attention mechanisms to model context across entire sequences.
    - Generate embeddings at the token, sentence, or document level.
    - Pretrained on massive datasets with unsupervised tasks and fine-tuned for downstream tasks.
- **Use Case:** Cutting-edge performance for complex NLP applications.

#### 9. CoVe (Contextualized Word Vectors)

- **How it Works:**
    - Uses pre-trained sequence-to-sequence models to generate embeddings.
    - Encodes semantic context and polysemy by training on translation tasks.
- **Use Case:** Provides context-sensitive embeddings for tasks like sentiment analysis.

#### 10. Latent Semantic Analysis (LSA)
- **How it Works:**
    - Reduces dimensionality of term-document matrices using singular value decomposition (SVD).
    - Captures latent semantic relationships between words.
- **Use Case:** Effective for smaller datasets and simple semantic tasks.


## D. Handling Imbalanced Class Distribution in Multi-Class Problems
In machine learning tasks like fraud detection, click prediction, or spam detection, it's common to have imbalanced labels. For example, in ad click prediction, you might have a 0.2% conversion rate, meaning out of 1,000 clicks, only two lead to a desired action. This imbalance can cause the model to focus too much on learning from the majority class.

Sometimes we can use under or oversampling or use SMOTE, but when dealing with multi-class problems, methods like SMOTE (Synthetic Minority Over-sampling Technique) are not always effective. Below are some strategies to handle class imbalance in multi-class settings:

#### 1. Class Weights in Loss Function
Adjusting class weights in the loss function allows the model to give more importance to the minority classes.

How it works:
```loss = - (w0 * y * log(p)) - (w1 * (1 - y) * log(1 - p))```

Effect: Helps the model focus on minority classes and reduces bias toward the majority class.

#### 2. Oversampling and Undersampling
a. Random Oversampling
Random oversampling duplicates instances from minority classes to balance the dataset.

```python
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()
X_res, y_res = ros.fit_resample(X, y)

```

b. Random Undersampling
Random undersampling reduces the number of majority class samples.

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler()
X_res, y_res = rus.fit_resample(X, y)
```

#### 3. Hybrid Approach: Combining Oversampling and Undersampling
A combination of oversampling for minority classes and undersampling for majority classes.

#### 4. Multi-Class Variants of SMOTE
There are several multi-class variants of SMOTE, like SMOTE-ENN and Borderline-SMOTE.

a. SMOTE-ENN
This combines SMOTE with Edited Nearest Neighbors to clean noisy samples.

```python
from imblearn.combine import SMOTEENN

smote_enn = SMOTEENN()
X_res, y_res = smote_enn.fit_resample(X, y)
```

b. Borderline-SMOTE
This method focuses on synthesizing samples specifically for borderline minority instances.

#### 5. Ensemble Methods
a. Balanced Random Forest
Balanced Random Forest undersamples the majority class at each bootstrap iteration, creating balanced datasets for each tree in the forest.

```python
from imblearn.ensemble import BalancedRandomForestClassifier

clf = BalancedRandomForestClassifier()
clf.fit(X, y)
```
b. EasyEnsemble
EasyEnsemble creates multiple balanced subsets from the original dataset using undersampling.

```python
from imblearn.ensemble import EasyEnsembleClassifier

clf = EasyEnsembleClassifier()
clf.fit(X, y)
```
#### 6. Data Augmentation
Data augmentation can help generate more samples for minority classes, especially useful for image, text, or time-series data.

These methods can help tackle the challenge of class imbalance in multi-class machine learning tasks.

#### 7. Focal Loss for Imbalanced Data

Focal Loss is a modification of standard cross-entropy loss designed to address class imbalance in datasets. It focuses on "hard-to-classify" examples while reducing the weight of "easy-to-classify" ones, thus preventing the model from being overwhelmed by the majority class.

#### Formula:
$ FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$

Where:
- $ p_t $: Predicted probability for the correct class.
- $ \alpha_t $: Balancing factor for class weights (optional).
- $ \gamma $: Focusing parameter that controls the down-weighting of easy examples.

#### Key Benefits:
1. **Handles Class Imbalance**: By reducing the loss contribution of well-classified examples, it allows the model to focus on minority or difficult cases.
2. **Customizable**: The hyperparameter $ \gamma $ can be tuned to adjust the focus on hard examples $ \gamma = 2 $ is commonly used.
3. **Improves Performance**: Particularly effective in scenarios like object detection and classification with highly imbalanced data.

## E. Regression Loss Functions

#### 1. Mean Square Error (MSE)
#### Description:
MSE calculates the average of the squared differences between the predicted and actual values.

#### Advantage:
- Penalizes larger errors more than smaller ones due to the square term.
- Smooth and differentiable, making it useful for gradient-based optimization.

#### Disadvantage:
- Sensitive to outliers since errors are squared, making large errors dominate the loss.

#### Best Suited For:
- Data without extreme outliers or when you want to penalize larger errors more.

#### Example:
```python
# Example data
y_true = [2, 3, 4]
y_pred = [2.5, 3.5, 3]

mse = np.mean((np.array(y_true) - np.array(y_pred))**2)
```

#### 2. Mean Absolute Error (MAE)
#### Description:
MAE calculates the average of the absolute differences between predicted and actual values.

#### Advantage:
Less sensitive to outliers compared to MSE.
Directly represents the average error in the same units as the output.

#### Disadvantage:
The loss is not differentiable at zero, making it less suitable for some optimization algorithms.

#### Best Suited For:
Data with outliers or where you want errors to contribute linearly.

#### Example:
```python
mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
```

#### 3. Huber Loss
#### Description:
Huber loss is a combination of MSE and MAE. It behaves like MAE for large errors and MSE for smaller errors.

#### Advantage:
Robust to outliers while still penalizing small errors like MSE.

#### Disadvantage:
Requires tuning a hyperparameter (δ) to switch between MSE and MAE behavior.

#### Best Suited For:
Data with some outliers, but where smaller errors still need to be penalized effectively.

#### Choosing a Good Value for Delta (δ) in Huber Loss

The value of **δ (delta)** in Huber Loss determines the threshold where the loss transitions from **quadratic** (like MSE) to **linear** (like MAE). Selecting δ depends on the scale of your data and the presence of outliers.

#### Guidelines for Choosing δ:
1. **Default Starting Point**:  
   - A typical default value is **1.0**, which works well for normalized data or when the target variable has a small scale.

2. **Based on Data Scale**:  
   - Set $ \delta $ to match the scale of residuals, such as the standard deviation  $\sigma $ of residuals:
     $\delta = \sigma$

3. **Outlier Sensitivity**:  
   - Smaller δ: More sensitive to small residuals, behaves like MSE.  
   - Larger δ: More robust to outliers, behaves like MAE.

4. **Hyperparameter Tuning**:  
   - Experiment with different δ values to optimize performance on validation data.

#### Practical Example:
- For residuals in the range \([-10, 10]\), start with δ values between **1 and 5**.
- For larger residuals, increase δ proportionally.

#### Summary:
- Begin with **δ = 1.0**, and adjust based on data scale or through tuning to achieve the best performance.


#### Example:
```python
delta = 1.0
huber_loss = np.mean(np.where(np.abs(y_true - y_pred) <= delta, 
                              0.5 * (y_true - y_pred)**2, 
                              delta * (np.abs(y_true - y_pred) - 0.5 * delta)))
```

#### 4. Quantile Loss

#### Description:
Quantile loss minimizes over- or under-estimation based on a quantile (τ). The loss penalizes differently based on whether the prediction is above or below the true value.

#### Advantage:
Useful for predicting conditional quantiles instead of the mean.

#### Disadvantage:
Requires setting a quantile τ, which needs domain knowledge or experimentation.

#### Best Suited For:
Asymmetric prediction intervals or for use in probabilistic forecasting.

#### Example:
```python
tau = 0.9
quantile_loss = np.mean(np.maximum(tau * (y_true - y_pred), (tau - 1) * (y_true - y_pred)))
```

#### 5. Mean Absolute Percentage Error (MAPE)
#### Description:
MAPE calculates the average percentage error between predicted and true values.

#### Advantage:
Scale-independent, making it useful for comparing across datasets with different ranges.

#### Disadvantage:
Sensitive to small values in the true labels, which can inflate the error.

#### Best Suited For:
Data where you care more about percentage errors than absolute differences.

#### Example:
```python
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

#### 6. Symmetric Absolute Percentage Error (sMAPE)
#### Description:
sMAPE adjusts MAPE to be symmetric, considering both over- and under- predictions equally.

#### Advantage:
More balanced compared to MAPE, especially for large over- or under-predictions.

#### Disadvantage:
Still sensitive to small values in the denominator.

#### Best Suited For:
Forecasting data, especially time-series, where you want a balanced error metric.

#### Example:
```python
smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
```

## F. CLassifcation Loss Functions

### 1. Binary Classification Loss Functions

#### 1. Focal Loss
#### Description:
Focal loss is designed to down-weight easy examples and focus on learning from hard, misclassified examples.

$$\text{Focal Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ \alpha(1 - p_i)^{\gamma} y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$

#### Advantage:
Useful in imbalanced datasets by focusing on harder-to-classify examples.

#### Disadvantage:
Requires tuning a focusing parameter (γ), which adds complexity.

#### Best Suited For:
Imbalanced regression or classification tasks where you want to focus on hard examples.

####  Example:

```python
gamma = 2.0
focal_loss = np.mean(((1 - np.abs(y_true - y_pred))**gamma) * (y_true - y_pred)**2)
```

**Use Cases:**
- Useful for addressing class imbalance by down-weighting well-classified examples.
- Often applied in object detection tasks.

**Avoid When:**
- When the dataset is balanced, as it may over-penalize easy examples.

#### 2. Hinge Loss
#### Description:
Hinge loss is commonly used for "maximum-margin" classification, such as in support vector machines. It penalizes predictions that are on the wrong side of the margin.

$$
\text{Hinge Loss} = \frac{1}{N} \sum_{i=1}^{N} \max(0, 1 - y_i f(x_i))
$$

#### Advantage:
Ensures that not just the correct label, but the margin is optimized.

#### Disadvantage:
Not suitable for regression tasks directly.

#### Best Suited For:
Classification tasks where margin-based optimization is important, such as support vector machines (SVMs).

#### Example:
```python
hinge_loss = np.mean(np.maximum(0, 1 - y_true * y_pred))
```

   **Use Cases:**
- Mainly used in Support Vector Machines (SVMs) for binary classification.

**Avoid When:**
- In cases where probabilistic output is desired, as it does not provide a probability.


#### 3. **Log Loss (Binary Cross-Entropy Loss)**

   **Formula:**
   $$
    \text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
    $$

   **Use Cases:**
   - Commonly used for binary classification problems.
   - Effective when class distributions are balanced.

   **Avoid When:**
   - Class imbalance is significant, as it may lead to misleading loss values.


### 2. Multi-Class Classification Loss Functions

#### 1. **Categorical Cross-Entropy Loss**

   **Formula:**
    $$
    \text{Categorical Cross-Entropy} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(p_{i,c})
    $$

   **Use Cases:**
   - Used for multi-class classification tasks where classes are one-hot encoded.
   - Effective when class distributions are balanced.

   **Avoid When:**
   - When the target classes are not mutually exclusive, as it assumes that classes are one-hot encoded.

#### 2. **Sparse Categorical Cross-Entropy Loss**

   **Formula:**
    $$
    \text{Sparse Categorical Cross-Entropy} = -\frac{1}{N} \sum_{i=1}^{N} \log(p_{i,y_i})
    $$

   **Use Cases:**
   - Similar to categorical cross-entropy but used when the target classes are provided as integer labels (not one-hot encoded).

   **Avoid When:**
   - When one-hot encoding is required for compatibility with certain models.

#### 3. **Kullback-Leibler Divergence (KL Divergence)**

   **Formula:**
    $$
    D_{KL}(P || Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)}
    $$

   **Use Cases:**
   - Useful for measuring how one probability distribution diverges from a second expected probability distribution.

   KL Divergence is used in multi-class classification when:

1. **Probability Distributions are Compared**:  
   - It measures the difference between two probability distributions: the **true distribution** (ground truth) and the **predicted distribution** from the model.

2. **Soft Labels or Probabilistic Targets**:  
   - Commonly used when the ground truth is not a one-hot encoded vector but a **probability distribution** over classes (e.g., in **knowledge distillation** or label smoothing).

3. **Output of the Model is a Probability Distribution**:  
   - Typically applied when the model uses a **softmax** activation function to produce class probabilities.

4. **Applications**:
   - **Knowledge Distillation**: Aligning the predicted distribution of a student model to that of a teacher model.  
   - **Regularization**: Ensuring smoother predictions in cases like label smoothing.


   **Avoid When:**
   - When actual class probabilities are not known or are very small, leading to instability in computation.

#### 4. **Normalized Cross-Entropy Loss**

   **Formula:**
    $$
    \text{Normalized Cross-Entropy} = -\frac{1}{N} \sum_{i=1}^{N} \left( \sum_{c=1}^{C} \frac{y_{i,c}}{\sum_{c'} y_{i,c'}} \log(p_{i,c}) \right)
    $$

   **Use Cases:**
   - Useful when class frequencies vary significantly, normalizing the contribution of each class.

   #### Key Applications:
1. **Multi-Class Classification**:
   - Widely used in tasks like text classification, image recognition, and other scenarios where class probabilities are modeled.

2. **Language Modeling**:
   - Optimizing large-scale models like Word2Vec or neural language models to handle large vocabulary sizes efficiently.

3. **Imbalanced Datasets**:
   - Helps mitigate the impact of imbalanced class distributions by incorporating normalization factors.

4. **Information Retrieval**:
   - Used in ranking tasks where probabilities of relevant items are modeled.

5. **Contrastive Learning**:
   - Applied in self-supervised learning, such as SimCLR, to maximize similarity between augmented data representations.

NCE loss is particularly effective in settings where computing the full softmax distribution is computationally expensive or unnecessary.

   **Avoid When:**
   - When the normalization may obscure the learning of relevant features.

#### 5. **Hinge Loss (Multi-Class)**

   **Formula:**
    $$
    \text{Multi-Class Hinge Loss} = \sum_{i=1}^{N} \sum_{j \neq y_i} \max(0, 1 - f(x_i, y_i) + f(x_i, j))
    $$

   **Use Cases:**
   - Applied in multi-class SVMs and problems where margin maximization is desired.

   **Avoid When:**
   - When probabilistic interpretations of the output are needed.

#### 6. **Triplet Loss**

   **Formula:**
    $$
    \text{Triplet Loss} = \max(0, d(a, p) - d(a, n) + \alpha)
    $$
   where \(d\) is the distance function, \(a\) is the anchor, \(p\) is the positive example, \(n\) is the negative example, and \(\alpha\) is a margin.

   **Use Cases:**
   - Useful in tasks involving similarity learning, such as face recognition.

   **Avoid When:**
   - When training data does not provide adequate positive/negative pairs. Good quality negative pairs are also needed.

## F. Evaluation Metrics 

#### 1. Area Under the Curve (AUC)
#### Description:
AUC measures the ability of a model to discriminate between positive and negative classes. It is calculated from the Receiver Operating Characteristic (ROC) curve, which plots the true positive rate against the false positive rate at various threshold settings.

#### Advantage:
- Provides a single metric to evaluate model performance across all classification thresholds.
- Intuitive interpretation as the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance.

#### Disadvantage:
- Can be misleading if the class distribution is highly imbalanced.
- Does not provide insight into the model’s performance at specific thresholds.

#### Best Suited For:
- Binary classification problems where understanding the trade-off between true and false positive rates is important.

#### Example:
```python
from sklearn.metrics import roc_auc_score

y_true = [0, 0, 1, 1]  # Ground truth (0: negative, 1: positive)
y_scores = [0.1, 0.4, 0.35, 0.8]  # Predicted probabilities

auc = roc_auc_score(y_true, y_scores)
```

#### 2. Mean Average Recall at K (MAR@K)
##### Description:
MAR@K measures the average recall of a model at the top K retrieved items. It is particularly useful in scenarios where only the top K results are considered relevant.

##### Advantage:
- Focuses on the most relevant items, making it suitable for recommendation systems and information retrieval tasks.
- Provides a clearer picture of recall when only a subset of results is of interest.

##### Disadvantage:
- May overlook relevant items that are not in the top K.
- Sensitive to the choice of K; different K values can yield different insights.

##### Best Suited For:
Scenarios where retrieving the top K relevant items is more important than retrieving all relevant items.

##### Example:
```python
def average_recall_at_k(y_true, y_pred, k):
    relevant_items = sum(y_true)
    retrieved_items = y_pred[:k]
    true_positives = sum([1 for i in range(k) if retrieved_items[i] == 1])
    return true_positives / relevant_items if relevant_items > 0 else 0

y_true = [0, 1, 1, 0, 1]  # Ground truth
y_pred = [1, 1, 0, 1, 0]  # Predicted top K items

mar_k = average_recall_at_k(y_true, y_pred, k=3)
```
##### Difference Between Recall and Recall@K

##### **Recall**:
- **Definition**: Recall measures the proportion of actual positive instances that the model correctly identified.
- **Formula**:  
  $$
  \text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}
  $$
- **Example**:
  Suppose we have 10 items, and 4 of them are relevant (ground truth). The model identifies 6 items as relevant, of which 3 are correct. Recall is:
 $$
  \text{Recall} = \frac{3}{4} = 0.75
  $$

##### **Recall@K**:
- **Definition**: Recall@K measures how many relevant items are retrieved in the **top K predictions**.
- **Example**:
  Let's assume:
  - There are 10 total items.
  - 4 items are relevant (ground truth).
  - The model produces a ranked list of items:  
    **[1, 2, 5, 3, 8, 7, 4, 6, 9, 10]**  
    (Items 1, 3, 4, 5 are relevant).

##### Case 1: Recall@3
- Look at the **top 3 predictions**:  
  **[1, 2, 5]**
- Relevant items in the top 3: **[1, 5]**
- Recall@3 = $$\frac{\text{Relevant items in top 3}}{\text{Total relevant items}}$$:  
  $$
  \text{Recall@3} = \frac{2}{4} = 0.5
  $$

##### Case 2: Recall@5
- Look at the **top 5 predictions**:  
  **[1, 2, 5, 3, 8]**
- Relevant items in the top 5: **[1, 3, 5]**
- Recall@5 = $$\frac{\text{Relevant items in top 5}}{\text{Total relevant items}}$$:  
  $$
  \text{Recall@5} = \frac{3}{4} = 0.75
  $$

##### Key Difference:
- **Recall** evaluates performance across all predictions, focusing on the overall ability to retrieve relevant items.
- **Recall@K** evaluates performance within the **top K predictions**, emphasizing ranking quality and relevance for top results (critical in recommender systems or search engines).


#### 3. Mean Average Precision (MAP)
##### Description:
MAP is the mean of average precision scores across multiple queries. Average precision summarizes the precision-recall curve for a single query and considers the order of predicted results.

##### Advantage:
Considers both precision and the rank of positive instances, providing a nuanced evaluation.
Useful for evaluating ranked retrieval tasks.
##### Disadvantage:
Requires careful computation, as it involves precision at each relevant item in the ranking.
May be sensitive to the number of queries in the dataset.

##### Best Suited For:
Information retrieval tasks where both the order and relevance of items matter.

##### Example:
```python
def average_precision(y_true, y_scores):
    sorted_indices = np.argsort(y_scores)[::-1]  # Sort scores in descending order
    y_true_sorted = [y_true[i] for i in sorted_indices]
    
    precision_scores = [np.sum(y_true_sorted[:k]) / k for k in range(1, len(y_true_sorted) + 1)]
    ap = np.mean([precision_scores[k - 1] for k in range(1, len(y_true_sorted) + 1) if y_true_sorted[k - 1] == 1])
    return ap

y_true = [0, 1, 1, 0, 1]  # Ground truth
y_scores = [0.1, 0.4, 0.35, 0.8, 0.3]  # Predicted scores

map_score = average_precision(y_true, y_scores)
```
##### Difference between MaP, AP, and Precision
- **Definition**: MAP is the mean of the Average Precision (AP) scores for multiple queries or predictions. 
- **Precision**: Measures the proportion of retrieved items that are relevant.
  $$
  \text{Precision} = \frac{\text{Relevant items retrieved}}{\text{Total items retrieved}}
  $$
- **Average Precision (AP)**: Captures precision at each rank where a relevant item is retrieved, averaged over all relevant items.


##### Example: Precision vs. MAP

##### Setup:
- **Ground Truth (relevant items)**:  
  For a query, the relevant items are: **[1, 3, 5]**  
- **Predictions (ranked list)**:  
  The model predicts: **[1, 2, 3, 4, 5]**


##### **Step 1: Precision at Each Rank**
| Rank (k) | Item Predicted | Relevant? | Precision@k |
|----------|----------------|-----------|-------------|
| 1        | 1              | ✅         | \( 1/1 = 1.0 \)   |
| 2        | 2              | ❌         | \( 1/2 = 0.5 \)   |
| 3        | 3              | ✅         | \( 2/3 = 0.67 \)  |
| 4        | 4              | ❌         | \( 2/4 = 0.5 \)   |
| 5        | 5              | ✅         | \( 3/5 = 0.6 \)   |


##### **Step 2: Average Precision (AP)**
- Precision is calculated **only at ranks where relevant items are retrieved**:
  - Precision@1 = \( 1.0 \)  
  - Precision@3 = \( 0.67 \)  
  - Precision@5 = \( 0.6 \)
- Average Precision (AP):
  $$
  \text{AP} = \frac{\text{Sum of Precision at relevant ranks}}{\text{Total relevant items}} = \frac{1.0 + 0.67 + 0.6}{3} = 0.7567
  $$


##### **Step 3: MAP for Multiple Queries**
- For multiple queries, calculate AP for each query and take the mean:
  $$
  \text{MAP} = \frac{\text{Sum of APs}}{\text{Number of queries}}
  $$


##### Key Difference: Precision vs. MAP
- **Precision** focuses on a single cut-off point (e.g., top-k).
- **MAP** averages precision over ranks, emphasizing ranking quality by rewarding early retrieval of relevant items.


#### 4. Mean Reciprocal Rank (MRR)
##### Description:
MRR measures the average of the reciprocal ranks of the first relevant item across multiple queries. It emphasizes the importance of retrieving the relevant item as early as possible.

##### Advantage:
Simple to compute and interpret.
Highlights the effectiveness of retrieval systems in providing relevant results early in the ranking.
##### Disadvantage:
Only considers the first relevant item, which may not provide a comprehensive view of the retrieval system's performance.
Sensitive to cases where there are no relevant items in the ranking.
##### Best Suited For:
Tasks where finding the first relevant item quickly is crucial, such as question-answering systems.
##### Example:
```python
def mean_reciprocal_rank(queries):
    ranks = []
    for query in queries:
        rank = next((i + 1 for i, relevance in enumerate(query) if relevance), None)
        ranks.append(1 / rank if rank else 0)
    return np.mean(ranks)

queries = [[0, 0, 1], [0, 1, 0]]  # List of queries with relevance
mrr = mean_reciprocal_rank(queries)
```

#### 5. Normalized Discounted Cumulative Gain (NDCG)
##### Description:
Normalized Discounted Cumulative Gain (NDCG) is a ranking metric that is widely used in information retrieval, such as in search engines and recommendation systems. It measures the usefulness (or "gain") of the results based on their relevance, while also considering the position of the results in the ranking. Higher-ranked items (those shown earlier) are given more importance compared to lower-ranked ones.

##### Key Concepts:

1. **Cumulative Gain (CG)**: This is the sum of the relevance scores of the retrieved items, without considering their positions.
   - Formula: 
   ```math
   CG = \sum_{i=1}^{k} rel_i
   ```

2. **Discounted Cumulative Gain (DCG)**: DCG penalizes the relevance scores based on their positions. Lower-ranked items are discounted, meaning their contribution to the overall score decreases as the rank increases.

3. **Ideal DCG (IDCG)**: This is the best possible DCG that could be obtained if the items were perfectly ranked according to their relevance. This is used to normalize the DCG.

4. **NDCG**: NDCG is the ratio of DCG to IDCG. It normalizes the score so that it lies between 0 and 1, where a higher score indicates better ranking.

##### Why is NDCG Important?
NDCG is especially useful when dealing with graded relevance, where items are not simply "relevant" or "irrelevant" but have varying degrees of relevance. By considering both the relevance of items and their positions in the ranking, NDCG provides a more nuanced evaluation of the ranking quality.


##### Advantage:
- Position-sensitive: Penalizes relevant items that are placed lower in the list, encouraging better ordering of items.
- Graded relevance: Handles varying levels of relevance, unlike binary relevance metrics like Precision or Recall.
- Normalized: Scores are normalized between 0 and 1, making them comparable across queries or datasets.

##### Disadvantage:
Requires relevance scores, which may not always be available.
Complexity in implementation compared to simpler metrics.

##### Best Suited For:
Ranking tasks in information retrieval where graded relevance is available.

##### Example:
```python
def ndcg(y_true, y_scores):
    sorted_indices = np.argsort(y_scores)[::-1]
    ideal_relevance = np.sort(y_true)[::-1]
    
    dcg = np.sum([(2**y_true[i] - 1) / np.log2(i + 2) for i in range(len(y_true)) if i in sorted_indices])
    idcg = np.sum([(2**ideal_relevance[i] - 1) / np.log2(i + 2) for i in range(len(ideal_relevance))])
    return dcg / idcg if idcg > 0 else 0

y_true = [3, 2, 3, 0, 1, 2]  # Relevance scores
y_scores = [0.1, 0.4, 0.35, 0.8, 0.3, 0.2]  # Predicted scores

ndcg_score = ndcg(y_true, y_scores)
```

##### Example:

Suppose we have a list of 5 retrieved items with the following relevance scores:

- Ground truth relevance: `[3, 2, 3, 0, 1]` # these can come from things such as purchase, click, add to cart etc
- Predicted ranking scores: `[0.5, 0.4, 0.9, 0.3, 0.2]`

1. **Calculate DCG**:
$$
DCG_5 = 3 + \frac{2}{\log_2(2+1)} + \frac{3}{\log_2(3+1)} + \frac{0}{\log_2(4+1)} + \frac{1}{\log_2(5+1)}
$$
$$
DCG_5 = 3 + \frac{2}{1.58496} + \frac{3}{2} + 0 + \frac{1}{2.58496} \approx 6.1487
$$

2. **Calculate IDCG**:
$$
IDCG_5 = 3 + \frac{3}{\log_2(2+1)} + \frac{2}{\log_2(3+1)} + \frac{1}{\log_2(4+1)} + 0
$$
$$
IDCG_5 \approx 6.27965
$$

3. **Calculate NDCG**:
$$
NDCG_5 = \frac{DCG_5}{IDCG_5} = \frac{6.1487}{6.27965} \approx 0.9792
$$

In this case, the NDCG score is approximately **0.979**, indicating a very well-ranked list.

#### 6. Cumulative Gain (CG)
##### Description:
Cumulative Gain measures the total relevance score of the retrieved items, regardless of their rank. It sums the relevance scores of the top K results.

##### Advantage:
Simple and intuitive to calculate, providing a straightforward measure of total relevance.
Useful for understanding overall retrieval effectiveness.
##### Disadvantage:
Ignores the rank of items, meaning it can give a false sense of performance if lower-ranked items are highly relevant.
##### Best Suited For:
Situations where the overall relevance of retrieved items is more important than their order.
##### Example:
```python
def cumulative_gain(y_true, k):
    return np.sum(y_true[:k])

y_true = [3, 2, 3, 0, 1, 2]  # Relevance scores
cg_score = cumulative_gain(y_true, k=3)
```

##### Normalized Discounted Cumulative Gain (NDCG) Based on Click Data

**Click data** can be used as a proxy for ground truth relevance scores, especially when explicit relevance labels (e.g., ratings or user feedback) are unavailable. In recommendation systems, clicks indicate user interest, with more clicks suggesting higher relevance. Here’s how click data can be transformed and used as ground truth (GT) relevance:

##### Transforming Click Data into Ground Truth (GT) Relevance:
- **Clicks as binary relevance**: If a user clicks on an item, it can be labeled as relevant (1), while non-clicked items are considered irrelevant (0).
- **Clicks as graded relevance**: You can assign higher relevance scores based on the number of clicks or interactions with an item. For instance:
  - 3+ clicks = highly relevant (relevance score 3)
  - 2 clicks = relevant (relevance score 2)
  - 1 click = somewhat relevant (relevance score 1)
  - 0 clicks = irrelevant (relevance score 0)

##### Example: Calculating NDCG Based on Click Data

Let's assume we have a set of 5 items and the following click data:

- **Ground truth relevance** (based on click data): `[3, 0, 2, 0, 1]`
- **Model-predicted scores**: `[0.9, 0.7, 0.6, 0.4, 0.2]`

**1. Calculate DCG:**

$$
DCG_5 = \frac{2^3 - 1}{\log_2(1+1)} + \frac{2^0 - 1}{\log_2(2+1)} + \frac{2^2 - 1}{\log_2(3+1)} + \frac{2^0 - 1}{\log_2(4+1)} + \frac{2^1 - 1}{\log_2(5+1)}
$$

$$
DCG_5 \approx 7 + 0 + 1.5 + 0 + 0.387 \approx 8.887
$$

**2. Calculate IDCG:**

$$
IDCG_5 = \frac{2^3 - 1}{\log_2(1+1)} + \frac{2^2 - 1}{\log_2(2+1)} + \frac{2^1 - 1}{\log_2(3+1)} + \frac{2^0 - 1}{\log_2(4+1)} + \frac{2^0 - 1}{\log_2(5+1)}
$$

$$
IDCG_5 \approx 7 + 1.892 + 0.5 + 0 + 0 = 9.392
$$

**3. Calculate NDCG:**

$$
NDCG_5 = \frac{DCG_5}{IDCG_5} = \frac{8.887}{9.392} \approx 0.946
$$

##### Interpretation:
- The NDCG score is approximately **0.946**, indicating that the ranking generated by the model is close to the ideal ranking.
- This score suggests that the model’s predictions align well with the ground truth relevance derived from click data.

##### Use of Click Data as Ground Truth:
1. **Advantages**:
   - **Implicit feedback**: Click data is automatically collected, so there's no need to rely on explicit feedback (like ratings or reviews).
   - **Reflects user interest**: Clicks represent user interactions and interest, making them a strong signal of relevance in many cases.

2. **Challenges**:
   - **Noisy signals**: Clicks may not always represent true interest or relevance (e.g., accidental clicks).
   - **Cold start problem**: Users with no click history or new items may not have any click data, making it difficult to assess relevance.

#### 7. Metrics for Imbalanced data in Classification

When working with imbalanced data, where the distribution of classes is uneven, traditional metrics like accuracy can be misleading. In such cases, here are the recommended metrics to evaluate model performance more effectively:

##### 1. Precision, Recall, and F1 Score

- Precision: Measures the accuracy of positive predictions, especially useful when false positives (incorrectly predicting the minority class) need to be minimized.

$Precision = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$
 
- Recall (Sensitivity/True Positive Rate): Measures how well the model captures positive instances, which is critical when false negatives (missing the minority class) are costly.

$Recall = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$
 
- F1 Score: The harmonic mean of precision and recall, which balances them. It's useful for cases where both false positives and false negatives need to be minimized.

$F1 Score = 2 * \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
​
 
##### 2. Area Under the ROC Curve (AUC-ROC)
- ROC Curve: Plots the true positive rate (recall) against the false positive rate at various threshold settings. AUC-ROC measures the model’s ability to distinguish between classes.
- AUC-ROC: A higher AUC-ROC (closer to 1) means better performance. It’s particularly effective for binary classification tasks and gives an overview of how well the model differentiates between classes across threshold values.

##### 3. Area Under the Precision-Recall Curve (AUC-PR)
- This metric is often more informative for imbalanced data than AUC-ROC, especially when the positive class is rare. It shows the trade-off between precision and recall for different thresholds.
- AUC-PR is especially valuable when the focus is on the minority class performance, as it focuses more on precision and recall.

The Precision-Recall Curve (PR Curve) visually represents the trade-off between precision and recall at various thresholds for a binary classifier. Here's the explanation:

##### Axes:
- **X-axis**: Recall (True Positives / (True Positives + False Negatives)).
- **Y-axis**: Precision (True Positives / (True Positives + False Positives)).

##### Interpretation:
- A high area under the curve (AUC-PR) indicates better performance, especially for imbalanced datasets where the positive class is rare.
- The curve highlights how well the model balances precision and recall across different thresholds.

##### Example Insight:
- The AUC-PR value quantifies the model's average performance for different thresholds.
- For example, at higher recall values, precision might drop due to more false positives being included, which is visually evident from the slope of the curve.

##### Why AUC-PR?
- In scenarios with imbalanced data, AUC-PR is more informative than AUC-ROC, as it emphasizes the performance on the minority class by focusing directly on precision and recall rather than true negatives, which are abundant in imbalanced settings.


##### 4. Specificity (True Negative Rate)
- Specificity measures how well the model avoids false positives. This is particularly useful when a high precision is crucial, such as in fraud detection, where predicting fraud incorrectly could lead to significant costs.

$Specificity = \frac{\text{True Negatives}}{\text{True Negatives} + \text{False Positives}}$
​
 
##### 5. Balanced Accuracy
- Balanced Accuracy: The average of recall across classes. It’s calculated as:

$Balanced Accuracy = \frac{2 \times \text{Recall of Positive Class} + \text{Recall of Negative Class}}{2}$
​
 
This metric is helpful when dealing with highly imbalanced datasets, as it weighs each class equally and gives a fairer assessment of model performance.
##### 6. Cohen’s Kappa

- Cohen’s Kappa compares the observed accuracy with the expected accuracy (random chance) and is robust in the case of class imbalance. Kappa values closer to 1 indicate a high level of agreement, while values closer to 0 indicate that the performance is near random.

##### 7. Geometric Mean (G-Mean)
- The G-Mean is the square root of the product of recall for each class. It’s used to measure the balance between classification performance on both the majority and minority classes. This metric penalizes models that perform well only on the majority class but not the minority class.

$G-Mean = \sqrt{\text{Recall of Positive Class} \times \text{Recall of Negative Class}}$
 
##### 8. Matthews Correlation Coefficient (MCC)
- MCC takes into account true and false positives and negatives, and is considered a balanced measure even if the classes are of very different sizes. It’s particularly useful for binary classification with imbalanced data:

​
 $MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$


## G. Sampling Techniques

In statistical analysis and machine learning, sampling techniques are used to select a subset of data from a larger population. These techniques allow for efficient computation and generalization. Below are some common sampling techniques, with examples where applicable.

#### 1. Random Sampling

##### Description:
Random Sampling is the simplest sampling technique where each data point in the population has an equal chance of being selected. It helps to ensure that the sample represents the population without bias.

##### Example:
Suppose you have a dataset of 1000 customer transactions. To randomly select 100 transactions for analysis:

```python
import random

# Data of 1000 customer transactions
transactions = list(range(1000))

# Randomly selecting 100 transactions
random_sample = random.sample(transactions, 100)
```

This method is unbiased and works well when the population is homogeneous.

##### Advantages:
- Easy to implement
- Unbiased if the population is uniform
##### Disadvantages:
- May not work well for non-homogeneous populations
- Sample may not represent smaller subgroups effectively

#### 2. Rejection Sampling
##### Description:
Rejection Sampling is a technique where samples are drawn from a proposal distribution, and then accepted or rejected based on how well they fit the target distribution. It is commonly used in probabilistic models and Monte Carlo simulations.

##### Example:
Consider a scenario where you want to sample from a target distribution P(x) but only have access to a simpler proposal distribution Q(x). You generate samples from Q(x) and accept them with probability 𝑃 ( 𝑥 ) / 𝑀 𝑄 ( 𝑥 ), where 𝑀 is a constant.

```python
import random

def target_distribution(x):
    return 0.5 * x  # Example target distribution

def proposal_distribution():
    return random.uniform(0, 2)  # Uniform proposal distribution

samples = []
for _ in range(1000):
    x = proposal_distribution()
    acceptance_prob = target_distribution(x) / 1  # Assume M=1
    if random.uniform(0, 1) < acceptance_prob:
        samples.append(x)
```

##### Advantages:
- Effective for complex distributions
- Flexible and adaptable to various target distributions
##### Disadvantages:
- Inefficient if many samples are rejected
- Requires a well-designed proposal distribution

#### 3. Weight Sampling (Weighted Random Sampling)
##### Description:
Weight Sampling involves selecting samples based on their assigned weights, giving more importance to some data points over others. Each data point has a probability proportional to its weight.

##### Example:
Suppose you have a list of items with corresponding weights:

```python
import random

items = ['A', 'B', 'C', 'D']
weights = [0.1, 0.3, 0.5, 0.1]

####  Select 1 item with weight-based probability
weighted_sample = random.choices(items, weights, k=1)
```

##### Advantages:
- Useful when some data points are more important than others
- Reduces bias toward less important data points
##### Disadvantages:
- Requires accurate weighting of data
- Weight assignment may be subjective

#### 4. Importance Sampling
##### Description:
Importance Sampling is a variance reduction technique used in Monte Carlo simulations. It involves drawing samples from a different (usually easier) distribution and adjusting for the difference by weighting the samples. The goal is to estimate properties of a distribution while sampling from a simpler distribution.

##### Example:
Let's estimate the mean of a target distribution P(x), using a proposal distribution Q(x):

```python
import numpy as np

def target_distribution(x):
    return np.exp(-x)  # Example target distribution (exponential decay)

def proposal_distribution():
    return np.random.normal(0, 2)  # Normal distribution as proposal

weights = []
samples = []

# Importance sampling
for _ in range(1000):
    x = proposal_distribution()
    w = target_distribution(x) / np.random.normal(0, 2)  # Weight adjustment
    samples.append(x)
    weights.append(w)

# Weighted mean estimate
estimate = np.average(samples, weights=weights)
```
##### Advantages:
- Reduces variance in estimates
- More efficient than brute-force sampling
##### Disadvantages:
- Choosing an appropriate proposal distribution is challenging
- Can lead to high variance if the weights vary significantly

#### 5. Stratified Sampling
##### Description:
Stratified Sampling involves dividing the population into distinct subgroups (strata) and sampling from each stratum proportionally. This ensures that each subgroup is adequately represented in the sample.

##### Example:
Suppose you have a population of students, divided into 3 strata based on grade levels: Grade A, Grade B, and Grade C. You want to ensure that each grade level is represented in your sample.

```python
import random

# Strata with different populations
grade_A = list(range(50))
grade_B = list(range(50, 150))
grade_C = list(range(150, 250))

# Sample proportionally from each stratum
sample_A = random.sample(grade_A, 5)
sample_B = random.sample(grade_B, 10)
sample_C = random.sample(grade_C, 10)

# Combine the stratified samples
stratified_sample = sample_A + sample_B + sample_C
```
##### Advantages:
- Ensures representation from all subgroups
- Reduces variability within each stratum
##### Disadvantages:
- Requires prior knowledge of strata
- More complex than simple random sampling

#### 6. Reservoir Sampling
##### Description:
Reservoir Sampling is used to sample a fixed number of items from a stream of data of unknown size, ensuring that each item has an equal probability of being included. It's efficient and works well with large datasets.

##### Example:
Suppose you have a data stream of unknown size and you want to select a random sample of 5 items:

```python
import random

def reservoir_sampling(stream, k):
    reservoir = []

    # Fill the reservoir with the first k items
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            # Replace items with gradually decreasing probability
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item

    return reservoir

# Simulated data stream of size 1000
stream = list(range(1000))

# Reservoir sample of size 5
reservoir_sample = reservoir_sampling(stream, 5)
```
##### Advantages:
- Works efficiently with large datasets
- No need to know the size of the data stream in advance
##### Disadvantages:
- Limited to uniform sampling
- May not work well for biased or weighted sampling needs
##### Conclusion:
Each sampling technique has its own advantages and limitations, depending on the type of data and the goal of the analysis. For simple data, random sampling may be sufficient, but for more complex datasets or streams, methods like stratified sampling and reservoir sampling may be more appropriate.

## H. A/B Testing

#### 1. Normal A/B Testing
In traditional A/B testing, two versions (A and B) of a product feature or webpage are compared to determine which performs better. A portion of users is randomly assigned to version A, and another portion is assigned to version B. Metrics such as click-through rate or conversion rate are measured, and statistical tests are used to determine which version performs best.

##### Example:
- **Version A**: Original homepage with a "Sign Up" button.
- **Version B**: New homepage with a "Get Started" button.
- If version B increases sign-ups by 10%, it may be selected as the better option.

### 2. Budget-Splitting A/B Testing
Budget-splitting A/B testing involves allocating different portions of a testing budget to multiple variants based on performance over time. Instead of splitting traffic equally, traffic is dynamically allocated to the variant that shows higher potential, maximizing returns while the test is still running.

##### Example:
- **Version A**: 40% of users (initial budget allocation).
- **Version B**: 60% of users (because it shows higher conversions after early results).
- If version B continues to outperform, more budget/traffic is allocated to it to maximize results during the test.


## I. Ranking Approaches in Machine Learning

##### Example Dataset:
We have a dataset of search results with relevance scores. Let's assume we have 3 documents (`Doc A`, `Doc B`, and `Doc C`) and their relevance scores for a given query:

| Document | Relevance Score |
|----------|-----------------|
| Doc A    | 3               |
| Doc B    | 1               |
| Doc C    | 2               |


#### 1. Point-wise Approach:
In the **point-wise approach**, each item or data point is treated independently, and the model is trained to predict a score or relevance for each item individually. The main idea is to minimize the difference between the predicted score and the actual score for each point, similar to traditional regression.

- **Example:** Predicting relevance scores for search results. Each document is given a relevance label, and the model predicts the score for each document without considering its relation to other documents.

- **Advantages:**
  - Simple and easy to implement.
  - Works well when the relevance of individual items is more important than their relative ranking.

- **Disadvantages:**
  - Does not directly optimize for ranking metrics like NDCG or MAP.
  - Ignores the relative ranking between items, which may lead to suboptimal ranking performance.

##### Training Data:
- `Doc A`: Label = 3
- `Doc B`: Label = 1
- `Doc C`: Label = 2

We train a model to predict the relevance score for each document individually.

##### Prediction:
After training, the model predicts the following relevance scores:
- `Doc A`: Predicted Score = 2.8
- `Doc B`: Predicted Score = 1.2
- `Doc C`: Predicted Score = 2.1

Based on these scores, the predicted ranking would be: `Doc A`, `Doc C`, `Doc B`.

#### 2. Pairwise Approach:
In the **pairwise approach**, the focus is on comparing pairs of items. The model is trained to predict the relative order between two items by learning which one is more relevant. Instead of predicting the absolute score, the model learns a preference between pairs of items.

- **Example:** Given two search results, A and B, the model learns to predict whether A is more relevant than B or vice versa.

- **Advantages:**
  - Optimizes the ranking directly by focusing on item pairs.
  - Reduces the problem of learning absolute scores and focuses on relative comparisons.

- **Disadvantages:**
  - The number of pairs grows quadratically with the number of items, making it computationally expensive.
  - Ignores absolute relevance scores.

In the pairwise approach, we focus on pairs of documents and learn which document should be ranked higher.

##### Pairs for Training:
- Compare `Doc A` and `Doc B`: Label = A is higher than B (3 > 1)
- Compare `Doc A` and `Doc C`: Label = A is higher than C (3 > 2)
- Compare `Doc B` and `Doc C`: Label = C is higher than B (2 > 1)

We train a model to predict the relative ranking between pairs of documents.

##### Prediction:
After training, the model predicts the following relative orders:
- `Doc A` > `Doc C`
- `Doc C` > `Doc B`

The predicted ranking would be: `Doc A`, `Doc C`, `Doc B`.

#### 3. RankNet:
**RankNet** is a specific type of pairwise ranking algorithm developed by Microsoft Research. It uses a neural network to predict the relative ranking between two items. The network outputs the probability that one item is ranked higher than the other, and the loss function used is a cross-entropy loss based on these probabilities.

- **How RankNet Works:**
  1. Two items are input into the network.
  2. The network predicts a score for each item.
  3. The predicted scores are then transformed into probabilities that one item is ranked higher than the other using a sigmoid function.
  4. The loss is computed using the cross-entropy between the predicted probability and the actual relative ranking.

- **Example:** For search engine ranking, RankNet compares pairs of documents and learns whether document A should be ranked higher than document B.

- **Advantages:**
  - Directly focuses on ranking pairs, making it effective for ranking tasks.
  - Flexible and can be used with different neural network architectures.

- **Disadvantages:**
  - Still requires generating pairs, which increases computational complexity.
  - It may not capture complex interactions as effectively as newer models like ListNet.

In RankNet, we input pairs of documents and the model outputs a probability that one document is ranked higher than the other.

##### Pairs for Training:
- `Doc A` vs `Doc B`: True Label = `Doc A` is higher (3 > 1)
- `Doc A` vs `Doc C`: True Label = `Doc A` is higher (3 > 2)
- `Doc C` vs `Doc B`: True Label = `Doc C` is higher (2 > 1)

The model is trained using these pairwise comparisons. It outputs probabilities based on which document should be ranked higher, using a neural network to predict scores for each document.

##### Prediction:
The model predicts the following probabilities:
- `Doc A` is ranked higher than `Doc B` with probability 0.9.
- `Doc A` is ranked higher than `Doc C` with probability 0.8.
- `Doc C` is ranked higher than `Doc B` with probability 0.85.

The predicted ranking would be: `Doc A`, `Doc C`, `Doc B`.

##### Summary:
- **Point-wise:** Each item is treated independently, simple but doesn't optimize ranking directly.
- **Point-wise Approach** predicts relevance scores directly for each document and ranks them.

- **Pairwise:** Focuses on comparing pairs of items, optimizes ranking but can be computationally expensive.
- **Pairwise Approach** compares documents in pairs and learns which document should be ranked higher.

- **RankNet:** A neural network-based pairwise model that predicts relative rankings using probabilities.
- **RankNet** uses a neural network to predict the relative ranking between pairs of documents, outputting probabilities that one document is ranked higher than another.

## J. Similarity Functions
#### 1. Euclidean Distance
Euclidean distance is the straight-line distance between two points in multi-dimensional space. It is often used in clustering (e.g., k-means) and nearest-neighbor algorithms.

##### Pros:
- Intuitive: Represents the actual geometric distance between points, easy to understand.
- Effective in low dimensions: Works well when the number of dimensions is small.
##### Cons:
- Sensitive to scale: If features are on different scales (e.g., age in years vs. height in centimeters), this can distort the distance. Normalization is needed.
- Curse of dimensionality: As the number of dimensions increases, Euclidean distance loses effectiveness due to all points appearing equidistant.

#### 2. Cosine Similarity
Cosine similarity measures the cosine of the angle between two non-zero vectors. It is commonly used in text analysis to measure the similarity of documents.

##### Pros:
- Scale-invariant: Focuses on the direction rather than the magnitude, so it's useful when comparing high-dimensional data like text, where the magnitude (e.g., document length) can vary.
- Works well for sparse data: Effective in cases where vectors are sparse, such as in document term matrices.
##### Pros:
 Cons:
- Ignores magnitude: If the magnitude of vectors matters (i.e., the size of the values), cosine similarity might not be suitable since it only considers the angle.
- Not appropriate for negative values: Works best when feature values are non-negative, such as in word count vectors or TF-IDF matrices.

#### 3. Manhattan Distance (L1 Norm)
Also known as "taxicab" or "city block" distance, Manhattan distance calculates the sum of the absolute differences between the coordinates of two points.

##### Pros:
- Robust to outliers: Less sensitive to large differences compared to Euclidean distance, which is affected by squared values.
- Works in high dimensions: Often more effective in high-dimensional spaces than Euclidean distance.
##### Cons:
- Not as intuitive: The geometric meaning of this distance can be less intuitive, especially in non-grid-like data.
- Sensitive to feature scaling: Like Euclidean distance, it requires normalization of features to avoid biasing toward features with larger ranges.

#### 4. Jaccard Similarity
Jaccard similarity is used for comparing the similarity and diversity of sets. It is the ratio of the intersection to the union of two sets. Commonly used in binary or categorical data.

##### Pros:
Effective for set-based similarity: Useful in applications involving set comparison, such as recommendation systems, document comparison, or binary features.
Good for sparse data: Works well when the data is sparse or binary.
##### Cons:
Ignores frequency information: Jaccard does not consider how many times an item appears; it only looks at whether it appears or not (presence/absence).
Sensitive to small sets: If one or both sets are small, Jaccard similarity can be misleading since small differences are amplified.

#### 5. Hamming Distance
Hamming distance measures the number of positions at which two strings of equal length differ. It is typically used for categorical variables and binary strings.

##### Pros:
- Simple to compute: Works well for binary and categorical data.
- Useful for exact match tasks: Ideal for cases where small deviations in sequences matter, such as in DNA sequences or binary data.
##### Cons:
- Not suitable for continuous variables: Designed for binary or categorical data, it doesn’t work well when the features are continuous.
- Sensitive to length: Requires strings or vectors to have the same length.

#### 6. Minkowski Distance
The Minkowski distance is a generalized metric that encompasses both Euclidean and Manhattan distances. It's defined as:

$$
d(x, y) = \left(\sum_{i=1}^{n} |x_i - y_i|^p\right)^{1/p}
$$

where:

- x and y are two points in n-dimensional space.
- p is a positive integer that determines the type of distance:
- p = 1: Manhattan distance (also known as L1 norm)
- p = 2: Euclidean distance (also known as L2 norm)

By varying the value of p, you can explore different distance metrics that may be more suitable for specific applications. For example, Manhattan distance might be more appropriate for data with a grid-like structure (like city blocks), while Euclidean distance is better suited for continuous spaces.

##### Pros:
- Flexibility: With the parameter 
𝑝, you can interpolate between Manhattan and Euclidean distances based on the problem at hand.
- General-purpose: Can be adapted to a wide range of scenarios depending on how the 
p-norm is set.
##### Cons:
- Interpretability: Can be harder to interpret, especially when p ≠ 1 or p ≠ 2.
- Sensitivity to p: The performance can vary widely based on the choice of p, and it may require tuning.

#### 7. Mahalanobis Distance
Mahalanobis distance measures the distance between two points while accounting for correlations in the dataset. It is a generalized form of Euclidean distance.

##### Pros:
- Accounts for correlations: Useful when there are relationships between features, as it takes into account the covariance between variables.
- Works in multivariate data: Suitable for situations where the features are interrelated.
##### Cons:
- Requires the inverse covariance matrix: Computing this matrix can be computationally expensive, especially in high dimensions or when the matrix is singular.
- Sensitive to outliers: If the dataset contains outliers, they can skew the covariance matrix, distorting the distance measure.

#### 8. Pearson Correlation
Pearson correlation measures the linear relationship between two variables. It ranges from -1 (perfect negative correlation) to 1 (perfect positive correlation).

##### Pros:
- Simplicity: Easy to compute and interpret.
- Linear relationship: Works well when the relationship between variables is linear.
##### Cons:
- Sensitive to outliers: Pearson correlation can be heavily influenced by outliers.
- Assumes linearity: Does not capture non-linear relationships.

#### 9. Spearman Correlation
Spearman correlation measures the rank correlation between two variables. It assesses how well the relationship between two variables can be described by a monotonic function.

##### Pros:
- Non-parametric: Does not assume a linear relationship between variables, capturing monotonic relationships.
- Less sensitive to outliers: Since it uses ranks, it's more robust to outliers compared to Pearson correlation.

##### Cons:
- Ignores magnitude of differences: Only looks at rank, not the actual differences between values.
- Less interpretable in some cases: Can be harder to interpret compared to Pearson correlation when trying to understand the strength of association.

##### Summary Table

| Similarity Function | Pros | Cons |
|---|---|---|
| Euclidean Distance | Intuitive, good for low-dimensional data | Sensitive to scale, not effective in high dimensions |
| Cosine Similarity | Scale-invariant, works well for text and sparse data | Ignores magnitude, not suited for negative values |
| Manhattan Distance | Robust to outliers, useful in high dimensions | Requires normalization, less intuitive |
| Jaccard Similarity | Effective for set-based and sparse data, works with binary features | Ignores frequency, sensitive to small sets |
| Hamming Distance | Simple, works well for binary and categorical data | Not suitable for continuous variables, requires equal-length strings |
| Minkowski Distance | Flexible, generalizes Euclidean and Manhattan distances | Requires tuning of the parameter p, can be hard to interpret |
| Mahalanobis Distance | Accounts for feature correlations, useful for multivariate data | Computationally expensive, sensitive to outliers |
| Pearson Correlation | Easy to compute and interpret, effective for linear relationships | Sensitive to outliers, assumes linearity |
| Spearman Correlation | Captures monotonic relationships, less sensitive to outliers | Ignores magnitude, may be harder to interpret |


## K. ML Model Implemented from Scratch

- **Linear Regression:** [Linear Regression Notebook](notebooks/linear_regression.ipynb)
- **Logistic Regression:** [Logistic Regression Notebook](notebooks/logistic_regression.ipynb)
- **Active Learning:** [Active Learning Notebook](notebooks/active_learning.ipynb)
- **Autoencoders:** [Autoencoders Notebook](notebooks/autoencoders.ipynb)
- **Association Rule:** [Association Rule Notebook](notebooks/association_rule_learning.ipynb)
- **Boosting and Bagging:** [Boosting and Bagging Notebook](notebooks/boosting_n_bagging.ipynb)
- **CNN:** [CNN Notebook](notebooks/cnn.ipynb)
- **Collaborative Filtering:** [Collaborative Filtering Notebook](notebooks/collaborative_filtering.ipynb)
- **Compare LSTM, GRU, RNN, Transformers:** [Compare LSTM GRU RNN Transformers Notebook](notebooks/compare_lstm_gru_rnn_transformer.ipynb)
- **Decision Tree:** [Decision Tree Notebook](notebooks/decision_tree.ipynb)
- **ML Deployment:** [ML Deployment Notebook](notebooks/deployment.ipynb)
- **Graph-based ML:** [Graph-based ML Notebook](notebooks/graph_based_ml.ipynb)
- **GRU:** [GRU Notebook](notebooks/gru.ipynb)
- **K-Means:** [KMeans Notebook](notebooks/kmeans.ipynb)
- **KNN:** [KNN Notebook](notebooks/knn.ipynb)
- **Naive Bayes:** [Naive Bayes Notebook](notebooks/naive_bayes.ipynb)
- **LSTM:** [LSTM Notebook](notebooks/lstm.ipynb)
- **Multimodal:** [Multimodal Notebook](notebooks/multimodal.ipynb)
- **PCA:** [PCA Notebook](notebooks/pca.ipynb)
- **Random Forest:** [Random Forest Notebook](notebooks/random_forest.ipynb)
- **RNN:** [RNN Notebook](notebooks/rnn.ipynb)
- **Transformers:** [Transformers Notebook](notebooks/transformers.ipynb)
- **SVM:** [SVM Notebook](notebooks/svm.ipynb)
- **Vision Transformer:** [Vision Transformer Notebook](notebooks/vision_transformer.ipynb)
- **Two Tower Model:** [Two Tower Model Notebook](notebooks/two_tower_model.ipynb)
- **XGBoost:** [XGBoost Notebook](notebooks/xgboost.ipynb)

## L. DL Concepts

#### 1. Overfitting vs. Underfitting

* **Overfitting:** Occurs when a model is too complex and learns the training data too well, including noise and specific details. This leads to poor performance on unseen data.
* **Underfitting:** Happens when a model is too simple and fails to capture the underlying patterns in the data. This results in poor performance on both training and test data.

#### 2. Bias-Variance Tradeoff

* **Bias:** Error introduced by simplifying assumptions made by the model. High bias leads to underfitting.
* **Variance:** Sensitivity of the model to small fluctuations in the training data. High variance leads to overfitting.

The goal is to find a balance between bias and variance to minimize overall error.

#### 3. Cross-Validation
Cross-validation is a technique to evaluate a model's performance by dividing the dataset into multiple folds. The model is trained on different subsets of the data and evaluated on the remaining subset. This helps assess the model's generalization ability.

#### 4. Additional Related Topics

* **Regularization:** Techniques like L1 and L2 regularization penalize complex models to prevent overfitting.
* **Hyperparameter Tuning:** Optimizing model parameters (e.g., learning rate, number of layers) to improve performance.
* **Learning Curves:** Visualize model performance on training and validation sets to diagnose overfitting or underfitting.
* **Model Complexity:** Balance model complexity to avoid overfitting and underfitting.
* **Resampling Techniques:** Create multiple datasets from the original data to improve generalization.
* **Ensemble Learning:** Combine multiple models to improve performance and reduce overfitting/underfitting.

#### 5. Types of Dropout

1. **Standard Dropout**  
   - **Description**: Randomly "drops" (sets to zero) a fraction of neurons in each layer during training, preventing overfitting by making the network less dependent on specific neurons.
   - **Example**: In a fully connected layer, applying a dropout rate of 0.5 will set half of the neurons to zero in each forward pass during training.

2. **Spatial Dropout**  
   - **Description**: Commonly used in convolutional networks, it drops entire feature maps (channels) rather than individual neurons. This maintains spatial information while still reducing overfitting.
   - **Example**: Applying spatial dropout on a 32-channel convolutional layer with a rate of 0.2 would randomly set about 6-7 entire channels to zero.

3. **DropConnect**  
   - **Description**: Drops individual weights (connections) instead of neurons, randomly setting a fraction of the weights in the network to zero during training.
   - **Example**: With a DropConnect rate of 0.5, half of the weights between neurons in a dense layer are randomly set to zero in each training step.

4. **Variational Dropout**  
   - **Description**: Often used in Bayesian networks, variational dropout applies a unique dropout rate per neuron that can adapt during training, useful for uncertainty estimation.
   - **Example**: In a probabilistic layer, variational dropout can adjust the dropout rate per neuron, allowing some neurons to remain active more consistently if they contribute significantly to the model.

5. **Concrete Dropout**  
   - **Description**: Learns the dropout rate as a continuous variable, making it differentiable and thus optimizable. Useful in tasks requiring adaptive dropout rates.
   - **Example**: A neural network can learn a dropout rate in real-time for specific layers, such as a high dropout rate in an overfitting layer and low dropout in a critical layer.

6. **Group Dropout**  
   - **Description**: Drops groups of neurons based on predefined groups or structures, such as groups of features. Unlike Spatial Dropout, it does not drop entire feature maps but rather specific groups within the network.
   - **Example**: In a multi-head network, Group Dropout can drop entire groups of neurons corresponding to particular feature sets, preserving the structure within other groups.

#### 6. Types of Normalization Techniques in Neural Networks

##### 1. **Layer Normalization**  
   - **Description**: Layer Normalization normalizes the activations across the features within each data sample (i.e., along the feature axis in a layer), rather than across the batch. This method stabilizes and accelerates training by reducing covariate shift and improving gradient flow.
   - **How It’s Done**: For each sample, the mean and variance of the features are computed. Each feature is then normalized using these statistics, with learnable parameters for rescaling and shifting the output.
   - **Formula**: Given a sample with features \( x_1, x_2, ..., x_d \):
     $     \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}     $
     where \( \mu \) and \( \sigma^2 \) are the mean and variance across all features in the sample, and \( \epsilon \) is a small constant.
   - **Use Case**: Ideal for RNNs, NLP, and sequence models where batch sizes are small or variable.
   - **When to Use**: When working with variable-length sequences or RNN-based tasks, as it stabilizes training without relying on batch size.

   Layer Normalization is typically applied after each layer’s linear transformation (e.g., fully connected layer or convolutional layer), but it’s not strictly necessary to apply it to every layer. In practice, it’s most beneficial in certain contexts, especially in models where sequential or recurrent dependencies are crucial, like RNNs, Transformers, and NLP applications.

   When used, Layer Normalization is generally applied right before or after the activation function in each layer, depending on the network architecture. For example:

    - In recurrent networks (RNNs, LSTMs, GRUs): Layer Normalization is often applied to stabilize the hidden state transformations, and it's usually added after each layer to address the internal covariate shift.
    - In Transformers and attention-based models: It’s often applied after each self-attention or feed-forward sublayer to help with gradient flow across layers.

    While it’s advantageous to use in models with sequential dependencies, in some architectures like CNNs or large-scale dense networks, other normalization methods (e.g., Batch Normalization) are more common, as they might offer better performance or efficiency with large batch sizes.

    Let's walk through an example of Layer Normalization in a simple neural network. We'll demonstrate how applying Layer Normalization improves stability during training. I'll create a small dataset, build a neural network with and without Layer Normalization, and compare the training stability.

    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt

    # Create a simple dataset
    # Features: 100 samples with 10 features each, Labels: 0 or 1
    torch.manual_seed(0)
    data = torch.randn(100, 10)
    labels = torch.randint(0, 2, (100,))

    # Define a simple neural network without Layer Normalization
    class SimpleNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Define a similar network with Layer Normalization
    class SimpleNetLayerNorm(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleNetLayerNorm, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.layer_norm = nn.LayerNorm(hidden_size)  # Apply Layer Normalization
            self.fc2 = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            x = torch.relu(self.layer_norm(self.fc1(x)))  # Apply after first layer
            x = self.fc2(x)
            return x

    # Training function
    def train(model, data, labels, epochs=50):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        losses = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses

    # Initialize models
    model_basic = SimpleNet(input_size=10, hidden_size=5, output_size=2)
    model_layer_norm = SimpleNetLayerNorm(input_size=10, hidden_size=5, output_size=2)

    # Train models
    losses_basic = train(model_basic, data, labels)
    losses_layer_norm = train(model_layer_norm, data, labels)

    # Plot the training loss for comparison
    plt.plot(losses_basic, label='Without Layer Normalization')
    plt.plot(losses_layer_norm, label='With Layer Normalization')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.show()

    ```

    - Explanation of the Code
        Dataset: We create a small synthetic dataset with 100 samples and 10 features. Labels are binary (0 or 1).

    - Models:

        - SimpleNet: A simple two-layer neural network without Layer Normalization.
        - SimpleNetLayerNorm: A similar network with Layer Normalization applied after the first layer.
    - Training Function: This trains each model using cross-entropy loss and SGD optimizer.

    - Comparison: After training, we plot the loss over epochs for both models.

    - Expected Outcome
        When you run the code, you should see a plot comparing the loss over training epochs for both models. The model with Layer Normalization will likely show more stable and faster convergence, while the model without Layer Normalization might exhibit more fluctuations and potentially slower convergence. This stability is one of the key benefits of using Layer Normalization, particularly for models with sequential dependencies.

    ##### Example of Layer Normalization

    Consider a hidden layer with **4 neurons** producing activations: `[10.0, 0.5, 2.0, 30.0]`.

    ##### Without Layer Normalization
    - The activations remain unadjusted, which can cause instability (one neuron, for instance, dominates with a value of `30.0`).

    ##### With Layer Normalization
    1. Calculate **mean**:  
    $    \text{mean} = \frac{(10.0 + 0.5 + 2.0 + 30.0)}{4} = 10.625$
    
    2. Calculate **variance**:  
    Average of  
    $[(10.0 - 10.625)^2, (0.5 - 10.625)^2, (2.0 - 10.625)^2, (30.0 - 10.625)^2] $

    3. Normalize each activation:  
    $    \text{normalized\_activation} = \frac{(activation - mean)}{\text{std dev}}$

    **Result**: All activations are now on a similar scale (e.g., `[-0.05, -0.80, -0.69, 1.53]`), stabilizing training and preventing any one neuron from dominating.

    [Layer Normalization - EXPLAINED](https://www.youtube.com/watch?v=G45TuC6zRf4)

##### 2. **Batch Normalization**  
   - **Description**: Batch Normalization normalizes the activations of each layer across the mini-batch, reducing internal covariate shift and improving training speed and stability. By normalizing the mean and variance across each mini-batch, it makes the network less sensitive to initialization and allows higher learning rates.
   - **How It’s Done**: For each mini-batch, the mean and variance of each feature are computed across the batch. Each feature is then normalized and scaled with learnable parameters.
   - **Formula**: For a batch of features \( x_1, x_2, ..., x_n \):
     $     \hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}     $
     where \( \mu_B \) and \( \sigma_B^2 \) are the batch mean and variance, and \( \epsilon \) is a small constant.
   - **Use Case**: Commonly used in CNNs and large-scale networks with stable, large batch sizes.
   - **When to Use**: When training CNNs or dense networks with large, consistent batch sizes; not ideal for small or variable batch sizes.

   [Batch Normalization - EXPLAINED](https://www.youtube.com/watch?v=DtEq44FTPM4)

##### 3. **Group Normalization**  
   - **Description**: Group Normalization divides channels in a layer into groups and normalizes each group separately, independent of the batch size. This technique is useful when batch sizes are small or vary significantly.
   - **How It’s Done**: Channels are divided into groups, and within each group, the mean and variance are calculated. Learnable parameters are applied for rescaling and shifting.
   - **Formula**: For each group \( g \) in the feature map:
     $     \hat{x}_i^g = \frac{x_i^g - \mu_g}{\sqrt{\sigma_g^2 + \epsilon}} $
     
     where \( \mu_g \) and \( \sigma_g^2 \) are the mean and variance across all elements in group \( g \).
   - **Use Case**: Effective in computer vision tasks and with small batch sizes.
   - **When to Use**: In CNNs when using small batch sizes or when batch normalization is ineffective due to batch size constraints.

##### 4. **Instance Normalization**  
   - **Description**: Instance Normalization normalizes each individual sample in the batch independently, with separate mean and variance for each feature map. This approach is particularly useful in style transfer applications where maintaining local contrast is important.
   - **How It’s Done**: For each sample, the mean and variance of each feature map are computed, normalizing each feature map independently.
   - **Formula**: For a sample with feature map \( x \):
     $     \hat{x}_i = \frac{x_i - \mu_{\text{instance}}}{\sqrt{\sigma_{\text{instance}}^2 + \epsilon}}     $
     where \( \mu_{\text{instance}} \) and \( \sigma_{\text{instance}}^2 \) are the mean and variance of each feature map.
   - **Use Case**: Commonly used in Generative Adversarial Networks (GANs) and style transfer models.
   - **When to Use**: When the task requires preserving detailed local features, such as in style transfer and GANs.

##### 5. **Layer Scale Normalization (LSN)**  
   - **Description**: This technique combines Layer and Instance Normalization by scaling normalized activations for each layer with a learned scaling factor. It allows the network to learn optimal normalization for each layer.
   - **How It’s Done**: Each layer's normalization is scaled by learnable parameters specific to that layer.
   - **Formula**: The normalized output is scaled as follows:
     $     \text{output} = \alpha \cdot \text{LayerNorm}(x) + \beta     $
     where \( \alpha \) and \( \beta \) are learnable parameters for scaling and shifting.
   - **Use Case**: Suitable for Transformer models and deep architectures.
   - **When to Use**: In Transformer-based or very deep networks, allowing for layer-specific normalization adjustments.

Each normalization technique is suited to specific network architectures and task requirements, offering different ways to improve training stability and convergence.

##### Choosing the Right Technique:

The choice of normalization technique depends on the specific task and dataset. Consider the following factors:

- Batch size: For small batch sizes, Group Normalization or Layer Normalization might be more suitable.
- Data distribution: For datasets with varying data distributions, Layer Normalization can be effective.
- Task complexity: For complex tasks, Batch Normalization can help improve performance.