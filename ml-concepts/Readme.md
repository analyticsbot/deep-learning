## Feature Engineering
### Bucketizing
Bucketizing, in the context of machine learning, refers to the process of dividing a continuous feature or variable into discrete intervals, or "buckets." It is often used when dealing with continuous numerical features but can also be applied to categorical features in certain scenarios.

For categorical features, bucketizing typically involves grouping similar categories together to reduce the dimensionality or complexity of the feature. This is useful when you have many distinct categories, and combining related ones into broader groups can make the model more generalizable or easier to train.

For example:

Original Categorical Feature: Types of fruits (apple, banana, orange, grape, mango)
Bucketized Feature: Grouped into "Citrus" (orange) and "Non-citrus" (apple, banana, grape, mango)
In the case of continuous features (like age or income), bucketizing involves converting ranges of values into categories (e.g., grouping ages 0-20, 21-40, etc.).

Bucketizing simplifies data, but it should be used carefully to avoid losing important nuances.

### Feature Hashing ####

Feature hashing, or hashing trick, converts text data, or categorical attributes with high cardinalities, into a feature vector of arbitrary dimensionality. In some AdTech companies (Twitter, Pinterest, etc.), it’s not uncommon for a model to have thousands of raw features.


In AdTech, consider a model that predicts user click behavior based on categorical features like "ad campaign ID" or "user location." These features can have thousands of unique values, which would make one-hot encoding inefficient. Feature hashing compresses these high-cardinality categorical features into a fixed-length feature vector by applying a hash function, allowing the model to handle a large number of features efficiently without exploding the feature space. For example, "ad campaign ID" might be hashed into a vector of length 1,000 instead of having individual binary columns for each ID.

#### Feature Hashing Example with Python and `sklearn`

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
FeatureHasher is particularly useful when working with datasets that have high cardinality categorical features.
By converting categorical features into fixed-length vectors, you can reduce the memory footprint and still retain useful information for machine learning models.


#### Cross Feature and Hashing Trick Example in Python

This example demonstrates **cross features** and how we can use the **hashing trick** to manage high-dimensional categorical data in Python using `sklearn`.

#### What is a Cross Feature?

A **cross feature** is simply a new feature created by combining two or more categorical features. For example, if we have the Uber pickup data containing `latitude` and `longitude` of locations, we can combine them (or cross them) to create a new feature that represents the pickup location more uniquely.

- Let's say we have two categorical features, `latitude` and `longitude`.
- If `latitude` has 1,000 possible values and `longitude` has 1,000 possible values, their cross feature would have 1,000 × 1,000 = 1,000,000 possible values. This makes the data very high-dimensional.

To handle this large number of possible combinations (or high-dimensional data), we use the **hashing trick**, which reduces the number of dimensions while still preserving useful information.

#### Example: Predict Uber Demand Using Cross Features

#### Data

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

#### Why Use the Hashing Trick?
Cross features often create many new categories when you combine existing features, making the data too large or "high-dimensional." Using the hashing trick helps keep the data manageable while still using the important information in your machine learning model.

## Word Embeddings
### CBOW and Skip-gram Models in Word2Vec

Word2Vec is a popular technique used to create word embeddings, where words are represented as dense vectors in a continuous vector space. There are two primary models used in Word2Vec: **CBOW (Continuous Bag of Words)** and **Skip-gram**.

#### 1. CBOW (Continuous Bag of Words)

- **Goal**: Predict the target word using the context words (neighboring words).
- In this model, the context is used to predict the center word.
  
For example, in the sentence "The cat sits on the mat," if "cat" is the target word, the context might be ["The", "sits"].

#### CBOW Python Example

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

#### Skip-gram Model
Goal: Predict the context words (neighboring words) given the target word.
In this model, the center word is used to predict its context.
For example, in the sentence "The cat sits on the mat," if "cat" is the target word, the model tries to predict its neighboring words "The" and "sits." based on a variable called window size.

Skip-gram Python Example
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

#### Visual Summary
CBOW: Context -> Target
Skip-gram: Target -> Context


### How Does Instagram Train User Embedding?

Instagram aims to recommend photos to users that align with their current interests during a session. Here's a simplified view of how Instagram could train user embeddings based on user interactions with other users' photos.

#### Concept

- **User A's session**: Imagine User A browsing photos from different accounts in one session.
- **Sequence assumption**: If User A is interested in certain topics, then the photos they view from User B and User C are likely related to that interest.
  
Instagram can model these sequences of interactions (e.g., "User A sees User B's photos, then User C's photos") as sequences of words in a sentence. This allows Instagram to learn user embeddings that reflect users' preferences during a session.

#### Example Scenario

- **Session**: User A → sees User B’s photos → sees User C’s photos.
  
This can be treated like a sentence where:
- Each session is a "sentence."
- Each user whose photos are viewed is like a "word" in that sentence.

By representing these sessions as sequences, we can train embeddings for each user, similar to how we might train word embeddings in natural language processing.

#### Sequence Embedding Example with Python

Here’s a Python example of how you can implement this using simple sequence embedding logic. We'll simulate user sessions and generate training data.

#### Python Example

```python
import numpy as np
from collections import defaultdict

# Simulate user sessions where each session contains a sequence of users' photos viewed
sessions = [
    ["user_A", "user_B", "user_C"],
    ["user_A", "user_D", "user_E"],
    ["user_B", "user_C", "user_F"]
]

# Vocabulary and user to index mapping
users = {user for session in sessions for user in session}
user2idx = {user: i for i, user in enumerate(users)}
idx2user = {i: user for user, i in user2idx.items()}

# Parameters
embedding_dim = 10  # Embedding size

# Generate training data (sequence style: context -> target)
def generate_sequence_data(sessions):
    data = []
    for session in sessions:
        for i, user in enumerate(session):
            target = user2idx[user]
            context = [user2idx[session[j]] for j in range(max(0, i - 1), min(len(session), i + 2)) if i != j]
            data.append((context, target))
    return data

training_data = generate_sequence_data(sessions)
print(f"Training data (Sequence Embedding): {training_data}")
```

#### Output
The output represents the context-target pairs used in sequence embedding training:
```Training data (Sequence Embedding): [([user_B_idx, user_C_idx], user_A_idx), ...]```

#### Explanation of the Code
Sessions: Each session is a sequence of users, where User A might see photos from User B, then User C.
Training data: We generate context-target pairs for training embeddings, where each user has surrounding users in the session as context.


### Recommendation System Design

#### Overview

This project focuses on building a recommendation system that efficiently suggests items to users based on their preferences. One of the key design choices is the use of the **Hadamard product** over more common functions like **cosine similarity**. This choice allows the model to learn its own distance function while reducing latency in online scoring.

#### Key Concepts

#### Hadamard Product

The Hadamard product is an element-wise multiplication of two matrices (or vectors). For two vectors \( A \) and \( B \), the Hadamard product \( A \circ B \) is calculated as:

\[
A \circ B = (A_1 \cdot B_1, A_2 \cdot B_2, \ldots, A_n \cdot B_n)
\]

#### Cosine Similarity

Cosine similarity measures the cosine of the angle between two non-zero vectors in an inner product space. It is commonly used to determine how similar two items are, regardless of their magnitude.

#### Distance Function

In machine learning, a distance function helps determine how "far apart" or "similar" two data points (like user preferences or item attributes) are. A model that learns its own distance function can adapt to specific data characteristics, improving performance.

#### Fully Connected Layer

In neural networks, a fully connected layer means that each neuron in one layer is connected to every neuron in the next layer. This can increase computational costs and latency, particularly in real-time systems like recommendation engines.

## Example Scenario

### Movie Recommendation System

In our movie recommendation system, we represent users and movies as vectors.

- **User Preferences Vector**:
    - \( U = [0.9, 0.1, 0.5] \) (indicating preferences for Action, Comedy, and Drama).

- **Movie Attributes Vector**:
    - \( M = [0.8, 0.4, 0.3] \).

#### Hadamard Product Approach

The Hadamard product \( U \circ M \) yields:

\[
U \circ M = [0.9 \cdot 0.8, 0.1 \cdot 0.4, 0.5 \cdot 0.3] = [0.72, 0.04, 0.15]
\]

The resulting vector indicates the "combined" score for each genre, allowing the model to learn how to weigh different attributes based on user preferences.

#### Cosine Similarity Approach

If we used cosine similarity, we would compute:

\[
\text{cosine similarity} = \frac{U \cdot M}{\|U\| \|M\|}
\]

This would return a single similarity score that tells how similar the user is to the movie, which might not provide as detailed a picture of preferences as the Hadamard product.

#### Advantages of the Hadamard Product

- **Flexibility**: The model can learn and adjust weights for each attribute independently, capturing the nuances of user preferences.
- **Reduced Latency**: By avoiding a fully connected layer, the system can operate more quickly, which is critical for online recommendation systems where users expect instant results.

#### Conclusion

The choice of the Hadamard product allows the model to create a more tailored representation of user preferences while maintaining efficiency in scoring recommendations. This flexibility can lead to improved performance in suggesting items that align closely with users' unique tastes.


### Handling Imbalanced Class Distribution in Multi-Class Problems
In machine learning tasks like fraud detection, click prediction, or spam detection, it's common to have imbalanced labels. For example, in ad click prediction, you might have a 0.2% conversion rate, meaning out of 1,000 clicks, only two lead to a desired action. This imbalance can cause the model to focus too much on learning from the majority class.

When dealing with multi-class problems, methods like SMOTE (Synthetic Minority Over-sampling Technique) are not always effective. Below are some strategies to handle class imbalance in multi-class settings:

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

3. Hybrid Approach: Combining Oversampling and Undersampling
A combination of oversampling for minority classes and undersampling for majority classes.

4. Multi-Class Variants of SMOTE
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

5. Ensemble Methods
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
6. Data Augmentation
Data augmentation can help generate more samples for minority classes, especially useful for image, text, or time-series data.

These methods can help tackle the challenge of class imbalance in multi-class machine learning tasks.

### Course Recommendations on LinkedIn Learning

#### Problem
The goal of Course Recommendations is to acquire new learners by showing highly relevant courses to learners. However, there are challenges:

1. **Lack of label data**: Without engagement signals like user activities (browse, click), we can't use implicit labels for training. This is known as the *Cold Start problem*.
   - A possible solution is user surveys during onboarding, where learners share the skills they want to learn. However, this is often insufficient.

#### Example Scenario
Given a learner with skills in *Big Data*, *Database*, and *Data Analysis*, LinkedIn Learning has two course options: *Data Engineering* and *Accounting 101*. The model should recommend *Data Engineering* since it aligns better with the learner's skills. This illustrates that skills can measure relevance.

#### Skill-Based Model

#### Course to Skill: Cold Start Model
1. **Manual Tagging**: 
   - Use taxonomy to map LinkedIn Learning courses to skills. Taxonomists perform this mapping, which achieves high precision but low coverage.
   
2. **Leverage LinkedIn Skill Taggers**: 
   - Use LinkedIn Skill Taggers to extract skills from course data.

3. **Supervised Model**: 
   - Train a classification model that takes a pair (course, skill) and returns `1` if relevant, `0` otherwise.

#### Data for Supervised Model
- Positive examples: From manual tagging and LinkedIn Skill Taggers.
- Negative examples: Randomly sampled data.
- Features: Course data such as title, description, and section names. Skill-to-skill similarity is also used.

#### Disadvantages:
- Heavy reliance on skill taggers' quality.
- A single logistic regression model may not capture skill-level effects.

4. **Semi-Supervised Learning**:
   - Train separate models for each skill, rather than one general model for all (course, skill) pairs.
   
5. **Data Augmentation**:
   - Use skill-correlation graphs to add positive labels. For example, if SQL is highly related to Data Analysis, both skills should be labeled positively for relevant courses.

#### Evaluation: Offline Metrics
1. **Skill Coverage**: Measures how many LinkedIn standardized skills are present.
2. **Precision and Recall**: Use human-generated mappings as ground truth to evaluate the model.

#### Member to Skill

1. **Member to Skill via Profile**: 
   - LinkedIn users add skills to their profiles, which is often noisy and needs to be standardized. Coverage is limited since not all users provide skill data.

2. **Member to Skill Using Title and Industry**:
   - Infer skills using cohort-level mapping based on job titles and industry. For example, a Machine Learning Engineer in Ad Tech might not provide skills but can be inferred from the cohort.

#### Final Skill Mapping
Combine profile-based and cohort-based mappings using a weighted combination. For instance, if SQL has a higher weight in the cohort mapping, it will influence the final score accordingly.


### Regression Loss Functions

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

### CLassifcation Loss

## Binary Classification Loss Functions

#### 1. Focal Loss
#### Description:
Focal loss is designed to down-weight easy examples and focus on learning from hard, misclassified examples.

$$
\text{Focal Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ \alpha(1 - p_i)^{\gamma} y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
$$

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


## Multi-Class Classification Loss Functions

1. **Categorical Cross-Entropy Loss**

   **Formula:**
    $$
    \text{Categorical Cross-Entropy} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(p_{i,c})
    $$

   **Use Cases:**
   - Used for multi-class classification tasks where classes are one-hot encoded.
   - Effective when class distributions are balanced.

   **Avoid When:**
   - When the target classes are not mutually exclusive, as it assumes that classes are one-hot encoded.

2. **Sparse Categorical Cross-Entropy Loss**

   **Formula:**
    $$
    \text{Sparse Categorical Cross-Entropy} = -\frac{1}{N} \sum_{i=1}^{N} \log(p_{i,y_i})
    $$

   **Use Cases:**
   - Similar to categorical cross-entropy but used when the target classes are provided as integer labels (not one-hot encoded).

   **Avoid When:**
   - When one-hot encoding is required for compatibility with certain models.

3. **Kullback-Leibler Divergence (KL Divergence)**

   **Formula:**
    $$
    D_{KL}(P || Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)}
    $$

   **Use Cases:**
   - Useful for measuring how one probability distribution diverges from a second expected probability distribution.

   **Avoid When:**
   - When actual class probabilities are not known or are very small, leading to instability in computation.

4. **Normalized Cross-Entropy Loss**

   **Formula:**
    $$
    \text{Normalized Cross-Entropy} = -\frac{1}{N} \sum_{i=1}^{N} \left( \sum_{c=1}^{C} \frac{y_{i,c}}{\sum_{c'} y_{i,c'}} \log(p_{i,c}) \right)
    $$

   **Use Cases:**
   - Useful when class frequencies vary significantly, normalizing the contribution of each class.

   **Avoid When:**
   - When the normalization may obscure the learning of relevant features.

5. **Hinge Loss (Multi-Class)**

   **Formula:**
    $$
    \text{Multi-Class Hinge Loss} = \sum_{i=1}^{N} \sum_{j \neq y_i} \max(0, 1 - f(x_i, y_i) + f(x_i, j))
    $$

   **Use Cases:**
   - Applied in multi-class SVMs and problems where margin maximization is desired.

   **Avoid When:**
   - When probabilistic interpretations of the output are needed.

6. **Triplet Loss**

   **Formula:**
    $$
    \text{Triplet Loss} = \max(0, d(a, p) - d(a, n) + \alpha)
    $$
   where \(d\) is the distance function, \(a\) is the anchor, \(p\) is the positive example, \(n\) is the negative example, and \(\alpha\) is a margin.

   **Use Cases:**
   - Useful in tasks involving similarity learning, such as face recognition.

   **Avoid When:**
   - When training data does not provide adequate positive/negative pairs.

### Evaluation Metrics for Information Retrieval and Ranking

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
#### Description:
MAR@K measures the average recall of a model at the top K retrieved items. It is particularly useful in scenarios where only the top K results are considered relevant.

#### Advantage:
Focuses on the most relevant items, making it suitable for recommendation systems and information retrieval tasks.
Provides a clearer picture of recall when only a subset of results is of interest.

#### Disadvantage:
May overlook relevant items that are not in the top K.
Sensitive to the choice of K; different K values can yield different insights.

#### Best Suited For:
Scenarios where retrieving the top K relevant items is more important than retrieving all relevant items.

#### Example:
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

#### 3. Mean Average Precision (MAP)
#### Description:
MAP is the mean of average precision scores across multiple queries. Average precision summarizes the precision-recall curve for a single query and considers the order of predicted results.

#### Advantage:
Considers both precision and the rank of positive instances, providing a nuanced evaluation.
Useful for evaluating ranked retrieval tasks.
#### Disadvantage:
Requires careful computation, as it involves precision at each relevant item in the ranking.
May be sensitive to the number of queries in the dataset.

#### Best Suited For:
Information retrieval tasks where both the order and relevance of items matter.

#### Example:
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

#### 4. Mean Reciprocal Rank (MRR)
#### Description:
MRR measures the average of the reciprocal ranks of the first relevant item across multiple queries. It emphasizes the importance of retrieving the relevant item as early as possible.

#### Advantage:
Simple to compute and interpret.
Highlights the effectiveness of retrieval systems in providing relevant results early in the ranking.
#### Disadvantage:
Only considers the first relevant item, which may not provide a comprehensive view of the retrieval system's performance.
Sensitive to cases where there are no relevant items in the ranking.
#### Best Suited For:
Tasks where finding the first relevant item quickly is crucial, such as question-answering systems.
#### Example:
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
#### Description:
Normalized Discounted Cumulative Gain (NDCG) is a ranking metric that is widely used in information retrieval, such as in search engines and recommendation systems. It measures the usefulness (or "gain") of the results based on their relevance, while also considering the position of the results in the ranking. Higher-ranked items (those shown earlier) are given more importance compared to lower-ranked ones.

#### Key Concepts:

1. **Cumulative Gain (CG)**: This is the sum of the relevance scores of the retrieved items, without considering their positions.
   - Formula: 
   ```math
   CG = \sum_{i=1}^{k} rel_i

2. **Discounted Cumulative Gain (DCG)**: DCG penalizes the relevance scores based on their positions. Lower-ranked items are discounted, meaning their contribution to the overall score decreases as the rank increases.

3. **Ideal DCG (IDCG)**: This is the best possible DCG that could be obtained if the items were perfectly ranked according to their relevance. This is used to normalize the DCG.

4. **NDCG**: NDCG is the ratio of DCG to IDCG. It normalizes the score so that it lies between 0 and 1, where a higher score indicates better ranking.

#### Why is NDCG Important?
NDCG is especially useful when dealing with graded relevance, where items are not simply "relevant" or "irrelevant" but have varying degrees of relevance. By considering both the relevance of items and their positions in the ranking, NDCG provides a more nuanced evaluation of the ranking quality.


#### Advantage:
- Position-sensitive: Penalizes relevant items that are placed lower in the list, encouraging better ordering of items.
- Graded relevance: Handles varying levels of relevance, unlike binary relevance metrics like Precision or Recall.
- Normalized: Scores are normalized between 0 and 1, making them comparable across queries or datasets.

#### Disadvantage:
Requires relevance scores, which may not always be available.
Complexity in implementation compared to simpler metrics.

#### Best Suited For:
Ranking tasks in information retrieval where graded relevance is available.

#### Example:
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

#### Example:

Suppose we have a list of 5 retrieved items with the following relevance scores:

- Ground truth relevance: `[3, 2, 3, 0, 1]`
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
#### Description:
Cumulative Gain measures the total relevance score of the retrieved items, regardless of their rank. It sums the relevance scores of the top K results.

#### Advantage:
Simple and intuitive to calculate, providing a straightforward measure of total relevance.
Useful for understanding overall retrieval effectiveness.
#### Disadvantage:
Ignores the rank of items, meaning it can give a false sense of performance if lower-ranked items are highly relevant.
#### Best Suited For:
Situations where the overall relevance of retrieved items is more important than their order.
#### Example:
```python
def cumulative_gain(y_true, k):
    return np.sum(y_true[:k])

y_true = [3, 2, 3, 0, 1, 2]  # Relevance scores
cg_score = cumulative_gain(y_true, k=3)
```



### Normalized Discounted Cumulative Gain (NDCG) Based on Click Data

**Click data** can be used as a proxy for ground truth relevance scores, especially when explicit relevance labels (e.g., ratings or user feedback) are unavailable. In recommendation systems, clicks indicate user interest, with more clicks suggesting higher relevance. Here’s how click data can be transformed and used as ground truth (GT) relevance:

#### Transforming Click Data into Ground Truth (GT) Relevance:
- **Clicks as binary relevance**: If a user clicks on an item, it can be labeled as relevant (1), while non-clicked items are considered irrelevant (0).
- **Clicks as graded relevance**: You can assign higher relevance scores based on the number of clicks or interactions with an item. For instance:
  - 3+ clicks = highly relevant (relevance score 3)
  - 2 clicks = relevant (relevance score 2)
  - 1 click = somewhat relevant (relevance score 1)
  - 0 clicks = irrelevant (relevance score 0)

### Example: Calculating NDCG Based on Click Data

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

### Interpretation:
- The NDCG score is approximately **0.946**, indicating that the ranking generated by the model is close to the ideal ranking.
- This score suggests that the model’s predictions align well with the ground truth relevance derived from click data.

### Use of Click Data as Ground Truth:
1. **Advantages**:
   - **Implicit feedback**: Click data is automatically collected, so there's no need to rely on explicit feedback (like ratings or reviews).
   - **Reflects user interest**: Clicks represent user interactions and interest, making them a strong signal of relevance in many cases.

2. **Challenges**:
   - **Noisy signals**: Clicks may not always represent true interest or relevance (e.g., accidental clicks).
   - **Cold start problem**: Users with no click history or new items may not have any click data, making it difficult to assess relevance.

### Conclusion:
Click data is a valuable source of implicit feedback that can be transformed into ground truth relevance for ranking tasks in recommendation systems. NDCG is an effective metric to evaluate the quality of rankings generated by models that use click-based relevance.


### Sampling Techniques

In statistical analysis and machine learning, sampling techniques are used to select a subset of data from a larger population. These techniques allow for efficient computation and generalization. Below are some common sampling techniques, with examples where applicable.

---

#### 1. Random Sampling

#### Description:
Random Sampling is the simplest sampling technique where each data point in the population has an equal chance of being selected. It helps to ensure that the sample represents the population without bias.

#### Example:
Suppose you have a dataset of 1000 customer transactions. To randomly select 100 transactions for analysis:

```python
import random

# Data of 1000 customer transactions
transactions = list(range(1000))

# Randomly selecting 100 transactions
random_sample = random.sample(transactions, 100)
```

This method is unbiased and works well when the population is homogeneous.

#### Advantages:
Easy to implement
Unbiased if the population is uniform
#### Disadvantages:
May not work well for non-homogeneous populations
Sample may not represent smaller subgroups effectively

#### 2. Rejection Sampling
#### Description:
Rejection Sampling is a technique where samples are drawn from a proposal distribution, and then accepted or rejected based on how well they fit the target distribution. It is commonly used in probabilistic models and Monte Carlo simulations.

#### Example:
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

#### Advantages:
Effective for complex distributions
Flexible and adaptable to various target distributions
#### Disadvantages:
Inefficient if many samples are rejected
Requires a well-designed proposal distribution

#### 3. Weight Sampling (Weighted Random Sampling)
#### Description:
Weight Sampling involves selecting samples based on their assigned weights, giving more importance to some data points over others. Each data point has a probability proportional to its weight.

#### Example:
Suppose you have a list of items with corresponding weights:

```python
import random

items = ['A', 'B', 'C', 'D']
weights = [0.1, 0.3, 0.5, 0.1]

####  Select 1 item with weight-based probability
weighted_sample = random.choices(items, weights, k=1)
```

#### Advantages:
Useful when some data points are more important than others
Reduces bias toward less important data points
#### Disadvantages:
Requires accurate weighting of data
Weight assignment may be subjective

#### 4. Importance Sampling
#### Description:
Importance Sampling is a variance reduction technique used in Monte Carlo simulations. It involves drawing samples from a different (usually easier) distribution and adjusting for the difference by weighting the samples. The goal is to estimate properties of a distribution while sampling from a simpler distribution.

#### Example:
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
#### Advantages:
Reduces variance in estimates
More efficient than brute-force sampling
#### Disadvantages:
Choosing an appropriate proposal distribution is challenging
Can lead to high variance if the weights vary significantly

#### 5. Stratified Sampling
#### Description:
Stratified Sampling involves dividing the population into distinct subgroups (strata) and sampling from each stratum proportionally. This ensures that each subgroup is adequately represented in the sample.

#### Example:
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
#### Advantages:
Ensures representation from all subgroups
Reduces variability within each stratum
#### Disadvantages:
Requires prior knowledge of strata
More complex than simple random sampling

#### 6. Reservoir Sampling
#### Description:
Reservoir Sampling is used to sample a fixed number of items from a stream of data of unknown size, ensuring that each item has an equal probability of being included. It's efficient and works well with large datasets.

#### Example:
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
#### Advantages:
Works efficiently with large datasets
No need to know the size of the data stream in advance
#### Disadvantages:
Limited to uniform sampling
May not work well for biased or weighted sampling needs
#### Conclusion:
Each sampling technique has its own advantages and limitations, depending on the type of data and the goal of the analysis. For simple data, random sampling may be sufficient, but for more complex datasets or streams, methods like stratified sampling and reservoir sampling may be more appropriate.

### Deep Cross Network with parallel architecture

In the context of neural networks and machine learning models, Deep & Cross Layers with Parallel Architecture refers to a specific architectural design used in models such as Deep & Cross Networks (DCN) for recommendation systems and other tasks involving tabular data. This architecture is designed to combine the strengths of deep learning and feature crossing techniques to better capture both high-order feature interactions and non-linear relationships in data.

#### Deep Layers:
Deep layers refer to the deep neural network (DNN) part of the architecture. These layers consist of multiple fully connected layers stacked on top of each other. Each layer applies a linear transformation followed by a non-linear activation function. The deep layers are good at learning complex, non-linear representations of the input features.

#### Key properties:

Complexity: The deep layers help capture non-linear and intricate patterns from the input features.
Feature Interaction: Deep layers automatically learn interactions between features through multiple transformations.
Typical Layers: Fully connected (dense) layers with activations like ReLU or sigmoid.


#### Cross Layers:
Cross layers focus on capturing explicit feature interactions by taking the Cartesian product of features at different levels. They are designed to efficiently learn cross-feature interactions without the need to manually specify the interactions. These layers perform a feature crossing operation, where the feature vectors from the input are multiplied with themselves at different depths to create new combined features.

#### Key properties:

Explicit Feature Interaction: Unlike deep layers, which learn interactions implicitly, cross layers capture explicit feature crossings.
Higher-Order Feature Interaction: Cross layers are effective in modeling high-order feature interactions in a more controlled and efficient manner.
Cross Product: These layers iteratively compute cross products between raw input features and transformed features.


#### Parallel Architecture:
In models with parallel architecture, both deep and cross layers are applied simultaneously to the input data, allowing the model to capture both types of relationships—explicit feature interactions (cross layers) and non-linear transformations (deep layers)—in parallel. The outputs of both types of layers are combined (e.g., concatenated) at the final stage to make predictions.

#### Key properties:

Parallel Processing: Deep and cross layers operate on the input data in parallel, and their outputs are fused later on.
Hybrid Strength: This structure leverages the strengths of both deep learning (complex non-linear patterns) and feature crossing (explicit interactions).
Effective for Tabular Data: This design is highly effective in domains like recommendation systems and click-through-rate (CTR) prediction, where high-order interactions between categorical and numerical features are crucial.

#### Example of Architecture:
#### Input Features: The model takes raw input features such as user behavior, product features, and contextual data.

####  Deep Layers:

Multiple dense layers process the input data.
Each layer applies a linear transformation followed by an activation function (e.g., ReLU).
#### Cross Layers:

Feature interactions are explicitly calculated by taking products of the input features at various stages.
Cross layers repeatedly combine the transformed feature vectors with the raw input in a multiplicative manner.
#### Parallel Structure:

The deep layers and cross layers operate in parallel.
Their outputs are then concatenated or merged and passed to a final output layer for prediction.

#### Applications:
This parallel design is particularly useful in recommendation systems (e.g., Google Play, YouTube), click-through-rate (CTR) prediction, and other domains where interactions between categorical and continuous variables are important for accurate predictions.

#### Summary:
Deep Layers: Learn complex, non-linear patterns in data.
Cross Layers: Learn explicit, high-order feature interactions.
#### Parallel Architecture: Combines both approaches to capture a wide range of interactions in data efficiently.

#### Sample code

Let's assume we have a dataset with two features, X1 and X2, and a binary target y for a recommendation task.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Example dataset: 100 samples, 2 features
X = torch.randn(100, 2)
y = torch.randint(0, 2, (100, 1)).float()

# DCN Architecture
class DCN(nn.Module):
    def __init__(self, input_dim, deep_dims, cross_layers):
        super(DCN, self).__init__()
        
        # Deep part
        deep_layers = []
        for dim in deep_dims:
            deep_layers.append(nn.Linear(input_dim, dim))
            deep_layers.append(nn.ReLU())
            input_dim = dim
        self.deep = nn.Sequential(*deep_layers)
        
        # Cross part
        self.cross_layers = cross_layers
        self.cross_w = nn.ModuleList([nn.Linear(input_dim, 1, bias=False) for _ in range(cross_layers)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(cross_layers)])
        
        # Final output layer
        self.fc = nn.Linear(input_dim + deep_dims[-1], 1)
    
    def forward(self, x):
        # Cross layers
        cross = x.clone()
        for i in range(self.cross_layers):
            cross = x * self.cross_w[i](cross) + cross + self.cross_b[i]

        # Deep layers
        deep = self.deep(x)

        # Concatenate deep and cross layers
        out = torch.cat([cross, deep], dim=-1)
        out = self.fc(out)
        return torch.sigmoid(out)

# Model, loss, optimizer
model = DCN(input_dim=2, deep_dims=[64, 32], cross_layers=3)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop (basic example)
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')
```

#### Breakdown:
##### Deep Layers: A simple fully connected feed-forward network with ReLU activations.
##### Cross Layers: Explicit feature interactions calculated iteratively.
##### Final Output: Concatenates the outputs of both deep and cross layers, and then a final linear layer is used for binary classification.


###  How Instagram “blend” video and photos in their recommendations?
When Instagram recommends videos and photos to users, it needs a way to combine or "blend" these two different types of content in its feed or suggestions. But videos and photos might be interacted with differently by users—some might get more likes, while others might get more comments or views. To fairly compare and recommend content from both types, Instagram uses a mathematical trick to "normalize" these interactions.

Here's how Instagram blends videos and photos:
Mapping User Interactions to a Probability: Instagram first takes the interaction a user has with content (like clicking, viewing, or liking) and calculates the probability that the user will perform that action, such as a p(like)—the probability that a user will like a video or a photo.

Gaussian Distribution: A Gaussian distribution (or bell curve) is a common way to describe how data points (in this case, the interactions like likes) are spread out. Gaussian distributions are "well-behaved" in the sense that they make it easier to compare data. If Instagram maps the probability of an action (like or view) onto a Gaussian curve, it creates a standardized way to measure and compare different types of content.

Why is this important?: Videos might get fewer likes but more views, while photos might get lots of likes but fewer views. If Instagram simply compared the raw number of likes or views, it wouldn't be a fair comparison because videos and photos are interacted with differently. By mapping these interactions to a Gaussian curve, Instagram normalizes the data so that it can compare them in a standardized way.

Blending Recommendations: Now that both videos and photos have been mapped to the same type of distribution, Instagram can blend the recommendations. This allows the platform to present a balanced mix of videos and photos that are both likely to engage the user, based on the probabilities and interactions Instagram has normalized.

Imagine Instagram sees that you tend to like 60% of the photos you come across, but only 30% of the videos. Instagram doesn’t want to only show you photos because they get more likes—there’s still a good chance you’d enjoy some of those videos! So, instead of just looking at the raw percentage, it maps both your photo likes and video likes onto a Gaussian curve to make a more even comparison. Now, Instagram can recommend both photos and videos to you in a balanced way, factoring in your preferences for each in a fair manner.

In the end, this process makes sure that the content you see is a well-rounded mix of both videos and photos that are likely to engage you.

#### A/B Testing

#### Normal A/B Testing
In traditional A/B testing, two versions (A and B) of a product feature or webpage are compared to determine which performs better. A portion of users is randomly assigned to version A, and another portion is assigned to version B. Metrics such as click-through rate or conversion rate are measured, and statistical tests are used to determine which version performs best.

#### Example:
- **Version A**: Original homepage with a "Sign Up" button.
- **Version B**: New homepage with a "Get Started" button.
- If version B increases sign-ups by 10%, it may be selected as the better option.

### Budget-Splitting A/B Testing
Budget-splitting A/B testing involves allocating different portions of a testing budget to multiple variants based on performance over time. Instead of splitting traffic equally, traffic is dynamically allocated to the variant that shows higher potential, maximizing returns while the test is still running.

#### Example:
- **Version A**: 40% of users (initial budget allocation).
- **Version B**: 60% of users (because it shows higher conversions after early results).
- If version B continues to outperform, more budget/traffic is allocated to it to maximize results during the test.


### Ranking Approaches in Machine Learning

#### Example Dataset:
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

#### Training Data:
- `Doc A`: Label = 3
- `Doc B`: Label = 1
- `Doc C`: Label = 2

We train a model to predict the relevance score for each document individually.

#### Prediction:
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

#### Pairs for Training:
- Compare `Doc A` and `Doc B`: Label = A is higher than B (3 > 1)
- Compare `Doc A` and `Doc C`: Label = A is higher than C (3 > 2)
- Compare `Doc B` and `Doc C`: Label = C is higher than B (2 > 1)

We train a model to predict the relative ranking between pairs of documents.

#### Prediction:
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

#### Pairs for Training:
- `Doc A` vs `Doc B`: True Label = `Doc A` is higher (3 > 1)
- `Doc A` vs `Doc C`: True Label = `Doc A` is higher (3 > 2)
- `Doc C` vs `Doc B`: True Label = `Doc C` is higher (2 > 1)

The model is trained using these pairwise comparisons. It outputs probabilities based on which document should be ranked higher, using a neural network to predict scores for each document.

#### Prediction:
The model predicts the following probabilities:
- `Doc A` is ranked higher than `Doc B` with probability 0.9.
- `Doc A` is ranked higher than `Doc C` with probability 0.8.
- `Doc C` is ranked higher than `Doc B` with probability 0.85.

The predicted ranking would be: `Doc A`, `Doc C`, `Doc B`.

#### Summary:
- **Point-wise:** Each item is treated independently, simple but doesn't optimize ranking directly.
- **Point-wise Approach** predicts relevance scores directly for each document and ranks them.

- **Pairwise:** Focuses on comparing pairs of items, optimizes ranking but can be computationally expensive.
- **Pairwise Approach** compares documents in pairs and learns which document should be ranked higher.

- **RankNet:** A neural network-based pairwise model that predicts relative rankings using probabilities.
- **RankNet** uses a neural network to predict the relative ranking between pairs of documents, outputting probabilities that one document is ranked higher than another.

#### Similarity Functions
#### 1. Euclidean Distance
Euclidean distance is the straight-line distance between two points in multi-dimensional space. It is often used in clustering (e.g., k-means) and nearest-neighbor algorithms.

#### Pros:
Intuitive: Represents the actual geometric distance between points, easy to understand.
Effective in low dimensions: Works well when the number of dimensions is small.
#### Cons:
Sensitive to scale: If features are on different scales (e.g., age in years vs. height in centimeters), this can distort the distance. Normalization is needed.
Curse of dimensionality: As the number of dimensions increases, Euclidean distance loses effectiveness due to all points appearing equidistant.

#### 2. Cosine Similarity
Cosine similarity measures the cosine of the angle between two non-zero vectors. It is commonly used in text analysis to measure the similarity of documents.

#### Pros:
Scale-invariant: Focuses on the direction rather than the magnitude, so it's useful when comparing high-dimensional data like text, where the magnitude (e.g., document length) can vary.
Works well for sparse data: Effective in cases where vectors are sparse, such as in document term matrices.
#### Cons:
Ignores magnitude: If the magnitude of vectors matters (i.e., the size of the values), cosine similarity might not be suitable since it only considers the angle.
Not appropriate for negative values: Works best when feature values are non-negative, such as in word count vectors or TF-IDF matrices.

#### 3. Manhattan Distance (L1 Norm)
Also known as "taxicab" or "city block" distance, Manhattan distance calculates the sum of the absolute differences between the coordinates of two points.

#### Pros:
Robust to outliers: Less sensitive to large differences compared to Euclidean distance, which is affected by squared values.
Works in high dimensions: Often more effective in high-dimensional spaces than Euclidean distance.
#### Cons:
Not as intuitive: The geometric meaning of this distance can be less intuitive, especially in non-grid-like data.
Sensitive to feature scaling: Like Euclidean distance, it requires normalization of features to avoid biasing toward features with larger ranges.

#### 4. Jaccard Similarity
Jaccard similarity is used for comparing the similarity and diversity of sets. It is the ratio of the intersection to the union of two sets. Commonly used in binary or categorical data.

#### Pros:
Effective for set-based similarity: Useful in applications involving set comparison, such as recommendation systems, document comparison, or binary features.
Good for sparse data: Works well when the data is sparse or binary.
#### Cons:
Ignores frequency information: Jaccard does not consider how many times an item appears; it only looks at whether it appears or not (presence/absence).
Sensitive to small sets: If one or both sets are small, Jaccard similarity can be misleading since small differences are amplified.

#### 5. Hamming Distance
Hamming distance measures the number of positions at which two strings of equal length differ. It is typically used for categorical variables and binary strings.

#### Pros:
Simple to compute: Works well for binary and categorical data.
Useful for exact match tasks: Ideal for cases where small deviations in sequences matter, such as in DNA sequences or binary data.
#### Cons:
Not suitable for continuous variables: Designed for binary or categorical data, it doesn’t work well when the features are continuous.
Sensitive to length: Requires strings or vectors to have the same length.

#### 6. Minkowski Distance
The Minkowski distance is a generalized metric that encompasses both Euclidean and Manhattan distances. It's defined as:

$$
d(x, y) = \left(\sum_{i=1}^{n} |x_i - y_i|^p\right)^{1/p}
$$

where:

x and y are two points in n-dimensional space.
p is a positive integer that determines the type of distance:
p = 1: Manhattan distance (also known as L1 norm)
p = 2: Euclidean distance (also known as L2 norm)
By varying the value of p, you can explore different distance metrics that may be more suitable for specific applications. For example, Manhattan distance might be more appropriate for data with a grid-like structure (like city blocks), while Euclidean distance is better suited for continuous spaces.

#### Pros:
Flexibility: With the parameter 
𝑝, you can interpolate between Manhattan and Euclidean distances based on the problem at hand.
General-purpose: Can be adapted to a wide range of scenarios depending on how the 
p-norm is set.
#### Cons:
Interpretability: Can be harder to interpret, especially when p ≠ 1 or p ≠ 2.
Sensitivity to p: The performance can vary widely based on the choice of p, and it may require tuning.

#### 7. Mahalanobis Distance
Mahalanobis distance measures the distance between two points while accounting for correlations in the dataset. It is a generalized form of Euclidean distance.

Pros:
Accounts for correlations: Useful when there are relationships between features, as it takes into account the covariance between variables.
Works in multivariate data: Suitable for situations where the features are interrelated.
Cons:
Requires the inverse covariance matrix: Computing this matrix can be computationally expensive, especially in high dimensions or when the matrix is singular.
Sensitive to outliers: If the dataset contains outliers, they can skew the covariance matrix, distorting the distance measure.

#### 8. Pearson Correlation
Pearson correlation measures the linear relationship between two variables. It ranges from -1 (perfect negative correlation) to 1 (perfect positive correlation).

#### Pros:
Simplicity: Easy to compute and interpret.
Linear relationship: Works well when the relationship between variables is linear.
#### Cons:
Sensitive to outliers: Pearson correlation can be heavily influenced by outliers.
Assumes linearity: Does not capture non-linear relationships.

#### 9. Spearman Correlation
Spearman correlation measures the rank correlation between two variables. It assesses how well the relationship between two variables can be described by a monotonic function.

#### Pros:
Non-parametric: Does not assume a linear relationship between variables, capturing monotonic relationships.
Less sensitive to outliers: Since it uses ranks, it's more robust to outliers compared to Pearson correlation.

#### Cons:
Ignores magnitude of differences: Only looks at rank, not the actual differences between values.
Less interpretable in some cases: Can be harder to interpret compared to Pearson correlation when trying to understand the strength of association.

#### Summary Table

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


### ML Model Implemented from Scratch
#### Linear Regression
[Linear Regression Notebook](notebooks/linear_regression.ipynb)


#### Logistic Regression
[Logistic Regression Notebook](notebooks/logistic_regression.ipynb)

