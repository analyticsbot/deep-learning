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

#### Explanation
The data consists of 3 categorical features: ad_campaign_id, user_location, and device.
We use FeatureHasher to convert these categorical features into a fixed-length feature vector of size 10 (n_features=10).
The hashed features reduce the memory usage compared to one-hot encoding while preserving some information about the original features.

#### Output
```
The hashed feature matrix will look like this:
[[ 0.  1. -1.  0.  1.  0.  0.  0. -1.  0.]
 [ 1.  0.  1.  1.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  0.  1.  0.  0.  0.  1.  0.  0.]
 [ 0.  2.  0.  0.  1.  0.  0.  0. -1.  0.]]
```
Each row represents the hashed features for a row of categorical data, which can be used as input to a machine learning model.

#### Requirements
Make sure you have scikit-learn installed:

```
pip install scikit-learn
```

#### Notes
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

#### Explanation
Cross Features: We combine the latitude and longitude values into a single feature called pickup_location.
Feature Hashing: Since this cross feature could have a large number of possible values (high cardinality), we use the hashing trick to convert it into a fixed-size feature vector of length 10 (n_features=10).
This helps reduce the complexity while keeping enough information to feed into a machine learning model.

#### Output
This will print a hashed feature matrix like this:
```[[ 0.  1. -1.  0.  1.  0.  0.  0. -1.  0.]
 [ 1.  0.  1.  1.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  0.  1.  0.  0.  0.  1.  0.  0.]
 [ 0.  2.  0.  0.  1.  0.  0.  0. -1.  0.]]
```

#### Why Use the Hashing Trick?
Cross features often create many new categories when you combine existing features, making the data too large or "high-dimensional." Using the hashing trick helps keep the data manageable while still using the important information in your machine learning model.


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

```
python
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

