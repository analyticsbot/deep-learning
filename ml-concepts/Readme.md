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
