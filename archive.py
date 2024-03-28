def normalize_vectors(vectors):
    """
    Normalize each vector in the list of vectors.
    Each vector is divided by its L2 norm. The vectors are assumed to be 11-dimensional,
    with the last element being the length of the text.
    """
    normalized = np.array(vectors, dtype=float)  # Ensure the array is in float for division
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Prevent division by zero
    return normalized / norms


def count_term_occurrences_as_vectors(text):
    words = ['said', 'mr', 'new', 'iran', 'iranian', 'year', 'york', 'rec', 'main', 'newslett']
    if isinstance(text, str):
        counts = [0] * len(words)
        for i, word in enumerate(words):
            pattern = re.compile(r'\b' + word + r'\b', re.IGNORECASE)
            counts[i] = len(pattern.findall(text))
        counts.append(len(text))  # Append the length of the text as the 11th dimension
        return counts
    else:
        return [0] * (len(words) + 1)  # Include a slot for text length

# Assuming X_train and X_val are defined and are arrays of text data
X_train_count_terms = np.array([count_term_occurrences_as_vectors(text) for text in X_train])
X_val_count_terms = np.array([count_term_occurrences_as_vectors(text) for text in X_val])

# Normalize the vectors
X_train_count_terms_normalized = normalize_vectors(X_train_count_terms)
X_val_count_terms_normalized = normalize_vectors(X_val_count_terms)

models = [
    'Logistic Regression',
    'Logistic Regression Normalized'
]

accuracies = [
    logistic_regression_performance(X_train_count_terms, y_train, X_val_count_terms, y_val),
    logistic_regression_performance(X_train_count_terms_normalized, y_train, X_val_count_terms_normalized, y_val,
                                    saveName="logistic_regression_vector_features.joblib")
]

display_model_comparisons(models, accuracies, title="Logistic Regression with vectors for terms")

from tensorflow.keras import layers, regularizers
from tensorflow.keras import models as md
from tensorflow.keras import callbacks

# Model architecture
model = md.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_count_terms_normalized.shape[1],),
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks for early stopping and learning rate reduction
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# Train the model with validation data and callbacks
history = model.fit(X_train_count_terms_normalized, y_train,
                    validation_data=(X_val_count_terms_normalized, y_val),
                    epochs=100,  # Increased number of epochs with early stopping will regulate it
                    batch_size=32,  # Consider experimenting with batch size
                    callbacks=[early_stopping, reduce_lr])

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(X_val_count_terms_normalized, y_val)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Assuming models and accuracies lists exist for comparison purposes
models.append('Neural Net')
accuracies.append(val_accuracy)

# Assuming display_model_comparisons is a function to visualize model comparisons
display_model_comparisons(models, accuracies, title="Advanced Model Accuracies")
dump(model, "neural_net.joblib")

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load pre-trained word embeddings
print('creating embedding index')
start_time = time.time()
embedding_index = {}
with open('glove.6B.100d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

elapsed_time = time.time() - start_time
print(f"finished creating embedding index in: {elapsed_time:.3f} seconds")


# Assuming embedding_index is your dictionary containing word: embedding pairs from GloVe
# Function to convert articles into averaged word embeddings
def text_to_avg_vector(text_series, tokenizer, embedding_index, embedding_dim=100):
    vectors = []
    for text in text_series:
        tokens = tokenizer.texts_to_sequences([text])[0]
        article_vectors = [embedding_index.get(tokenizer.index_word[token], np.zeros(embedding_dim)) for token in
                           tokens]
        if article_vectors:
            vectors.append(np.mean(article_vectors, axis=0))
        else:
            vectors.append(np.zeros(embedding_dim))
    return np.array(vectors)


# Tokenize text
print('fitting tokenizer')
start_time = time.time()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
elapsed_time = time.time() - start_time
print(f"finished fitting tokenizer in: {elapsed_time:.3f} seconds")

# Convert each article to averaged word embeddings
print('converting content of articles to vectors')
start_time = time.time()
X_train_embedded = text_to_avg_vector(X_train, tokenizer, embedding_index)
X_val_embedded = text_to_avg_vector(X_val, tokenizer, embedding_index)
elapsed_time = time.time() - start_time
print(f"finished converting articles in: {elapsed_time:.3f} seconds")

# Pad sequences
print('padding')
start_time = time.time()

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_train_padded = pad_sequences(X_train_sequences, maxlen=max([len(x) for x in X_train_sequences]))
X_val_sequences = tokenizer.texts_to_sequences(X_val)
X_val_padded = pad_sequences(X_val_sequences, maxlen=max([len(x) for x in X_val_sequences]))

elapsed_time = time.time() - start_time
print(f"finished padding in: {elapsed_time:.3f} seconds")

models.append('Logistic regression embedded')
accuracies.append(logistic_regression_performance(X_train_padded, y_train, X_val_padded, y_val, simple=True,
                                                  saveName="logistic_regression_embedded.joblib"))
display_model_comparisons(models, accuracies, title="Advanced model accuracies")

print('Naive Bayes')
start_time = time.time()

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)  # Fit and transform with train data
X_val_tfidf = tfidf_vectorizer.transform(X_val)  # Only transform validation data

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Validation set accuracy
y_val_pred = model.predict(X_val_tfidf)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Set Accuracy: {val_accuracy}')
dump(model, 'nb_1_1.joblib')
dump(tfidf_vectorizer, 'vectorizer_1_1.joblib')

elapsed_time = time.time() - start_time
print(f"Finished Naive Bayes in: {elapsed_time:.3f} seconds")

print('Naive Bayes')
start_time = time.time()

tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)  # Fit and transform with train data
X_val_tfidf = tfidf_vectorizer.transform(X_val)  # Only transform validation data

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Validation set accuracy
y_val_pred = model.predict(X_val_tfidf)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Set Accuracy: {val_accuracy}')
dump(model, 'nb_2_2.joblib')
dump(tfidf_vectorizer, 'vectorizer_2_2.joblib')

elapsed_time = time.time() - start_time
print(f"Finished Naive Bayes in: {elapsed_time:.3f} seconds")

from scipy.sparse import hstack

print('loading vectorizers')
tfidf_unigram_vectorizer = load('vectorizer_1_1.joblib')
tfidf_bigram_vectorizer = load('vectorizer_2_2.joblib')


def vectorize_combined_unigram_bigram(X):
    X_unigram_tfidf = tfidf_unigram_vectorizer.transform(X)
    X_bigram_tfidf = tfidf_bigram_vectorizer.transform(X)
    return hstack([X_unigram_tfidf, X_bigram_tfidf])

print('combining training data vectors')
X_train_tfidf_combined = vectorize_combined_unigram_bigram(X_train)

print('combining validation data vectors')
X_val_tfidf_combined = vectorize_combined_unigram_bigram(X_val)

print('training model')
model = MultinomialNB()
model.fit(X_train_tfidf_combined, y_train)

print('making predictions')
y_val_tfidf_combined_pred = model.predict(X_val_tfidf_combined)
val_accuracy_tfidf_combined = accuracy_score(y_val, y_val_tfidf_combined_pred)
print(f'Validation Set Accuracy: {val_accuracy_tfidf_combined}')