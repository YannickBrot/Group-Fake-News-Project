import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


nltk.download('punkt')
nltk.download('stopwords')
from cleantext import clean
import re
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv'

news_sample = pd.read_csv(url)
unique_words_set_before = set()
unique_words_set_after_cleaning = set()
unique_words_set_after_tokenization = set()
unique_words_set_after_stop_word_removal = set()
unique_words_set_after_stemming = set()
stop_words = set(stopwords.words('english'))

for content in news_sample['content']:
    content = content.split(' ')
    unique_words_set_before.update(content)


def clean_with_cleantext(text):
    text = clean(text,
                 lower=True,
                 no_urls=True,
                 no_emails=True,
                 no_numbers=True,
                 no_digits=True,
                 replace_with_url="",
                 replace_with_email="",
                 replace_with_number="",
                 replace_with_digit="",
                 lang="en")
    text = re.sub(r'[-\/]', ' ', text)
    text = re.sub(r'[!\+\/@#$%^&?!=\<\>_*.,€:;\[\]\{\}\'\"\´\¨\(\)\\]', '', text)
    return text


news_sample['content'] = news_sample['content'].apply(clean_with_cleantext)

for content in news_sample['content']:
    after_clean = content.split(' ')
    unique_words_set_after_cleaning.update(after_clean)
    content = content.lower()
    content = nltk.word_tokenize(content)
    unique_words_set_after_tokenization.update(content)
    content = [token for token in content if token not in stop_words]
    unique_words_set_after_stop_word_removal.update(content)
    porter = nltk.PorterStemmer()
    content = [porter.stem(token) for token in content]
    unique_words_set_after_stemming.update(content)

len_before = len(unique_words_set_before)
len_after_clean = len(unique_words_set_after_cleaning)
len_after_token = len(unique_words_set_after_tokenization)
len_after_stopwords = len(unique_words_set_after_stop_word_removal)
len_after_stemming = len(unique_words_set_after_stemming)

plt.figure(figsize=(12, 6))
plt.bar(['Before', 'After cleaning', 'After tokenization', 'After removing stopwords', 'After stemming'],
        [len_before, len_after_clean, len_after_token, len_after_stopwords, len_after_stemming])
plt.xlabel('step')
plt.ylabel('number of unique words')

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')
plt.tight_layout() # Prevent labels from overlapping

plt.show()


def split_dataset(df, test_size=0.1, random_state=42):
    """
    Split the dataset into training, validation, and test sets.
    """
    # Split into train+val and test
    train_val, test = train_test_split(df, test_size=test_size, random_state=random_state)
    
    # Split train+val into train and val
    train, val = train_test_split(train_val, test_size=test_size/(1-test_size), random_state=random_state)
    
    return train, val, test