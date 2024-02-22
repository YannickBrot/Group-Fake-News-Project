import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from cleantext import clean
import re


url = 'https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv'

news_sample = pd.read_csv(url)
unique_words_set_before = set()
unique_words_set_after_cleaning = set()
unique_words_set_after_tokenization = set()
unique_words_set_after_stop_word_removal = set()
unique_words_set_after_stemming = set()
stop_words = set(stopwords.words('english'))

for content in news_sample['content']:
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
    text = re.sub(r'[-\/]',' ', text)
    text = re.sub(r'[!\+\/@#$%^&?!=\<\>_*.,€:;\[\]\{\}\'\"\´\¨\(\)\\]', '', text)
    return text

news_sample['content'] = news_sample['content'].apply(clean_with_cleantext)

for content in news_sample['content']:
    unique_words_set_after_cleaning.update(content)
    content = content.lower()
    content = nltk.word_tokenize(content)
    unique_words_set_after_tokenization.update(content)
    content  = [token for token in content if token not in stop_words]
    unique_words_set_after_stop_word_removal.update(content)
    porter = nltk.PorterStemmer()
    content = [porter.stem(token) for token in content]
    unique_words_set_after_stemming.update(content)
