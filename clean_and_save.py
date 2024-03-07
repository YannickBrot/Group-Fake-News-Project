import pandas as pd
import re
from cleantext import clean

# Specify the URL of the CSV file
url = 'https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv'

# Use pandas to read the CSV directly from the URL
df = pd.read_csv(url)

def clean_text(text):
    # Lowercase the text
    text = text.lower()
    # Replace URLs
    text = re.sub(r'(https?://)[^, \n]*', '<URL>', text)
    text = re.sub(r'(www.)[^, \n]*', '<URL>', text)
    # Replace emails
    text = re.sub(r'\S+@(\S+\.)+\S+', '<EMAIL>', text)
    # Replace dates (simple patterns, for demonstration)
    text = re.sub(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec) [0-9]{2,4}', '<DATE>', text)
    # Replace number
    text = re.sub(r'\d+(,\d+)*(\.\d+)?', '<NUM>', text)
    # Replace various consecutive whitespaces with just one
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'\t{2,}', '\t', text)
    return text


def clean_with_cleantext(text):
    if type(text) != str:
        text = str(text)
    text = text.lower()
    text = re.sub(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec) [0-9]{2,4}', '<DATE>', text)
    return clean(text,
                 lower=False,
                 no_urls=True,
                 no_emails=True,
                 no_digits=True,
                 no_numbers=True,
                 replace_with_url="<URL>",
                 replace_with_email="<EMAIL>",
                 replace_with_number="<NUM>",
                 replace_with_digit="<NUM>",
                 normalize_whitespace=True
                 )


df['content'] = df['content'].apply(clean_with_cleantext)

df.to_csv("sample_cleaned.csv", index=False)