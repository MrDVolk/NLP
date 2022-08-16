import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup

# nltk.download('stopwords')
english_stopwords = stopwords.words('english')
english_stopwords = set(english_stopwords)


def remove_html(row):
    row = BeautifulSoup(row, 'html.parser').get_text()
    return row


def remove_stop_words(row):
    words = row.split(' ')
    row = ' '.join([word for word in words if word not in english_stopwords])
    return row


def remove_email(row):
    row = re.sub(r'\S*@\S*\s?', '', row)
    return row


def remove_hyperlinks(row):
    row = re.sub(r'https?://[^\s\n\r]+', '', row)
    return row


def preprocess_text(row):
    row = remove_html(row)

    row = row.lower()
    row = row.replace('\n', ' ')
    row = row.replace('\t', ' ')

    row = remove_email(row)
    row = remove_hyperlinks(row)
    row = remove_stop_words(row)

    row = re.sub(r'[^a-z ]', ' ', row)
    row = re.sub(r'[a-z]{35,}', ' ', row)
    row = re.sub(r' {2,}', ' ', row)
    row = row.strip()

    return row


def tokenize(row, *, stem=False):
    tokens = word_tokenize(row)
    if stem:
        stemmer = PorterStemmer()
        stemmed_tokens = []
        for token in tokens:
            stemmed_tokens.append(stemmer.stem(token))

        return stemmed_tokens

    return tokens
