import re
from bs4 import BeautifulSoup

corpus = [
    "<p>Hello! Visit https://example.com</p> Email: test@gmail.com",
    "NLP is GREAT!!! Call 12345 today. #NLP"
]

cleaned_corpus = []

for text in corpus:
    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    cleaned_corpus.append(text)

print("Original Corpus:")
print(corpus)

print("\nCleaned Corpus:")
print(cleaned_corpus)

print("\nComparison:")
for original, cleaned in zip(corpus, cleaned_corpus):
    print("Before:", original)
    print("After :", cleaned)
