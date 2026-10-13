import re
from bs4 import BeautifulSoup

reviews = [
    "<p>This product is AMAZING!!!</p> Visit https://shop.com",
    "The quality is very good 😊 and I paid 999 rupees.",
    "Not satisfied with the product. Contact: user@gmail.com",
    "Excellent product!!! Delivery was FAST. Order 12345.",
    "I really LOVE this phone!!! Great camera & battery. #BestPhone"
]

cleaned_reviews = []

for review in reviews:
    text = review.lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    cleaned_reviews.append(text)

for i in range(5):
    print("Review", i + 1)
    print("Original:", reviews[i])
    print("Cleaned :", cleaned_reviews[i])
    print()
