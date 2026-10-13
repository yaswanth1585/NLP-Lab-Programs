from gensim.models import Word2Vec

corpus = [
    "The movie was excellent and very interesting",
    "I enjoyed the movie and loved the story",
    "The film was amazing and entertaining",
    "The movie had a great story and actors",
    "I disliked the movie because it was boring",
    "The film was terrible and disappointing",
    "The movie was boring and too long",
    "I hated the film because the story was poor"
]

tokens = [sentence.lower().split() for sentence in corpus]

model = Word2Vec(
    sentences=tokens,
    vector_size=50,
    window=3,
    min_count=1,
    sg=0,
    epochs=100,
    seed=42
)

print("Vocabulary:")
print(model.wv.index_to_key)

print("\nMovie Embedding:")
print(model.wv["movie"])

print("\nFilm Embedding:")
print(model.wv["film"])

print("\nVector Dimension:", model.wv.vector_size)

print("\nTop 5 words similar to movie:")
print(model.wv.most_similar("movie", topn=5))

print("\nSimilarity movie-film:",
      round(model.wv.similarity("movie", "film"), 3))
print("Similarity movie-story:",
      round(model.wv.similarity("movie", "story"), 3))
