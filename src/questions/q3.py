from nltk import load, FreqDist, word_tokenize, SnowballStemmer
from nltk.corpus import stopwords
from numpy import sqrt, sum

FILES = [
    "data/review1.txt",
    "data/review2.txt",
    "data/review3.txt",
]

POSITIVE_WORDS = "data/positive_words.csv"
NEGATIVE_WORDS = "data/negative_words.csv"


def preprocess(data):
    stemmer = SnowballStemmer("english")
    return [stemmer.stem(w.lower()) for w in word_tokenize(data) if w not in stopwords.words("english")]


def distance(bag1, bag2):
    vocab = set(bag1).union(bag2)
    intersection = set(bag1).intersection(bag2)
    up = sum(
        bag1[w] * bag2[w] for w in vocab
    )
    down = (
        sqrt(sum(bag1[w] * bag1[w] for w in vocab)) *
        sqrt(sum(bag2[w] * bag2[w] for w in vocab))
    )
    return (up / down), len(intersection)


def demo():
    positive_bag = FreqDist(
        preprocess(load(POSITIVE_WORDS, format="text"))
    )
    negative_bag = FreqDist(
        preprocess(load(NEGATIVE_WORDS, format="text"))
    )

    for f in FILES:
        words_bag = FreqDist(preprocess(load(f)))
        positive, positive_count = distance(words_bag, positive_bag)
        negative, negative_count = distance(words_bag, negative_bag)
        classification = (positive - negative) / (positive + negative)
        print(f"Arquivo: {f}")
        print(f"\tPositivo: {positive} ({positive_count} palavras)")
        print(f"\tNegativo: {negative} ({negative_count} palavras)")
        print(f"\tClassificação: {classification}")
