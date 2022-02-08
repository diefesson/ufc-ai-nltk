from nltk import load
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree


FILES = [
    "data/apoloxi.txt",
    "data/french-revolution.txt",
]
TYPES = ["GPE", "LOCATION", "PERSON"]


def demo():
    for f in FILES:
        types_count = {e: 0 for e in TYPES}
        corpora = load(f)
        tree = ne_chunk(pos_tag(word_tokenize(corpora)))
        for e in tree:
            if type(e) == Tree:
                label = e.label()
                if label in TYPES:
                    types_count[label] = types_count[label] + 1
        print(f"Arquivo: {f}")
        for t, c in types_count.items():
            print(f"\t instancias de {t}: {c}")
