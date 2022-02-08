from nltk.corpus import gutenberg
from numpy import average

FILES = [
    "shakespeare-caesar.txt",
    "shakespeare-hamlet.txt",
    "shakespeare-macbeth.txt",
]


def count_repetitions(words: list[str]) -> dict[str, int]:
    counts = {}
    for w in words:
        if w in counts:
            counts[w] = counts[w] + 1
        else:
            counts[w] = 1
    return counts


def demo():
    for f in FILES:
        words = gutenberg.words(f)
        sents = gutenberg.sents(f)
        word_count = len(words)
        sent_count = len(sents)
        repetition_count = count_repetitions(words)
        non_repeat_count = len(
            [w for w, c in repetition_count.items() if c == 1]
        )
        repeat_count = len(
            [w for w, c in repetition_count.items() if c > 1]
        )
        average_sentence_len = average([len(s) for s in sents])
        print(f"Arquivo {f}")
        print(f"\tPalavras:{word_count}")
        print(f"\tSentenças: {sent_count}")
        print(f"\tPalavras não repetidas: {non_repeat_count}")
        print(f"\tPalavras repetidas: {repeat_count}")
        print(f"\tMédia de palavras por sentença: {average_sentence_len}")
