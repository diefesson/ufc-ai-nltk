import nltk
import questions.q1 as q1

DEMOS = [
    ("download data", nltk.download),
    ("q1", q1.demo)
]


def main():
    for i in range(len(DEMOS)):
        name = DEMOS[i][0]
        print(f"{i} - {name}")
    choice = int(input())
    DEMOS[choice][1]()


if __name__ == "__main__":
    main()
