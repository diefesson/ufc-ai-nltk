import nltk
import questions.q1 as q1
import questions.q2 as q2

DEMOS = [
    ("download data", nltk.download),
    ("q1", q1.demo),
    ("q2", q2.demo),
]


def main():
    for i in range(len(DEMOS)):
        name = DEMOS[i][0]
        print(f"{i} - {name}")
    choice = int(input())
    DEMOS[choice][1]()


if __name__ == "__main__":
    main()
