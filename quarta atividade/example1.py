from indexer import Indexer

d = [
    ["O peã e o caval são pec de xadrez. O caval é o melhor do jog."],
    ["A jog envolv a torr, o peã e o rei."],
    ["O peã lac o boi"],
    ["Caval de rodei!"],
    ["Polic o jog no xadrez."],
]
sw = ["a", "o", "e", "é", "de", "do", "no", "são"]
st = [" ", ",", ".", "!", "?"]
query = "xadrez peã caval torr"
R = [d[1], d[2]]


def run_case_01():
    search_engine = Indexer(documents=d, stopwords=sw, spliters=st, lang="portuguese")
    ip, apv = search_engine.calculate_metrics(query=query, R=R, exact=False)

    print(f"Interpolated Precision at Recall: {ip}")
    print(f"Average Precision: {apv}")
