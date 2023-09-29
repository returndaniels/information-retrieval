from indexer import Indexer

d = [
    ["Parasita é o grande vencedor do Oscar 2020, com quatro prêmios"],
    ["Green Book, Roma e Bohemian Rhapsody são os principais vencedores do Oscar 2019"],
    [
        "Oscar 2020: Confira lista completa de vencedores. Parasita e 1917 foram os grandes vencedores da noite"
    ],
    ["Em boa fase, Oscar sonha em jogar a Copa do Mundo da Rússia"],
    [
        "Conheça os indicados ao Oscar 2020; Cerimônia de premiação acontece em fevereiro"
    ],
    [
        "Oscar Schmidt receberá Troféu no Prêmio Brasil Olímpico 2019. Jogador de basquete com mais pontos em Jogos Olímpicos."
    ],
    [
        "Seleção brasileira vai observar de 35 a 40 jogadores para definir lista da Copa América"
    ],
    ["Oscar 2020: saiba como é a escolha dos jurados e como eles votam"],
    ["Bem, Amigos! discute lista da Seleção, e Galvão dá recado a Tite: Cadê o Luan?"],
    ["IFAL-Maceió convoca aprovados em lista de espera do SISU para chamada oral"],
    [
        "Arrascaeta e Matías Viña são convocados pelo Uruguai para eliminatórias da Copa. Além deles, há outros destaques na lista."
    ],
    ["Oscar do Vinho: confira os rótulos de destaque da safra 2018"],
    ["Parasita é o vencedor da Palma de Ouro no Festival de Cannes"],
    ["Estatísticas. Brasileirão Série A: Os artilheiros e garçons da temporada 2020"],
    ["Setembro chegou! Confira o calendário da temporada 2020/2021 do futebol europeu"],
    ["Cerimônia do Oscar 2021 é adiada devido à pandemia de COVID-19"],
    [
        "A ascensão do streaming: filmes da Netflix e Amazon Prime ganham destaque no Oscar"
    ],
    ["Atores icônicos que nunca ganharam um Oscar"],
    ["Oscar 2020: Discurso emocionante de Joaquin Phoenix sobre meio ambiente"],
    ["Copa do Mundo da Rússia 2018: França conquista o título"],
    ["Novos talentos do cinema indie recebem destaque no Festival de Sundance"],
    ["Futebol brasileiro: Flamengo conquista a Copa Libertadores 2019"],
    ["Crescimento da audiência: Oscar ganha mais telespectadores em 2020"],
    ["Grandes diretores de cinema que nunca ganharam um Oscar"],
    ["Oscar 2020: Destaque para os filmes estrangeiros"],
]

sw = [
    "a",
    "o",
    "e",
    "é",
    "de",
    "do",
    "da",
    "no",
    "na",
    "são",
    "dos",
    "com",
    "como",
    "eles",
    "em",
    "os",
    "ao",
    "para",
    "pelo",
]
st = [" ", ",", ".", "!", "?", ":", ";", "/"]

if __name__ == "__main__":
    search_engine = Indexer(documents=d, stopwords=sw, spliters=st, lang="portuguese")
    results = search_engine.search(query="Parasita oscar", exact=True)

    documents = results[0]
    # query_weights = results[1]
    # document_weights = results[2]
    # scored_docs = search_engine.classify_documents(
    #     documents, query_weights, document_weights
    # )

    # print("Respostas classificadas por TF-IDF\n")

    # index = 0
    # for doc, score in scored_docs:
    #     print(index, f": score = {score} \n\t {doc}\n")
    #     index += 1

    # results = search_engine.search_with_vector_model(query="Parasita oscar", exact=True)
    ranked_documents = search_engine.rank_documents(
        query="Parasita oscar", documents=documents
    )

    print(ranked_documents)
