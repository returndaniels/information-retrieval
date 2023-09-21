from indexer import Indexer

d = [['Parasita é o grande vencedor do Oscar 2020, com quatro prêmios'],
    ['Green Book, Roma e Bohemian Rhapsody são os principais vencedores do Oscar 2019'],
    ['Oscar 2020: Confira lista completa de vencedores. Parasita e 1917 foram os grandes vencedores da noite'],
    ['Em boa fase, Oscar sonha em jogar a Copa do Mundo da Rússia'],
    ['Conheça os indicados ao Oscar 2020; Cerimônia de premiação acontece em fevereiro'],
    ['Oscar Schmidt receberá Troféu no Prêmio Brasil Olímpico 2019. Jogador de basquete com mais pontos em Jogos Olímpicos.'],
    ['Seleção brasileira vai observar de 35 a 40 jogadores para definir lista da Copa América'],
    ['Oscar 2020: saiba como é a escolha dos jurados e como eles votam'],
    ['Bem, Amigos! discute lista da Seleção, e Galvão dá recado a Tite: Cadê o Luan?'],
    ['IFAL-Maceió convoca aprovados em lista de espera do SISU para chamada oral'],
    ['Arrascaeta e Matías Viña são convocados pelo Uruguai para eliminatórias da Copa. Além deles, há outros destaques na lista.'],
    ['Oscar do Vinho: confira os rótulos de destaque da safra 2018'],
    ['Parasita é o vencedor da Palma de Ouro no Festival de Cannes'],
    ['Estatísticas. Brasileirão Série A: Os artilheiros e garçons da temporada 2020'],
    ['Setembro chegou! Confira o calendário da temporada 2020/2021 do futebol europeu']]
sw = ['a', 'o', 'e', 'é', 'de', 'do', 'da', 'no', 'na', 'são', 'dos', 'com','como','eles', 'em', 'os', 'ao', 'para', 'pelo']
st = [' ',',','.','!','?',':',';','/']

if __name__ == "__main__":
    search_engine = Indexer(documents=d, stopwords=sw, spliters=st, lang="portuguese")
    results = search_engine.search(query='Parasita oscar 2020', exact=True)
    
    for doc, score in results:
        print("Documento:", ' '.join(doc))
        print(f"Score de Similaridade TF-IDF: {score}\n")
