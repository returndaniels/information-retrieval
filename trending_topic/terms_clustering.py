from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Leitura do arquivo e armazenamento dos termos e suas pontuações TF-IDF
file_content = []

with open("out", "r", encoding="utf-8") as file:
    for line in file:
        term, tfidf_score = line.split(":")
        file_content.append(term.strip())

# Criando um vetor TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(file_content)

# Calculando a similaridade de cosseno entre os termos
similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Definindo um limite de similaridade para o agrupamento
threshold = 0.65

# Agrupamento dos termos com alta similaridade
similar_items = defaultdict(list)

for i in range(len(file_content)):
    for j in range(i + 1, len(file_content)):
        if similarities[i][j] >= threshold:
            similar_items[file_content[i]].append(file_content[j])
            similar_items[file_content[j]].append(file_content[i])

# Removendo duplicatas nos grupos
for key, value in similar_items.items():
    similar_items[key] = list(set(value))

# Exibição dos grupos de termos com alta similaridade
for key, value in similar_items.items():
    if value:
        print(f"Termo: {key}")
        print(f"Similar Terms: {value}\n")
