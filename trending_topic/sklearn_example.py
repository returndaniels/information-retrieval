from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def preprocess_documents(data: list, lang: str = "english"):
    """
    Pré-processa os documentos removendo pontuações, stopwords e palavras com apenas uma letra.

    Args:
    - data (list): Lista de strings representando os documentos.
    - lang (str): String com o idioma dos docuemntos a serem tratados.

    Returns:
    - list: Lista de strings contendo documentos pré-processados.
    """
    processed_data = []
    stop_words = set(stopwords.words(lang))

    for doc in data:
        # Remover pontuações e caracteres especiais, converter para minúsculas
        doc = re.sub(r"[^a-zA-Z\s]", " ", doc.lower())

        # Tokenizar as palavras
        tokens = word_tokenize(doc)

        # Remover stopwords e palavras com apenas uma letra
        filtered_tokens = [
            word for word in tokens if word not in stop_words and len(word) > 1
        ]

        # Reunir as palavras processadas em uma string
        processed_doc = " ".join(filtered_tokens)
        processed_data.append(processed_doc)

    return processed_data


def calculate_tfidf(processed_data: list, max_terms: int = None):
    """
    Calcula os scores TF-IDF dos termos nos documentos pré-processados.

    Args:
    - processed_data (list): Lista de strings representando os documentos pré-processados.
    - max_terms (int): Número máximo de termos a serem considerados.

    Returns:
    - list: Lista de tuplas contendo os termos e seus scores TF-IDF ordenados por relevância.
    """
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(processed_data)

    # Obtendo os termos (palavras) e seus scores TF-IDF
    terms = tfidf_vectorizer.get_feature_names_out()
    if max_terms:
        terms = terms[:max_terms]

    # Criando um dicionário para armazenar os termos e seus scores TF-IDF
    term_scores = {}
    for i, term in enumerate(terms):
        term_scores[term] = tfidf_matrix[:, i].mean()

    # Classificando os termos por seus scores TF-IDF
    sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_terms


def display_top_terms(sorted_terms: list, num_top_terms: int = 10):
    """
    Exibe os termos mais relevantes.

    Args:
    - sorted_terms (list): Lista de tuplas com termos e seus scores TF-IDF ordenados por relevância.
    - num_top_terms (int): Número de termos a serem exibidos.
    """
    print(f"Top {num_top_terms} termos em trending topics:")
    for term, score in sorted_terms[:num_top_terms]:
        print(f"{term}: {score}")


def sort_terms_by_tfidf(sorted_terms: list):
    """
    Ordena os termos por scores TF-IDF do mais relevante ao menos relevante.

    Args:
    - sorted_terms (list): Lista de tuplas com termos e seus scores TF-IDF ordenados por relevância.

    Returns:
    - list: Lista de tuplas com os termos ordenados por scores TF-IDF.
    """
    return sorted(sorted_terms, key=lambda x: x[1], reverse=True)


def main():
    # Carregando uma coleção de documentos de exemplo
    data = fetch_20newsgroups(subset="train").data

    # Pré-processamento dos documentos
    processed_data = preprocess_documents(data)

    # Calculando os scores TF-IDF
    scored_terms = calculate_tfidf(processed_data=processed_data)

    # Ordenar os termos por relevância de TF-IDF
    sorted_terms = sort_terms_by_tfidf(scored_terms)

    # Exibindo os termos mais relevantes
    display_top_terms(sorted_terms=sorted_terms, num_top_terms=100)


if __name__ == "__main__":
    main()
