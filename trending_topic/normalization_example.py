import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def preprocess_documents(
    data: list, lang: str = "english", custom_stopwords: list = None
):
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

    if custom_stopwords:
        stop_words.update(custom_stopwords)

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
    Calcula os scores TF-IDF normalizados por Verboseness e Burstiness dos termos nos documentos pré-processados.

    Args:
    - processed_data (list): Lista de strings representando os documentos pré-processados.
    - max_terms (int): Número máximo de termos a serem considerados.

    Returns:
    - list: Lista de tuplas contendo os termos e seus scores TF-IDF normalizados por Verboseness e Burstiness,
            ordenados por relevância.
    """
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(processed_data)

    # Obtendo os termos (palavras) e seus scores TF-IDF
    terms = tfidf_vectorizer.get_feature_names_out()
    if max_terms:
        terms = terms[:max_terms]

    # Criando um dicionário para armazenar os termos e seus scores TF-IDF
    verbose_values = defaultdict(float)
    bursty_values = defaultdict(float)

    # Calculando a Verboseness por documento
    for i in range(len(processed_data)):
        # Selecionando apenas as ocorrências do documento i
        doc_occurrences = tfidf_matrix[i, :]

        # Calculando o comprimento do documento
        document_length = doc_occurrences.sum()

        # Calculando o número de termos distintos no documento
        distinct_terms = (doc_occurrences > 0).sum()

        # Calculando a Verboseness para o documento i
        verbose_values[i] = (
            document_length / distinct_terms if distinct_terms != 0 else 0.0
        )

    # Calculando a Burstiness por termo
    for i, term in enumerate(terms):
        term_occurrences = tfidf_matrix[:, i]

        # Calculando o número de documentos onde o termo ocorre
        num_docs_with_term = (term_occurrences > 0).sum()

        # Calculando o total de ocorrências do termo
        total_term_occurrences = term_occurrences.sum()

        # Calculando a Burstiness para o termo
        bursty_values[term] = (
            total_term_occurrences / num_docs_with_term
            if num_docs_with_term != 0
            else 0.0
        )

    # Normalizando os scores TF-IDF com base em Verboseness e Burstiness
    term_scores = {}
    for i, term in enumerate(terms):
        # Calculando o TF-IDF normalizado
        # term_scores[term] = tfidf_matrix[:, i].mean()
        term_scores[term] = (
            tfidf_matrix[:, i].mean() / (verbose_values[i] * bursty_values[term])
            if verbose_values[i] != 0
            else 0.0
        )

    # Ordenando os termos por TF-IDF normalizado
    sorted_normalized_tfidf = sorted(
        term_scores.items(), key=lambda x: x[1], reverse=True
    )

    return sorted_normalized_tfidf


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


def get_data_from_csv(data_path: str, data_col: str):
    dataset = pd.read_csv(data_path, delimiter=";")
    data = dataset.loc[:, data_col].values
    return data


def main():
    # Carregando uma coleção de documentos de exemplo
    data_path = "./datasets/twitter_ptbr_train_datasets/Train50.csv"
    data_col = "tweet_text"
    data = get_data_from_csv(data_path=data_path, data_col=data_col)

    # Pré-processamento dos documentos
    df = pd.read_fwf("./stopwords-pt-br.txt", header=None)
    custom_stopwords = df.values.tolist()
    custom_stopwords = [s[0] for s in custom_stopwords]

    processed_data = preprocess_documents(
        data, lang="portuguese", custom_stopwords=custom_stopwords
    )

    # Calculando os scores TF-IDF
    scored_terms = calculate_tfidf(processed_data=processed_data)  # , max_terms=10000)

    # Ordenar os termos por relevância de TF-IDF
    sorted_terms = sort_terms_by_tfidf(scored_terms)

    # Exibindo os termos mais relevantes
    display_top_terms(sorted_terms=sorted_terms, num_top_terms=100)


if __name__ == "__main__":
    main()
