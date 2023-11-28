import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict

NGRAM_RANGE = (1, 2)
NUM_TOP_TERMS = 100
MAX_DOCS_PREPROCESS = 250
DATA_LANG = "portuguese"
PATH_DATASET = "./datasets/twitter_ptbr_train_datasets/Train50.csv"
PATH_STOPWORDS = "./stopwords-pt-br.txt"
CONTENT_COL = "tweet_text"
CSV_DELIMITER = ";"


def preprocess_documents(data, lang="english", custom_stopwords=None):
    """
    Realiza o pré-processamento dos documentos removendo pontuações, stopwords e palavras com apenas uma letra.

    Args:
    - data (list): Lista de strings representando os documentos.
    - lang (str): Idioma dos documentos a serem tratados.
    - custom_stopwords (list): Lista de stopwords personalizadas a serem adicionadas.

    Returns:
    - list: Lista de strings contendo os documentos pré-processados.
    """
    processed_data = []
    stop_words = set(stopwords.words(lang))

    if custom_stopwords:
        stop_words.update(custom_stopwords)

    for doc in data:
        # Remover pontuações e caracteres especiais, converter para minúsculas
        doc = re.sub(r"[^a-zA-Z\s]", " ", doc.lower())
        tokens = word_tokenize(doc)  # Tokenizar as palavras

        # Remover stopwords e palavras com apenas uma letra
        filtered_tokens = [
            word for word in tokens if word not in stop_words and len(word) > 1
        ]

        # Reunir as palavras processadas em uma string
        processed_doc = " ".join(filtered_tokens)
        processed_data.append(processed_doc)

    return processed_data


def calculate_verboseness(processed_data: list, max_docs: int = None):
    """
    Calcula a Verboseness dos documentos pré-processados.

    Args:
    - processed_data (list): Lista de strings representando os documentos pré-processados.
    - max_docs (int): Número máximo de documentos a serem considerados.

    Returns:
    - defaultdict: Dicionário contendo os valores de Verboseness por documento.
    """
    verbose_values = defaultdict(float)

    for i, doc in enumerate(processed_data[:max_docs]):
        distinct_terms = len(set(doc.split()))
        document_length = len(doc)
        verbose_values[i] = (
            document_length / distinct_terms if distinct_terms != 0 else 0.0
        )

    return verbose_values


def calculate_burstiness(terms_matrix, terms):
    """
    Calcula a Burstiness dos termos nos documentos.

    Args:
    - terms_matrix: Matriz de termos nos documentos.
    - terms: Array de termos dos documentos.

    Returns:
    - defaultdict: Dicionário contendo os valores de Burstiness por termo.
    """
    bursty_values = defaultdict(float)

    for i, term in enumerate(terms):
        term_occurrences = terms_matrix[:, i]
        num_docs_with_term = (term_occurrences > 0).sum()
        total_term_occurrences = term_occurrences.sum()
        bursty_values[term] = (
            total_term_occurrences / num_docs_with_term
            if num_docs_with_term != 0
            else 0.0
        )

    return bursty_values


def calculate_tfidf(processed_data, max_docs=None):
    """
    Calcula os scores TF-IDF normalizados por Verboseness e Burstiness dos termos nos documentos pré-processados.

    Args:
    - processed_data (list): Lista de strings representando os documentos pré-processados.
    - max_docs (int): Número máximo de documentos a serem considerados.

    Returns:
    - list: Lista de tuplas contendo os termos e seus scores TF-IDF normalizados por Verboseness e Burstiness, ordenados por relevância.
    """
    tfidf_vectorizer = TfidfVectorizer(ngram_range=NGRAM_RANGE)
    tfidf_matrix = tfidf_vectorizer.fit_transform(processed_data[:max_docs])
    terms = tfidf_vectorizer.get_feature_names_out()

    verbose_values = calculate_verboseness(processed_data, max_docs)
    bursty_values = calculate_burstiness(tfidf_matrix, terms)

    term_scores = {}
    for i, term in enumerate(terms):
        term_occurrences = tfidf_matrix[:, i]

        # Encontrando os índices dos documentos que possuem o termo
        doc_indices_with_term = [
            idx for idx, occurrence in enumerate(term_occurrences) if occurrence > 0
        ]

        # Calculando a Verboseness média dos documentos que possuem o termo
        mean_verbose_with_term = (
            sum(verbose_values[d] for d in doc_indices_with_term)
            / len(doc_indices_with_term)
            if len(doc_indices_with_term) > 0
            else 0.0
        )

        term_scores[term] = (
            tfidf_matrix[:, i].mean() / (mean_verbose_with_term * bursty_values[term])
            if bursty_values[term] * mean_verbose_with_term != 0
            else 0.0
        )

    sorted_normalized_tfidf = sorted(
        term_scores.items(), key=lambda x: x[1], reverse=True
    )
    return sorted_normalized_tfidf


def display_top_terms(sorted_terms, num_top_terms=NUM_TOP_TERMS):
    """
    Exibe os termos mais relevantes.

    Args:
    - sorted_terms (list): Lista de tuplas com termos e seus scores TF-IDF ordenados por relevância.
    - num_top_terms (int): Número de termos a serem exibidos.
    """
    print(f"Top {num_top_terms} termos em trending topics:")
    for term, score in sorted_terms[:num_top_terms]:
        print(f"{term}: {score}")


def main():
    """
    Função principal para executar todo o fluxo de trabalho.
    """
    # Carregando uma coleção de documentos de exemplo
    data_path = PATH_DATASET
    data_col = CONTENT_COL
    data = pd.read_csv(data_path, delimiter=CSV_DELIMITER)[data_col].values

    # Pré-processamento dos documentos
    df = pd.read_fwf(PATH_STOPWORDS, header=None)
    custom_stopwords = df.values.tolist()
    custom_stopwords = [s[0] for s in custom_stopwords]
    processed_data = preprocess_documents(
        data, lang=DATA_LANG, custom_stopwords=custom_stopwords
    )

    # Calculando os scores TF-IDF
    scored_terms = calculate_tfidf(
        processed_data=processed_data, max_docs=MAX_DOCS_PREPROCESS
    )

    # Exibindo os termos mais relevantes
    display_top_terms(scored_terms, num_top_terms=NUM_TOP_TERMS)


if __name__ == "__main__":
    main()
