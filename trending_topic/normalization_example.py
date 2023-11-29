import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize.casual import TweetTokenizer
from collections import defaultdict
import datetime

# from nltk.tokenize import word_tokenize
# from nltk.stem.porter import PorterStemmer

NGRAM_RANGE = (1, 4)
NUM_TOP_TERMS = 100
MAX_DOCS_PREPROCESS = 400
DATA_LANG = "portuguese"
PATH_DATASET = "./datasets/twitter_ptbr_train_datasets/Train50.csv"
PATH_STOPWORDS = "./stopwords-pt-br.txt"
CONTENT_COL = "tweet_text"
CSV_DELIMITER = ";"


def get_stopwords():
    df = pd.read_fwf(PATH_STOPWORDS, header=None)
    custom_stopwords = df.values.tolist()
    custom_stopwords = [s[0] for s in custom_stopwords]
    stop_words = set(stopwords.words(DATA_LANG))
    stop_words.update(custom_stopwords)
    return stop_words


def preprocess_documents(data: list):
    """
    Realiza o pré-processamento dos documentos removendo pontuações, stopwords e palavras com apenas uma letra.

    Args:
    - data (list): Lista de strings representando os documentos.

    Returns:
    - list: Lista de strings contendo os documentos pré-processados.
    """
    tokenizer = TweetTokenizer()

    processed_data = []
    for doc in data:
        # tokens = word_tokenize(doc)  # Tokenizar as palavras
        tokens = tokenizer.tokenize(doc.lower())  # Tokenizar as tweets

        # Reunir as palavras processadas em uma string
        processed_doc = " ".join(tokens)
        processed_data.append(processed_doc)

    return processed_data


def calculate_verboseness(processed_data: list):
    """
    Calcula a Verboseness dos documentos pré-processados.

    Args:
    - processed_data (list): Lista de strings representando os documentos pré-processados.

    Returns:
    - defaultdict: Dicionário contendo os valores de Verboseness por documento.
    """
    verbose_values = defaultdict(float)

    for i, doc in enumerate(processed_data):
        distinct_terms = len(set(doc.split()))
        document_length = len(doc.split())
        verbose_values[i] = (
            document_length / distinct_terms if distinct_terms != 0 else 0.0
        )

    return verbose_values


def calculate_burstiness(terms_matrix, terms):
    """
    Calcula a Burstiness dos termos nos documentos.

    Args:
    - terms_matrix: Matriz de ocorrências dos termos nos documentos.
    - terms: Array de termos dos documentos.

    Returns:
    - defaultdict: Dicionário contendo os valores de Burstiness por termo.
    """
    dense_matrix = terms_matrix.toarray()
    term_occurrences = np.sum(dense_matrix > 0, axis=0)
    term_occurrences_dict = dict(zip(terms, term_occurrences))

    term_counts = dict(zip(terms, terms_matrix.sum(axis=0).A1))

    bursty_values = defaultdict(float)
    for i, term in enumerate(terms):
        num_docs_with_term = term_occurrences_dict[term]
        total_term_occurrences = term_counts[term]

        bursty_values[term] = (
            total_term_occurrences / num_docs_with_term
            if num_docs_with_term != 0
            else 0.0
        )

        if "perfeitamente" == term or "indo" == term:
            print(term, total_term_occurrences, num_docs_with_term, bursty_values[term])

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
    count_vectorizer = CountVectorizer(ngram_range=NGRAM_RANGE)

    tfidf_matrix = tfidf_vectorizer.fit_transform(processed_data[:max_docs])
    terms = tfidf_vectorizer.get_feature_names_out()

    stop_words = get_stopwords()
    filtered_terms = [
        term for term in terms if term.split()[-1] not in stop_words and len(term) > 3
    ]

    count_matrix = count_vectorizer.fit_transform(processed_data[:max_docs])

    verbose_values = calculate_verboseness(processed_data[:max_docs])
    bursty_values = calculate_burstiness(count_matrix, filtered_terms)

    term_scores = {}
    for i, term in enumerate(filtered_terms):
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


def display_top_terms(sorted_terms: list, num_top_terms: int = None):
    """
    Exibe os termos mais relevantes.

    Args:
    - sorted_terms (list): Lista de tuplas com termos e seus scores TF-IDF ordenados por relevância.
    - num_top_terms (int): Número de termos a serem exibidos.
    """
    print(f"Top {num_top_terms if num_top_terms else 'N'} termos em trending topics:")
    for term, score in sorted_terms[:num_top_terms]:
        print(f"{term}: {score}")


def main():
    """
    Função principal para executar todo o fluxo de trabalho.
    """
    start_time = datetime.datetime.now()
    print("Iniciando o processo...")

    # Carregando uma coleção de documentos de exemplo
    data_path = PATH_DATASET
    data_col = CONTENT_COL
    data = pd.read_csv(data_path, delimiter=CSV_DELIMITER)[data_col].values

    start_preprocess_time = datetime.datetime.now()
    print("Iniciando o pré-processamento dos documentos...")
    processed_data = preprocess_documents(data)
    end_preprocess_time = datetime.datetime.now()
    preprocess_duration = end_preprocess_time - start_preprocess_time
    print(f"Pré-processamento concluído. Tempo decorrido: {preprocess_duration}")

    start_tfidf_time = datetime.datetime.now()
    print("Calculando os scores TF-IDF...")
    scored_terms = calculate_tfidf(
        processed_data=processed_data, max_docs=MAX_DOCS_PREPROCESS
    )
    end_tfidf_time = datetime.datetime.now()
    tfidf_duration = end_tfidf_time - start_tfidf_time
    print(f"Cálculo TF-IDF concluído. Tempo decorrido: {tfidf_duration}")

    start_display_time = datetime.datetime.now()
    print("Exibindo os termos mais relevantes...")
    display_top_terms(scored_terms)  # , num_top_terms=NUM_TOP_TERMS)
    end_display_time = datetime.datetime.now()
    display_duration = end_display_time - start_display_time
    print(f"Exibição concluída. Tempo decorrido: {display_duration}")

    end_time = datetime.datetime.now()
    total_duration = end_time - start_time
    print(f"Processo completo. Tempo total decorrido: {total_duration}")


if __name__ == "__main__":
    main()
