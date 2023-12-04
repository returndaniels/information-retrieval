#!/usr/bin/python3
import pandas as pd
import numpy as np
import concurrent.futures

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from utils import log, log_step, get_stopwords, preprocess_documents, write_output
from dotenv import load_dotenv

import datetime
import os

load_dotenv()

NGRAM_RANGE = eval(os.getenv("NGRAM_RANGE"))
MAX_DOCS_PREPROCESS = eval(os.getenv("MAX_DOCS_PREPROCESS"))
DATA_LANG = os.getenv("DATA_LANG")
PATH_DATASET = os.getenv("PATH_DATASET")
PATH_STOPWORDS = os.getenv("PATH_STOPWORDS")
CONTENT_COL = os.getenv("CONTENT_COL")
CSV_DELIMITER = os.getenv("CSV_DELIMITER")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
OUTPUT_FILENAME = os.getenv("OUTPUT_FILENAME")

LOG_MSG_START = "Iniciando o processo..."
LOG_MSG_PREPROCESS = "pré-processamento dos documentos"
LOG_MSG_VERBOSENESS = "o cálculo a Verboseness para os termos"
LOG_MSG_BURSTINESS = "o cálculo a Burstiness para os documentos"
LOG_MSG_CALC_TFIDF = "o cálculo dos scores TF-IDF"
LOG_MSG_SCORE_START = "Calculando os scores normalizados..."
LOG_MSG_SCORE_COMPLETE = "Cálculo dos scores normalizados concluído"
LOG_MSG_WRITE_TERMS = "o salvamento dos termos mais relevantes em um arquivo CSV"
LOG_MSG_COMPLETE = "Processo completo. Tempo total decorrido:"

global_start_time = None


def calculate_verboseness(processed_data: pd.DataFrame):
    """
    Calcula a Verboseness dos documentos pré-processados.

    Args:
    - processed_data (pd.DataFrame): Lista de strings representando os documentos pré-processados.

    Returns:
    - ict: Dicionário contendo os valores de Verboseness por documento.
    """
    document_lengths = np.array([len(doc.split()) for doc in processed_data])
    distinct_terms = np.array([len(set(doc.split())) for doc in processed_data])

    verbose_values = None
    with np.errstate(divide="ignore", invalid="ignore"):
        verbose_values = np.where(
            distinct_terms != 0, document_lengths / distinct_terms, 0.0
        )

    verbose_values_dict = dict(zip(range(len(verbose_values)), verbose_values))

    return verbose_values_dict


def calculate_burstiness(terms_matrix, terms):
    """
    Calcula a Burstiness dos termos nos documentos.

    Args:
    - terms_matrix: Matriz de ocorrências dos termos nos documentos.
    - terms: Array de termos dos documentos.

    Returns:
    - dict: Dicionário contendo os valores de Burstiness por termo.
    """

    with np.errstate(divide="ignore", invalid="ignore"):
        term_counts = np.asarray(terms_matrix.sum(axis=0)).flatten()
        docs_with_terms = np.asarray(terms_matrix.astype(bool).sum(axis=0)).flatten()
        bursty_values = np.where(
            docs_with_terms != 0, term_counts / docs_with_terms, 0.0
        )
    return dict(zip(terms, bursty_values))


def calculate_score_for_term(
    i,
    term,
    term_index,
    tfidf_matrix,
    verbose_values,
    bursty_values,
):
    term_tf_idf = tfidf_matrix[:, term_index].sum()
    term_occurrences = tfidf_matrix[:, i]
    doc_indices_with_term = term_occurrences.nonzero()[0]
    mean_verbose_with_term = (
        np.mean([verbose_values[d] for d in doc_indices_with_term])
        if len(doc_indices_with_term) > 0
        else 0.0
    )

    adjusted_score = term_tf_idf / (mean_verbose_with_term * bursty_values[term])

    return {
        "adjusted_score": adjusted_score,
        "TF-IDF": term_tf_idf,
        "mean_verbose_with_term": mean_verbose_with_term,
        "bursty_values": bursty_values[term],
    }


def process_terms(
    start,
    end,
    filtered_terms,
    tfidf_vectorizer,
    tfidf_matrix,
    verbose_values,
    bursty_values,
):
    partial_scores = {}
    try:
        for i in range(start, end):
            term = filtered_terms[i]
            term_index = tfidf_vectorizer.vocabulary_.get(term)
            partial_scores[term] = calculate_score_for_term(
                i,
                term,
                term_index,
                tfidf_matrix,
                verbose_values,
                bursty_values,
            )
    except Exception as e:
        print(f"Erro durante o processamento: {e}")
    return partial_scores


def calculate_tfidf(processed_data: pd.DataFrame):
    """
    Calcula os scores TF-IDF normalizados por Verboseness e Burstiness dos termos nos documentos pré-processados.

    Args:
    - processed_data (pd.DataFrame): Lista de strings representando os documentos pré-processados.

    Returns:
    - dict: Dicionário contendo os termos e seus scores TF-IDF normalizados por Verboseness e Burstiness, ordenados por relevância.
    """
    tfidf_vectorizer = TfidfVectorizer(ngram_range=NGRAM_RANGE)
    count_vectorizer = CountVectorizer(ngram_range=NGRAM_RANGE)

    tfidf_matrix = tfidf_vectorizer.fit_transform(processed_data)
    terms = tfidf_vectorizer.get_feature_names_out()

    stop_words = get_stopwords(PATH_STOPWORDS, DATA_LANG)
    term_df = pd.DataFrame({"term": terms})
    term_df["last_word"] = term_df["term"].str.split().str[-1]
    filtered_terms = term_df[
        (term_df["last_word"].isin(stop_words) == False)
        & (term_df["term"].str.len() > 3)
    ]["term"].tolist()

    count_matrix = count_vectorizer.fit_transform(processed_data)

    verbose_values = log_step(
        global_start_time, LOG_MSG_VERBOSENESS, calculate_verboseness, processed_data
    )
    bursty_values = log_step(
        global_start_time,
        LOG_MSG_BURSTINESS,
        calculate_burstiness,
        count_matrix,
        filtered_terms,
    )

    num_terms = len(filtered_terms)
    chunk_size = 250
    start_indices = list(range(0, num_terms, chunk_size))
    end_indices = start_indices[1:] + [num_terms]

    term_scores = {}

    log(global_start_time, LOG_MSG_SCORE_START)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for start, end in zip(start_indices, end_indices):
            future = executor.submit(
                process_terms,
                start,
                end,
                filtered_terms,
                tfidf_vectorizer,
                tfidf_matrix,
                verbose_values,
                bursty_values,
            )
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            try:
                partial_scores = future.result()
                term_scores.update(partial_scores)
            except Exception as e:
                print(f"Erro ao obter resultado do futuro: {e}")

    log(global_start_time, LOG_MSG_SCORE_COMPLETE)

    sorted_normalized_tfidf = sorted(
        term_scores.items(), key=lambda x: x[1]["adjusted_score"], reverse=True
    )
    return {term: values for term, values in sorted_normalized_tfidf}


def main():
    """
    Função principal para executar todo o fluxo de trabalho.
    """
    log(global_start_time, LOG_MSG_START)

    # Carregando uma coleção de documentos de exemplo
    data_path = PATH_DATASET
    data_col = CONTENT_COL
    data = pd.read_csv(data_path, delimiter=CSV_DELIMITER, nrows=MAX_DOCS_PREPROCESS)[
        data_col
    ]

    processed_data = log_step(
        global_start_time, LOG_MSG_PREPROCESS, preprocess_documents, data
    )
    scored_terms = log_step(
        global_start_time, LOG_MSG_CALC_TFIDF, calculate_tfidf, processed_data
    )

    fieldnames = [
        "term",
        "adjusted_score",
        "TF-IDF",
        "mean_verbose_with_term",
        "bursty_values",
    ]

    log_step(
        global_start_time,
        LOG_MSG_WRITE_TERMS,
        write_output,
        scored_terms,
        OUTPUT_DIR,
        OUTPUT_FILENAME,
        fieldnames,
    )


if __name__ == "__main__":
    global_start_time = datetime.datetime.now()
    main()
