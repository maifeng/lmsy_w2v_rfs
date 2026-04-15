"""Document and firm-year scoring.

Three flavors: TF (raw counts), TFIDF (log ``N/df``), and WFIDF
(``log(1+tf) * log(N/df)``). Each can be combined with a per-word
similarity weight to produce TFIDF+SIMWEIGHT or WFIDF+SIMWEIGHT.

Streaming-friendly: document frequencies and doc-level corpora are
built with a single pass over the sentence file, so the full corpus
never needs to sit in RAM (the original 2021 implementation pickled
the whole corpus; this one does not).
"""

from __future__ import annotations

import math
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import tqdm
from sklearn import preprocessing

ScoringMethod = Literal["TF", "TFIDF", "WFIDF", "TFIDF+SIMWEIGHT", "WFIDF+SIMWEIGHT"]


def iter_doc_level_corpus(
    sent_corpus_path: Path | str,
    sent_id_path: Path | str,
) -> Iterator[tuple[str, str]]:
    """Yield ``(doc_id, document_text)`` by folding sentences on their doc prefix.

    The CoreNLP pass produces sentence IDs shaped ``docID_sentenceN``.
    This function groups consecutive sentence lines with matching doc
    prefixes and yields one document at a time.

    Args:
        sent_corpus_path: File of cleaned sentences, one per line.
        sent_id_path: Matching sentence IDs.

    Yields:
        ``(doc_id, concatenated_document_text)`` pairs.
    """
    sent_corpus_path = Path(sent_corpus_path)
    sent_id_path = Path(sent_id_path)

    current_id: str | None = None
    buffer: list[str] = []
    with sent_corpus_path.open("r", encoding="utf-8") as f_txt, sent_id_path.open(
        "r", encoding="utf-8"
    ) as f_ids:
        for txt, sid in zip(f_txt, f_ids, strict=False):
            doc_id = sid.strip().split("_")[0]
            if current_id is not None and doc_id != current_id:
                yield current_id, " ".join(buffer)
                buffer = []
            current_id = doc_id
            buffer.append(txt.strip())
    if current_id is not None:
        yield current_id, " ".join(buffer)


def document_frequencies(
    documents: Iterable[str],
    show_progress: bool = True,
) -> tuple[dict[str, int], int]:
    """Compute document frequency for every token.

    Args:
        documents: Iterable of whitespace-tokenized documents.
        show_progress: Print a tqdm bar.

    Returns:
        ``(df_dict, n_documents)``.
    """
    df: dict[str, int] = defaultdict(int)
    n = 0
    it = tqdm.tqdm(documents, disable=not show_progress, desc="df")
    for doc in it:
        n += 1
        for w in set(doc.split()):
            df[w] += 1
    return dict(df), n


def score_document(
    document: str,
    expanded_words: dict[str, list[str] | set[str]],
    method: ScoringMethod = "TF",
    df_dict: dict[str, int] | None = None,
    n_docs: int | None = None,
    word_weights: dict[str, float] | None = None,
) -> tuple[list[float], int]:
    """Score one document across all dimensions.

    Args:
        document: Whitespace-tokenized text.
        expanded_words: Expanded dictionary per dimension.
        method: One of TF, TFIDF, WFIDF, TFIDF+SIMWEIGHT, WFIDF+SIMWEIGHT.
        df_dict: Document frequencies. Required for non-TF methods.
        n_docs: Total document count. Required for non-TF methods.
        word_weights: Per-word weights. Required for SIMWEIGHT methods.

    Returns:
        ``(scores_sorted_by_dim, document_length)``.

    Raises:
        ValueError: On inconsistent arguments.
    """
    tokens = document.split()
    doc_len = len(tokens)
    counts = Counter(tokens)
    scores: OrderedDict[str, float] = OrderedDict((d, 0.0) for d in sorted(expanded_words))

    use_idf = method != "TF"
    use_sim = method.endswith("+SIMWEIGHT")

    if use_idf and (df_dict is None or n_docs is None):
        raise ValueError("df_dict and n_docs are required for non-TF methods")
    if use_sim and word_weights is None:
        raise ValueError("word_weights is required for SIMWEIGHT methods")

    for w, tf in counts.items():
        for dim, words in expanded_words.items():
            if w not in words:
                continue
            idf = math.log(n_docs / df_dict[w]) if use_idf else 1.0  # type: ignore[index]
            if method == "TF":
                weight = tf
            elif method.startswith("WFIDF"):
                weight = (1 + math.log(tf)) * idf
            else:  # TFIDF and TFIDF+SIMWEIGHT
                weight = tf * idf
            if use_sim:
                weight *= word_weights[w]  # type: ignore[index]
            scores[dim] += weight
    return list(scores.values()), doc_len


def score_documents(
    documents: Iterable[tuple[str, str]],
    expanded_words: dict[str, list[str] | set[str]],
    method: ScoringMethod = "TFIDF",
    df_dict: dict[str, int] | None = None,
    n_docs: int | None = None,
    word_weights: dict[str, float] | None = None,
    normalize: bool = False,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Score an iterable of documents and return a DataFrame.

    Args:
        documents: Iterable of ``(doc_id, text)`` pairs.
        expanded_words: Expanded dictionary.
        method: Scoring method.
        df_dict: Document frequencies (non-TF methods).
        n_docs: Total document count (non-TF methods).
        word_weights: Per-word weights (SIMWEIGHT methods).
        normalize: L2-normalize score vector per document.
        show_progress: Print a tqdm bar.

    Returns:
        DataFrame with ``Doc_ID``, the sorted dimensions, and
        ``document_length``.
    """
    dims = sorted(expanded_words)
    rows: list[list[float | int]] = []
    ids: list[str] = []
    docs = tqdm.tqdm(documents, disable=not show_progress, desc=method)
    for doc_id, text in docs:
        scores, length = score_document(
            text,
            expanded_words,
            method=method,
            df_dict=df_dict,
            n_docs=n_docs,
            word_weights=word_weights,
        )
        rows.append([*scores, length])
        ids.append(doc_id)
    if not rows:
        return pd.DataFrame(columns=["Doc_ID", *dims, "document_length"])
    arr = np.asarray(rows, dtype=float)
    if normalize:
        arr[:, : len(dims)] = preprocessing.normalize(arr[:, : len(dims)])
    df = pd.DataFrame(arr, columns=[*dims, "document_length"])
    df.insert(0, "Doc_ID", ids)
    return df


def zca_whiten(
    scores: pd.DataFrame,
    dims: list[str],
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    """Apply ZCA whitening to the dimension columns of a scores DataFrame.

    ZCA (zero-phase component analysis) whitening is a linear transform that
    decorrelates the columns (makes the covariance the identity) while
    staying as close as possible to the original axes. Unlike PCA whitening,
    it does not rotate the data into a new basis, so after whitening the
    column named ``integrity`` still measures something close to integrity
    (not "principal component 1"). This matters when downstream analysis
    interprets each dimension by name.

    This is a post-scoring transform. Input columns retain their names;
    output values are the whitened coordinates. Non-dimension columns
    (``Doc_ID``, ``document_length``) pass through unchanged.

    The whitening transform is fit on ``scores[dims]`` itself. If you plan
    to score new documents later and want them on the same whitened scale,
    compute and persist the transform separately (see the "Notes" section
    of the docs page on whitening).

    Similar in spirit to the post-processing step in the Marketing Measures
    package: https://github.com/Marketing-Measures/marketing-measures.

    Args:
        scores: DataFrame from :func:`score_documents`, with one column per
            dimension plus ``Doc_ID`` and ``document_length``.
        dims: List of dimension column names to whiten.
        epsilon: Eigenvalue floor for numerical stability. Raise if the
            covariance is near-singular (small corpora, highly correlated
            dimensions).

    Returns:
        A new DataFrame with the same shape. Dimension columns are
        whitened; every other column is copied through.
    """
    X = scores[dims].to_numpy(dtype=float)
    if X.shape[0] < 2:
        # Cannot estimate covariance with 0 or 1 rows; return input unchanged.
        return scores.copy()
    mu = X.mean(axis=0)
    Xc = X - mu
    n = X.shape[0]
    sigma = (Xc.T @ Xc) / max(n - 1, 1)
    # Use eigh for symmetric matrices (faster and more stable than eig).
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.clip(eigvals, 0.0, None)
    scaling = np.diag(1.0 / np.sqrt(eigvals + epsilon))
    W = eigvecs @ scaling @ eigvecs.T
    X_white = Xc @ W
    out = scores.copy()
    out[dims] = X_white
    return out


def aggregate_to_firm_year(
    scores: pd.DataFrame,
    id_to_firm: pd.DataFrame,
    dims: list[str],
    doc_id_col: str = "Doc_ID",
    id_col: str = "document_id",
    firm_col: str = "firm_id",
    time_col: str = "time",
) -> pd.DataFrame:
    """Aggregate document-level scores to firm-year means.

    Each dimension is first divided by document length and multiplied
    by 100 to put units in "per 100 tokens."

    Args:
        scores: DataFrame from ``score_documents``.
        id_to_firm: DataFrame with the document-id to firm-year mapping.
        dims: Dimension column names to normalize.
        doc_id_col: Document-ID column in ``scores``.
        id_col: Document-ID column in ``id_to_firm``.
        firm_col: Firm-ID column in ``id_to_firm``.
        time_col: Time column in ``id_to_firm``.

    Returns:
        Firm-year DataFrame sorted by ``firm_id, time``.
    """
    merged = scores.merge(id_to_firm, how="left", left_on=[doc_id_col], right_on=id_col)
    merged = merged.drop(columns=[doc_id_col, id_col], errors="ignore")
    for dim in dims:
        merged[dim] = 100.0 * merged[dim] / merged["document_length"]
    agg = merged.groupby([firm_col, time_col], as_index=False)[dims].mean()
    return agg.sort_values([firm_col, time_col]).reset_index(drop=True)
