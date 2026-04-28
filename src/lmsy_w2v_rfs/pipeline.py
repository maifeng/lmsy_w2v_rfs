"""High-level orchestrator.

Wraps the five stages of the 2021 paper behind one class:

1. **parse**: optional CoreNLP multi-word-expression tagging.
2. **clean**: strip POS/NER tags, drop stopwords and punctuation.
3. **phrase**: gensim Phrases bigram (and optional trigram) pass.
4. **train**: Word2Vec on the phrase-expanded corpus.
5. **expand + score**: build the dictionary and score documents.

All intermediate artifacts land under ``work_dir`` so runs can resume
by skipping stages whose outputs already exist.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import tqdm
from gensim.models import Word2Vec

from .cleaner import clean_plain_line
from .config import Config
from .preprocessors import build_preprocessor
from .preprocessors.base import apply_mwe_list, load_mwe_list
from .dictionary import (
    deduplicate_keywords,
    expand_words_dimension_mean,
    rank_by_similarity,
    read_dict_csv,
    similarity_weights,
    write_dict_csv,
)
from .phrases import learn_phrases
from .scoring import (
    ScoringMethod,
    aggregate_to_firm_year,
    document_frequencies,
    iter_doc_level_corpus,
    score_documents,
    zca_whiten,
)
from .w2v import load_word2vec, train_word2vec

log = logging.getLogger(__name__)


class Pipeline:
    """End-to-end seed-expansion measurement pipeline.

    Typical usage::

        from lmsy_w2v_rfs import Pipeline, Config

        seeds = {
            "risk":   ["risk", "uncertainty", "volatility"],
            "growth": ["growth", "expansion", "scale"],
        }
        p = Pipeline(
            texts=my_texts,
            doc_ids=my_ids,
            work_dir="runs/demo",
            config=Config(seeds=seeds, preprocessor="none"),
        )
        p.run()
        p.show_dictionary(top_k=10)
        scores = p.score_df("TFIDF")

    Attributes:
        texts: Raw document strings.
        doc_ids: Matching IDs.
        work_dir: Directory for all intermediate and output files.
        config: Hyperparameters. Required.
    """

    def __init__(
        self,
        texts: Iterable[str] | None = None,
        doc_ids: Iterable[str] | None = None,
        *,
        work_dir: str | Path = "runs",
        config: Config | None = None,
    ) -> None:
        if config is None:
            raise ValueError(
                "Pipeline requires config=Config(seeds=...). "
                "Pass a Config built with your own seed dictionary, e.g. "
                'Config(seeds={"risk": ["risk", ...], "growth": [...]}).'
            )
        self.config = config
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._texts: list[str] | None = list(texts) if texts is not None else None
        self._doc_ids: list[str] | None = list(doc_ids) if doc_ids is not None else None
        if self._texts is not None and self._doc_ids is None:
            self._doc_ids = [str(i) for i in range(len(self._texts))]
        if self._texts is not None and self._doc_ids is not None:
            assert len(self._texts) == len(self._doc_ids), (
                "texts and doc_ids must have the same length"
            )
        self._w2v_model: Word2Vec | None = None
        self._culture_dict: dict[str, list[str]] | None = None
        self._scores: dict[str, pd.DataFrame] = {}

    # ---------- stage output paths -----------------------------------

    @property
    def parsed_sents_path(self) -> Path:
        """Sentence output from Phase 1a, one preprocessed sentence per line."""
        return self.work_dir / "parsed" / "sentences.txt"

    @property
    def parsed_ids_path(self) -> Path:
        """Parallel sentence IDs file; one ``docID_sentN`` ID per line."""
        return self.work_dir / "parsed" / "sentence_ids.txt"

    @property
    def cleaned_path(self) -> Path:
        """Cleaned sentences after stopword and punctuation removal."""
        return self.work_dir / "cleaned" / "sentences.txt"

    @property
    def phrase_corpus_path(self) -> Path:
        """Sentences after the last gensim ``Phrases`` pass."""
        return self.work_dir / "corpora" / f"pass{self.config.phrase_passes}.txt"

    @property
    def training_corpus_path(self) -> Path:
        """Final sentence file fed into Word2Vec."""
        if self.config.use_gensim_phrases:
            return self.phrase_corpus_path
        return self.cleaned_path

    @property
    def w2v_path(self) -> Path:
        """Saved gensim ``Word2Vec`` model."""
        return self.work_dir / "models" / "w2v.mod"

    @property
    def dict_path(self) -> Path:
        """Expanded per-dimension word list as CSV."""
        return self.work_dir / "outputs" / "expanded_dict.csv"

    def scores_path(self, method: ScoringMethod) -> Path:
        """Path to the document-level scores CSV for a given method.

        Args:
            method: One of ``TF``, ``TFIDF``, ``WFIDF``,
                ``TFIDF+SIMWEIGHT``, ``WFIDF+SIMWEIGHT``.

        Returns:
            Expected CSV path under ``work_dir/outputs/``.
        """
        return self.work_dir / "outputs" / f"scores_{method.replace('+', '_')}.csv"

    # ---------- stage 1: parse ---------------------------------------

    def parse(self, *, force: bool = False) -> None:
        """Run Phase 1: preprocess raw documents with the configured backend.

        Each sentence is written as one line of space-joined tokens. MWE
        groups are joined by ``_`` and named entities are replaced with
        ``[NER:TYPE]`` placeholders. Sentence IDs go to a parallel file.

        If ``config.mwe_list`` is set, a static-MWE post-pass runs after
        the main preprocessor to catch MWEs the parser missed.

        Args:
            force: Rerun even if output files exist.
        """
        if not force and self.parsed_sents_path.exists() and self.parsed_ids_path.exists():
            log.info("parse: reusing %s", self.parsed_sents_path)
            return
        if self._texts is None or self._doc_ids is None:
            raise RuntimeError("Pipeline was constructed without texts; nothing to parse.")
        self.parsed_sents_path.parent.mkdir(parents=True, exist_ok=True)

        preprocessor = build_preprocessor(self.config)
        mwe_extra = (
            load_mwe_list(self.config.mwe_list)
            if self.config.mwe_list and self.config.preprocessor != "static"
            else None
        )
        # Filter out empty inputs; keep matching doc_ids aligned.
        pairs = [(t, d) for t, d in zip(self._texts, self._doc_ids, strict=False) if t and t.strip()]
        texts_nonempty = [t for t, _ in pairs]
        ids_nonempty = [d for _, d in pairs]

        try:
            with self.parsed_sents_path.open("w", encoding="utf-8", newline="\n") as f_txt, \
                 self.parsed_ids_path.open("w", encoding="utf-8", newline="\n") as f_ids:
                i = 0
                # process_documents() is optional on the Protocol; fall back
                # to a serial loop over process() for simple backends.
                stream = (
                    preprocessor.process_documents(texts_nonempty)
                    if hasattr(preprocessor, "process_documents")
                    else (preprocessor.process(t) for t in texts_nonempty)
                )
                for did, sentences in zip(ids_nonempty, stream, strict=False):
                    if mwe_extra:
                        sentences = apply_mwe_list(sentences, mwe_extra)
                    for j, sent in enumerate(sentences):
                        if not sent:
                            continue
                        f_txt.write(" ".join(sent) + "\n")
                        f_ids.write(f"{did}_{j}\n")
                    i += 1
                    if i % 100 == 0:
                        log.info("parse: %d docs", i)
        finally:
            close = getattr(preprocessor, "close", None)
            if callable(close):
                close()

    # ---------- stage 2: clean ---------------------------------------

    def clean(self, *, force: bool = False) -> None:
        """Lowercase, drop stopwords and punctuation, strip CoreNLP tags.

        Args:
            force: Rerun even if output exists.
        """
        if not force and self.cleaned_path.exists():
            log.info("clean: reusing %s", self.cleaned_path)
            return
        self.cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        # The preprocessor already emits lemmatized, NER-masked, MWE-joined
        # tokens. The clean pass drops stopwords, punctuation, and 1-letter
        # tokens, preserving ``[NER:*]`` placeholders and underscore joins.
        clean_fn = clean_plain_line
        with self.parsed_sents_path.open("r", encoding="utf-8") as f_in, \
             self.cleaned_path.open("w", encoding="utf-8", newline="\n") as f_out:
            for line in tqdm.tqdm(f_in, desc="clean"):
                out = clean_fn(line, self.config.stopwords)
                f_out.write(out + "\n")

    # ---------- stage 3: phrases -------------------------------------

    def phrase(self, *, force: bool = False) -> Path:
        """Run gensim Phrases (if enabled) and return the final corpus path.

        Args:
            force: Rerun even if output exists.

        Returns:
            Path to the final phrase-expanded sentence file.
        """
        if not self.config.use_gensim_phrases:
            return self.cleaned_path
        if not force and self.phrase_corpus_path.exists():
            log.info("phrase: reusing %s", self.phrase_corpus_path)
            return self.phrase_corpus_path
        return learn_phrases(self.cleaned_path, self.work_dir, self.config)

    # ---------- stage 4: train ---------------------------------------

    def train(self, *, force: bool = False) -> Word2Vec:
        """Train (or load) the Word2Vec model.

        Args:
            force: Retrain even if ``w2v.mod`` exists.

        Returns:
            The loaded or trained model.
        """
        if not force and self.w2v_path.exists():
            log.info("train: loading %s", self.w2v_path)
            self._w2v_model = load_word2vec(self.w2v_path)
            return self._w2v_model
        self._w2v_model = train_word2vec(
            self.training_corpus_path, self.w2v_path, self.config
        )
        return self._w2v_model

    # ---------- stage 5: expand + score ------------------------------

    def expand_dictionary(self, *, force: bool = False) -> dict[str, list[str]]:
        """Expand seeds into a per-dimension culture dictionary.

        Args:
            force: Rebuild even if the dictionary CSV exists.

        Returns:
            Mapping of dimension name to rank-sorted word list.
        """
        if not force and self.dict_path.exists():
            log.info("expand_dictionary: reusing %s", self.dict_path)
            self._culture_dict, _ = read_dict_csv(self.dict_path)
            return self._culture_dict
        if self._w2v_model is None:
            self.train()
        assert self._w2v_model is not None
        expanded = expand_words_dimension_mean(
            self._w2v_model,
            self.config.seeds,
            n=self.config.n_words_dim,
            restrict_vocab=self.config.dict_restrict_vocab,
            min_similarity=self.config.min_similarity,
        )
        deduped = deduplicate_keywords(self._w2v_model, expanded, self.config.seeds)
        ranked = rank_by_similarity(deduped, self.config.seeds, self._w2v_model)
        write_dict_csv(ranked, self.dict_path)
        self._culture_dict = ranked
        return ranked

    def score(
        self,
        methods: Sequence[ScoringMethod] = ("TF", "TFIDF", "WFIDF"),
        *,
        force: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Score every document under each requested method.

        Args:
            methods: Any subset of TF, TFIDF, WFIDF, TFIDF+SIMWEIGHT,
                WFIDF+SIMWEIGHT.
            force: Recompute even if CSV outputs exist.

        Returns:
            Mapping of method name to scores DataFrame.
        """
        if self._culture_dict is None:
            self.expand_dictionary()
        assert self._culture_dict is not None

        expanded = {d: set(ws) for d, ws in self._culture_dict.items()}
        weights = similarity_weights(self._culture_dict)

        docs_for_df = list(
            iter_doc_level_corpus(self.training_corpus_path, self.parsed_ids_path)
        )
        df_dict, n_docs = document_frequencies(text for _, text in docs_for_df)

        for method in methods:
            out_path = self.scores_path(method)
            if not force and out_path.exists():
                log.info("score[%s]: reusing %s", method, out_path)
                self._scores[method] = pd.read_csv(out_path)
                continue
            df = score_documents(
                docs_for_df,
                expanded_words=expanded,
                method=method,
                df_dict=df_dict,
                n_docs=n_docs,
                word_weights=weights if "SIMWEIGHT" in method else None,
                normalize=self.config.tfidf_normalize,
            )
            if self.config.zca_whiten:
                df = zca_whiten(df, self.config.dims, epsilon=self.config.zca_epsilon)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            self._scores[method] = df
        return self._scores

    # ---------- one-call convenience ---------------------------------

    def run(
        self,
        *,
        methods: Sequence[ScoringMethod] = ("TF", "TFIDF", "WFIDF"),
        force: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Run every stage end to end.

        Args:
            methods: Scoring methods to compute.
            force: Rerun every stage regardless of existing outputs.

        Returns:
            Mapping of method name to scores DataFrame.
        """
        self._dump_config()
        self.parse(force=force)
        self.clean(force=force)
        self.phrase(force=force)
        self.train(force=force)
        self.expand_dictionary(force=force)
        return self.score(methods, force=force)

    # ---------- accessors --------------------------------------------

    def score_df(self, method: ScoringMethod = "TFIDF") -> pd.DataFrame:
        """Return the scores DataFrame for one method."""
        if method not in self._scores:
            path = self.scores_path(method)
            if path.exists():
                self._scores[method] = pd.read_csv(path)
            else:
                raise KeyError(f"No scores for {method}. Run .score().")
        return self._scores[method]

    def firm_year(
        self,
        id_to_firm: pd.DataFrame,
        method: ScoringMethod = "TFIDF",
    ) -> pd.DataFrame:
        """Aggregate scores to firm-year level.

        Args:
            id_to_firm: DataFrame with ``document_id``, ``firm_id``, ``time``.
            method: Which scores to aggregate.

        Returns:
            Firm-year DataFrame.
        """
        return aggregate_to_firm_year(
            self.score_df(method), id_to_firm, self.config.dims
        )

    @property
    def culture_dict(self) -> dict[str, list[str]]:
        """The expanded dictionary, rank-sorted.

        Alias for ``expanded_dict``; kept for compatibility with prose
        from the 2021 paper. The returned dict is the SAME object held
        on the pipeline; mutating it mutates pipeline state.
        """
        if self._culture_dict is None:
            self.expand_dictionary()
        assert self._culture_dict is not None
        return self._culture_dict

    @property
    def expanded_dict(self) -> dict[str, list[str]]:
        """The expanded dictionary, rank-sorted, theory-agnostic name."""
        return self.culture_dict

    @property
    def w2v(self) -> Word2Vec:
        """The trained Word2Vec model."""
        if self._w2v_model is None:
            self.train()
        assert self._w2v_model is not None
        return self._w2v_model

    # ---------- inspection + manual curation -------------------------

    def dictionary_preview(self, top_k: int = 10) -> pd.DataFrame:
        """Return a per-dimension preview of the expanded dictionary.

        Useful for notebook display: rendering the returned DataFrame
        shows seeds and the top-``top_k`` expanded words side by side.

        Args:
            top_k: How many expanded words to show per dimension.

        Returns:
            A DataFrame with one row per dimension and columns
            ``dimension``, ``seeds``, ``expanded_top_k``, ``n_expanded``.
        """
        d = self.culture_dict
        rows = []
        for dim in self.config.dims:
            words = d.get(dim, [])
            seed_words = list(self.config.seeds.get(dim, []))
            rows.append(
                {
                    "dimension": dim,
                    "seeds": ", ".join(seed_words),
                    "expanded_top_k": ", ".join(words[:top_k]),
                    "n_expanded": len(words),
                }
            )
        return pd.DataFrame(rows)

    def show_dictionary(self, top_k: int = 10) -> None:
        """Pretty-print the expanded dictionary, dimension by dimension.

        Replaces the boilerplate ``for dim in DIMS: print(...)`` loop.
        Prints seeds with an in-vocab marker and the top-``top_k``
        expanded words for each dimension.

        Args:
            top_k: How many expanded words to show per dimension.
        """
        d = self.culture_dict
        vocab = self._w2v_model.wv.key_to_index if self._w2v_model is not None else None
        for dim in self.config.dims:
            seed_words = list(self.config.seeds.get(dim, []))
            if vocab is not None:
                seed_strs = [s if s in vocab else f"{s} (oov)" for s in seed_words]
            else:
                seed_strs = seed_words
            words = d.get(dim, [])
            print(f"\n=== {dim} ({len(words)} words) ===")
            print(f"  seeds:    {', '.join(seed_strs)}")
            print(f"  expanded: {', '.join(words[:top_k])}")

    def edit_dictionary(
        self,
        remove: dict[str, Iterable[str]] | None = None,
        add: dict[str, Iterable[str]] | None = None,
    ) -> dict[str, list[str]]:
        """Manually curate the expanded dictionary.

        The 2021 paper allowed researchers to drop noisy expansion
        candidates (and occasionally append domain words the model
        missed) before scoring. This method does both atomically:
        in-memory dict and on-disk CSV are updated together.

        Use programmatically from a notebook, or pair with
        :meth:`reload_dictionary` to drive curation from a spreadsheet.

        Args:
            remove: Mapping of dimension name to words to drop. Words
                not present are silently ignored.
            add: Mapping of dimension name to words to append (at the
                end, after existing rank-sorted entries). Duplicates
                within a dimension are deduped.

        Returns:
            The updated dict.

        Raises:
            KeyError: If a referenced dimension is not in
                ``config.seeds``.
        """
        if self._culture_dict is None:
            self.expand_dictionary()
        assert self._culture_dict is not None

        known = set(self.config.dims)
        for source in (remove or {}, add or {}):
            for dim in source:
                if dim not in known:
                    raise KeyError(
                        f"Unknown dimension {dim!r}. Known: {sorted(known)}."
                    )

        for dim, words in (remove or {}).items():
            drop = set(words)
            self._culture_dict[dim] = [w for w in self._culture_dict[dim] if w not in drop]

        for dim, words in (add or {}).items():
            existing = list(self._culture_dict[dim])
            seen = set(existing)
            for w in words:
                if w not in seen:
                    existing.append(w)
                    seen.add(w)
            self._culture_dict[dim] = existing

        write_dict_csv(self._culture_dict, self.dict_path)
        # Drop any cached scores; they were computed against the old dict.
        self._scores.clear()
        return self._culture_dict

    def reload_dictionary(self) -> dict[str, list[str]]:
        """Reread the dictionary CSV from disk.

        Use this after editing ``pipeline.dict_path`` in a spreadsheet
        or text editor. Cached scores are dropped because they were
        computed against the previous dict.

        Returns:
            The reloaded dict.
        """
        if not self.dict_path.exists():
            raise FileNotFoundError(
                f"No dictionary CSV at {self.dict_path}. "
                "Run .expand_dictionary() first."
            )
        self._culture_dict, _ = read_dict_csv(self.dict_path)
        self._scores.clear()
        return self._culture_dict

    def _dump_config(self) -> None:
        try:
            obj: dict[str, Any] = asdict(self.config)
            obj["stopwords"] = sorted(self.config.stopwords)
            (self.work_dir / "config.json").write_text(
                json.dumps(obj, indent=2, default=list)
            )
        except Exception as e:  # pragma: no cover
            log.warning("Could not dump config: %s", e)

    # ---------- construction sugar -----------------------------------

    @classmethod
    def from_text_file(
        cls,
        text_path: str | Path,
        id_path: str | Path | None = None,
        *,
        work_dir: str | Path = "runs",
        config: Config | None = None,
    ) -> Pipeline:
        """Construct a pipeline from a one-document-per-line text file.

        Args:
            text_path: Input text file, one document per line.
            id_path: Optional matching IDs file.
            work_dir: Where to write artifacts.
            config: Pipeline config.

        Returns:
            A new ``Pipeline``.
        """
        texts = Path(text_path).read_text(encoding="utf-8", errors="ignore").splitlines()
        ids: list[str] | None = None
        if id_path is not None:
            ids = Path(id_path).read_text(encoding="utf-8").splitlines()
        return cls(texts=texts, doc_ids=ids, work_dir=work_dir, config=config)

    @classmethod
    def from_directory(
        cls,
        dir_path: str | Path,
        pattern: str = "*.txt",
        *,
        work_dir: str | Path = "runs",
        config: Config | None = None,
    ) -> Pipeline:
        """Construct a pipeline from a directory of one-file-per-document text.

        Common pattern: SEC filings where each 10-K is ``10k_AAPL_2024.txt``.
        Each file's contents become one document; each file's stem (filename
        without extension) becomes the document ID.

        Args:
            dir_path: Directory to scan.
            pattern: Glob pattern, e.g. ``"*.txt"`` or ``"**/*.md"``.
            work_dir: Where to write artifacts.
            config: Pipeline config.

        Returns:
            A new ``Pipeline``.
        """
        files = sorted(Path(dir_path).glob(pattern))
        texts = [p.read_text(encoding="utf-8", errors="ignore") for p in files]
        ids = [p.stem for p in files]
        return cls(texts=texts, doc_ids=ids, work_dir=work_dir, config=config)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        text_col: str = "text",
        id_col: str | None = "id",
        *,
        work_dir: str | Path = "runs",
        config: Config | None = None,
    ) -> Pipeline:
        """Construct a pipeline from a pandas DataFrame.

        Most workshop / teaching examples load a CSV with ``pd.read_csv``
        and then have a DataFrame with one row per document. This is the
        no-nonsense way in.

        Args:
            df: DataFrame with at least a text column.
            text_col: Name of the column with document text.
            id_col: Name of the column with document IDs. Pass ``None`` to
                use the DataFrame's row index as IDs.
            work_dir: Where to write artifacts.
            config: Pipeline config.

        Returns:
            A new ``Pipeline``.
        """
        texts = df[text_col].astype(str).tolist()
        if id_col is None:
            ids = [str(i) for i in df.index.tolist()]
        else:
            ids = df[id_col].astype(str).tolist()
        return cls(texts=texts, doc_ids=ids, work_dir=work_dir, config=config)

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        text_col: str = "text",
        id_col: str | None = "id",
        *,
        work_dir: str | Path = "runs",
        config: Config | None = None,
        **read_csv_kwargs: Any,
    ) -> Pipeline:
        """Construct a pipeline from a CSV file.

        Equivalent to ``from_dataframe(pd.read_csv(csv_path, **kwargs))``.
        Extra kwargs pass through to ``pandas.read_csv`` so you can specify
        delimiters, encodings, dtypes, etc.

        Args:
            csv_path: Path to the CSV file.
            text_col: Name of the column with document text.
            id_col: Name of the column with document IDs.
            work_dir: Where to write artifacts.
            config: Pipeline config.
            **read_csv_kwargs: Forwarded to ``pandas.read_csv``.

        Returns:
            A new ``Pipeline``.
        """
        df = pd.read_csv(csv_path, **read_csv_kwargs)
        return cls.from_dataframe(
            df, text_col=text_col, id_col=id_col, work_dir=work_dir, config=config
        )

    @classmethod
    def from_jsonl(
        cls,
        jsonl_path: str | Path,
        text_key: str = "text",
        id_key: str | None = "id",
        *,
        work_dir: str | Path = "runs",
        config: Config | None = None,
    ) -> Pipeline:
        """Construct a pipeline from a JSON Lines file.

        Each line of the file is a standalone JSON object. Useful for
        records exported from an API or a database in a streaming format.

        Args:
            jsonl_path: Path to ``.jsonl`` file.
            text_key: JSON key holding document text.
            id_key: JSON key holding document ID, or ``None`` to use the
                1-based line number.
            work_dir: Where to write artifacts.
            config: Pipeline config.

        Returns:
            A new ``Pipeline``.
        """
        import json

        texts: list[str] = []
        ids: list[str] = []
        with Path(jsonl_path).open("r", encoding="utf-8") as f:
            for i, raw in enumerate(f, 1):
                raw = raw.strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                texts.append(str(obj[text_key]))
                ids.append(str(obj[id_key]) if id_key else str(i))
        return cls(texts=texts, doc_ids=ids, work_dir=work_dir, config=config)
