"""Command-line entry point for ``lmsy-w2v-rfs``.

Two subcommands:

- ``run``: end-to-end pipeline. Reads documents from disk (text, CSV, JSONL,
  or a directory of files), parses + cleans + phrases + trains + scores.
- ``download-corenlp``: one-time CoreNLP install into the local cache dir.

Typical invocations are in the how-to ``docs/how-to/run-from-cli.md``; a
short cheat sheet lives in the ``--help`` output of each subcommand.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Config, load_seeds
from .pipeline import Pipeline


_PREPROCESSOR_CHOICES = ("none", "static", "stanza", "corenlp", "spacy")
_INPUT_FORMAT_CHOICES = ("text", "csv", "jsonl", "directory")


def _add_run_args(p: argparse.ArgumentParser) -> None:
    # ----- input ---------------------------------------------------------
    g_in = p.add_argument_group("input")
    g_in.add_argument("--input", "-i", type=Path, required=True,
                      help="Input path. Interpreted by --input-format.")
    g_in.add_argument("--input-format", choices=_INPUT_FORMAT_CHOICES, default="text",
                      help="How to read --input: 'text' (default, one doc per line), "
                           "'csv' (pandas.read_csv, pass --text-col/--id-col), "
                           "'jsonl' (one JSON object per line, pass --text-key/--id-key), "
                           "'directory' (one file per doc, pass --glob-pattern).")
    g_in.add_argument("--ids", type=Path, default=None,
                      help="[text format] Optional matching IDs file, one ID per line.")
    g_in.add_argument("--text-col", default="text",
                      help="[csv format] Column holding document text.")
    g_in.add_argument("--id-col", default="id",
                      help="[csv format] Column holding document IDs. Set to '' to "
                           "use the DataFrame's row index.")
    g_in.add_argument("--text-key", default="text",
                      help="[jsonl format] JSON key holding document text.")
    g_in.add_argument("--id-key", default="id",
                      help="[jsonl format] JSON key holding document IDs. Set to '' "
                           "to use 1-based line numbers.")
    g_in.add_argument("--glob-pattern", default="*.txt",
                      help="[directory format] Glob pattern for matching files. "
                           "Document IDs come from the file stem.")

    # ----- output --------------------------------------------------------
    g_out = p.add_argument_group("output")
    g_out.add_argument("--out", "-o", type=Path, default=Path("runs/out"),
                       help="Output directory for all artifacts.")
    g_out.add_argument("--force", action="store_true",
                       help="Rerun every stage regardless of existing outputs.")

    # ----- seeds ---------------------------------------------------------
    g_seeds = p.add_argument_group("seeds")
    g_seeds.add_argument("--seeds", default=None,
                         help="Path to a seed dictionary file (.json or .txt). "
                              "Omit to use the 2021 paper's 5-dim default "
                              "(integrity, teamwork, innovation, respect, quality). "
                              "See how-to/use-your-own-seeds.md.")

    # ----- Phase 1 -------------------------------------------------------
    g_p1 = p.add_argument_group("phase 1 (preprocessor)")
    g_p1.add_argument("--preprocessor", choices=_PREPROCESSOR_CHOICES, default="corenlp",
                      help="Phase 1a backend. Default 'corenlp' (paper-faithful, "
                           "needs Java + lmsy-w2v-rfs download-corenlp, best "
                           "syntactic MWE coverage). Alternatives: 'spacy' "
                           "(fastest, no Java), 'stanza' (Python-native, slow), "
                           "'static' (curated-list only), 'none' (no parser).")
    g_p1.add_argument("--mwe-list", default="none",
                      help="Optional static MWE list applied after the main "
                           "preprocessor. 'none' (default) skips it. 'finance' uses "
                           "the packaged earnings-call example list. Otherwise pass "
                           "a path to your own newline-delimited list.")
    g_p1.add_argument("--spacy-model", default="en_core_web_sm",
                      help="spaCy model when --preprocessor=spacy. "
                           "Common: en_core_web_sm (default, fast) or "
                           "en_core_web_trf (slower, best NER).")
    g_p1.add_argument("--n-cores", type=int, default=4,
                      help="Parallel workers for CoreNLP (JVM threads) or "
                           "stanza / spaCy (Python processes). 4 is safe on an "
                           "8-core laptop; bump to 8 on workstations.")

    # ----- Phase 2 -------------------------------------------------------
    g_p2 = p.add_argument_group("phase 2 (gensim phrases)")
    g_p2.add_argument("--no-phrases", action="store_true",
                      help="Skip the gensim Phrases pass entirely.")
    g_p2.add_argument("--phrase-passes", type=int, default=2,
                      help="1 yields bigrams only, 2 yields bigrams then trigrams.")
    g_p2.add_argument("--phrase-min-count", type=int, default=10)
    g_p2.add_argument("--phrase-threshold", type=float, default=10.0)

    # ----- Word2Vec ------------------------------------------------------
    g_w = p.add_argument_group("word2vec")
    g_w.add_argument("--w2v-dim", type=int, default=300,
                     help="Vector dimension.")
    g_w.add_argument("--w2v-window", type=int, default=5,
                     help="Context window.")
    g_w.add_argument("--w2v-min-count", type=int, default=5,
                     help="Drop tokens that appear fewer than this many times.")
    g_w.add_argument("--w2v-epochs", type=int, default=20,
                     help="Training epochs.")

    # ----- Dictionary + scoring -----------------------------------------
    g_score = p.add_argument_group("dictionary and scoring")
    g_score.add_argument("--n-words-dim", type=int, default=500,
                         help="Top-k expanded words per dimension.")
    g_score.add_argument("--methods", nargs="+", default=["TF", "TFIDF", "WFIDF"],
                         help="Space-separated list of scoring methods. "
                              "Any subset of TF, TFIDF, WFIDF, TFIDF+SIMWEIGHT, "
                              "WFIDF+SIMWEIGHT.")
    g_score.add_argument("--zca-whiten", action="store_true",
                         help="Apply ZCA whitening to the dimension columns of "
                              "every scores DataFrame. Decorrelates while keeping "
                              "column names interpretable.")


def _make_config(args: argparse.Namespace) -> Config:
    mwe_list: str | None = args.mwe_list
    if mwe_list and mwe_list.lower() == "none":
        mwe_list = None
    seeds = load_seeds(args.seeds) if args.seeds else load_seeds(None)
    return Config(
        seeds=seeds,
        preprocessor=args.preprocessor,
        mwe_list=mwe_list,
        spacy_model=args.spacy_model,
        use_gensim_phrases=not args.no_phrases,
        phrase_passes=args.phrase_passes,
        phrase_min_count=args.phrase_min_count,
        phrase_threshold=args.phrase_threshold,
        n_cores=args.n_cores,
        w2v_dim=args.w2v_dim,
        w2v_window=args.w2v_window,
        w2v_min_count=args.w2v_min_count,
        w2v_epochs=args.w2v_epochs,
        n_words_dim=args.n_words_dim,
        zca_whiten=args.zca_whiten,
    )


def _build_pipeline(args: argparse.Namespace, cfg: Config) -> Pipeline:
    """Dispatch to the right ``Pipeline.from_*`` factory based on input format.

    Args:
        args: Parsed CLI namespace.
        cfg: Assembled ``Config``.

    Returns:
        A ready-to-run ``Pipeline``.
    """
    fmt = args.input_format
    if fmt == "text":
        return Pipeline.from_text_file(args.input, args.ids, work_dir=args.out, config=cfg)
    if fmt == "csv":
        id_col = args.id_col if args.id_col != "" else None
        return Pipeline.from_csv(args.input, text_col=args.text_col, id_col=id_col,
                                 work_dir=args.out, config=cfg)
    if fmt == "jsonl":
        id_key = args.id_key if args.id_key != "" else None
        return Pipeline.from_jsonl(args.input, text_key=args.text_key, id_key=id_key,
                                   work_dir=args.out, config=cfg)
    if fmt == "directory":
        return Pipeline.from_directory(args.input, pattern=args.glob_pattern,
                                       work_dir=args.out, config=cfg)
    raise ValueError(f"Unknown --input-format: {fmt!r}")


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Optional argv. Uses ``sys.argv`` when ``None``.

    Returns:
        Process exit code. 0 on success, 1 on usage errors.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(
        prog="lmsy-w2v-rfs",
        description=(
            "Word2Vec corporate-culture measurement. "
            "Run the full pipeline (run) or install CoreNLP (download-corenlp)."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser(
        "run",
        help="End-to-end pipeline run.",
        description=(
            "Read documents, preprocess, train Word2Vec, expand seed "
            "dictionary, score every document on every dimension. "
            "Outputs land under --out."
        ),
    )
    _add_run_args(p_run)

    sub.add_parser(
        "download-corenlp",
        help="Install Stanford CoreNLP into the cache directory (~1 GB).",
        description=(
            "Download and unpack Stanford CoreNLP into $LMSY_W2V_RFS_HOME or "
            "~/.cache/lmsy_w2v_rfs. Required once before the first "
            "--preprocessor corenlp run."
        ),
    )

    args = parser.parse_args(argv)

    if args.cmd == "download-corenlp":
        from . import download_corenlp

        path = download_corenlp()
        print(f"CoreNLP installed at: {path}")
        return 0

    if args.cmd == "run":
        cfg = _make_config(args)
        pipe = _build_pipeline(args, cfg)
        pipe.run(methods=tuple(args.methods), force=args.force)
        print(f"Done. Outputs under: {args.out}")
        print(f"  scores:     {args.out}/outputs/scores_*.csv")
        print(f"  dictionary: {args.out}/outputs/expanded_dict.csv")
        print(f"  w2v model:  {args.out}/models/w2v.mod")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
