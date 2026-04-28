# Seeds and vocabulary utilities

This page covers the functions and constants used to load seed dictionaries,
access built-in example seeds, manage the SRAF stopword list, and install the
optional CoreNLP dependency. These utilities are all importable directly from
`lmsy_w2v_rfs`.

## load_seeds

Read a seed dictionary from a plain Python dict, a `.json` file, or a
`.txt` file with one dimension-name header per section.

::: lmsy_w2v_rfs.load_seeds

## load_example_seeds

Retrieve a named seed dictionary that is bundled with the package. Currently
ships one example: `"culture_2021"` (the 5-dimension, 47-word dictionary
from Li, Mai, Shen, Yan 2021).

::: lmsy_w2v_rfs.load_example_seeds

## STOPWORDS_SRAF

A set of 121 generic stopwords drawn from the Loughran-McDonald
Software-Readable Accounting Forms (SRAF) list. Passed to `Config` as
`stopwords=STOPWORDS_SRAF` (the default).

::: lmsy_w2v_rfs.STOPWORDS_SRAF

## download_corenlp

One-call helper that installs Stanford CoreNLP into the local cache
directory. Requires the `[corenlp]` optional extra (`pip install
"lmsy_w2v_rfs[corenlp]"`).

::: lmsy_w2v_rfs.download_corenlp
