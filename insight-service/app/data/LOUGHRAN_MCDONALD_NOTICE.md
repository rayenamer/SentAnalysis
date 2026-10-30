# Loughran–McDonald financial sentiment dictionary

Authors: Tim Loughran and Bill McDonald, University of Notre Dame.
Source: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
Version: Master Dictionary 1993–2025, released March 2026.
Retrieved 30 October 2026 with the user's authorization for academic,
non-commercial research. Commercial use requires a license from the authors:
loughranmcdonald@gmail.com. This file does not grant additional rights.

Reference: Loughran, T. and McDonald, B. (2011), “When Is a Liability Not a
Liability? Textual Analysis, Dictionaries, and 10-Ks”, Journal of Finance,
66(1), 35–65.

The official CSV download hit a Google Drive quota. The alternative official
spreadsheet linked by the same authors was successfully exported as CSV.
The provenance JSON records the URL, version, source and derived SHA-256 hashes.
The compact CSV retains Word and all seven sentiment category flags, and only
rows having at least one active category. Positive years indicate membership;
negative years indicate removal and are excluded by the analyzer. No entries
were invented. The base non-sentiment dictionary and corpus statistics are
omitted. There are 3,859 retained words, including 347 positive and 2,345 negative.

The CPU analyzer lazily caches immutable word sets once per process. Unicode
NFC/case folding and letter tokenization precede exact English lookup.
Score = (positive occurrences - negative occurrences) / (positive occurrences
+ negative occurrences); zero recognized sentiment tokens gives zero.
The sign gives the label; confidence is null. Repeated words count repeatedly.
Other category counts are available in the internal score_text function but
do not affect polarity or change API response fields.

Select model="loughran_mcdonald" at /analyze or /analyze/batch. Results can use
the existing aggregation contracts. The Ingestion transfer model choices
remain VADER/FinBERT. Text bodies are scored; titles are not separately scored.
The existing API still rejects empty document text; score_text accepts empty
strings and returns zero. There is no stemming, translation, negation,
phrase reasoning or calibrated confidence. “Not excellent” therefore remains
positive under this word-count formula. English financial terms such as
liability/capital are not automatically negative.

No network access, transformers or additional packages are needed at runtime.
Do not redistribute the dictionary for commercial use without the authors'
permission. Consult the source terms before changing the project's use.
