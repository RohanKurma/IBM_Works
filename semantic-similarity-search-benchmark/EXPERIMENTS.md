# Experiment Log

Use this file to record experiments you actually run.

## Goal

Study how semantic-search behavior changes when varying:

- embedding model
- similarity/distance metric
- query wording
- document ambiguity
- corpus size
- top-k value

## Suggested model comparison

- `sentence-transformers/paraphrase-MiniLM-L6-v2`
- `sentence-transformers/all-MiniLM-L6-v2`
- `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`

## Suggested metric comparison

- cosine similarity
- dot-product similarity
- Euclidean distance

## Experiment table

| Run | Embedding model | Metric | Query | Expected Top-1 | Actual Top-1 | Hit@1 | Latency | Notes |
|---|---|---|---|---|---|---:|---:|---|
| 1 | | | | | | | | |
| 2 | | | | | | | | |
| 3 | | | | | | | | |

## Questions worth investigating

1. Does cosine similarity rank semantically related sentences more reliably than raw dot product?
2. How much do vector magnitudes affect dot-product rankings?
3. Do normalized embeddings make dot product and cosine similarity equivalent?
4. Which embedding model best separates different meanings of the same word?
5. How does query rephrasing affect the top-ranked result?
6. How does search quality change as the document collection becomes larger?
