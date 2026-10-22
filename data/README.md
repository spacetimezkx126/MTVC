# Dataset placement

Place the datasets under `CSMD50/`, `CSMD300/` and `massive_data/`. Dataset contents are distributed separately from the source archive and excluded from Git.

Required inputs for the recorded recipes:

- All datasets: `price/preprocessed/`, `news/`, `comp_industry/comp_indus.csv`, and `.mtvc_file_order.json`.
- CSMD300 MTVC, PEN and StockNet inputs use `news/`; extracted-news and pattern annotation folders are not required.
- Massive additionally requires `code_name.json`.

The recorded MTVC CSMD recipes select raw news from `news/`, not `news_txt/`. Price-only baselines read price files directly. Precomputed news-vector or legacy news-bundle caches are not required for the current entry points. Vocabularies under `../dict/` retain their original token-to-ID mappings.

Keep `.mtvc_file_order.json` with each dataset. It preserves the original enumeration order of retained input files and their content checksums. Verify a dataset with `python scripts/verify_data.py data/CSMD50`. Do not rebuild a vocabulary, reorder inputs or substitute a different data release when checking a checkpoint.
