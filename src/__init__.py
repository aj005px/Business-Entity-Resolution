"""Business entity resolution pipeline (Power Puff Girls).

Modules:
    config        - central configuration / paths
    normalize     - text cleaning, abbreviation expansion, tokenization
    data_io       - TSV/parquet helpers
    blocking      - candidate generation (country + token + TF-IDF NN)
    features      - per-pair similarity features
    label         - ground-truth labeling of candidate pairs
    f05_scorer    - exact competition metric (macro F0.5)
    train_model   - binary classifier training
    threshold_tuning - threshold sweep on validation F0.5
    recall_check  - blocking recall ceiling vs ground truth
    eda           - exploratory data analysis
    inference     - test-time matching results
    pipeline      - ordered end-to-end entry point
"""