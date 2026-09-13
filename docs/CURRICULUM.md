# AI/ML curriculum using Wikipedia data

AtlasMind turns the original preparation list into one cohesive knowledge-system project. Wikipedia articles replace OHLCV data throughout the learning path.

## Python fundamentals

Study typed functions, dataclasses, classes, exceptions, context managers, generators, modules, packages, environment configuration, JSON Lines file handling, and command-line parsing across `src/atlasmind`.

## Python for data and mathematics

Use the collected JSON Lines corpus to practice lists, dictionaries, NumPy arrays, Pandas DataFrames, filtering, grouping, joins, missing values, duplicates, token counts, document-length distributions, mean, median, variance, standard deviation, probability, conditional probability, Bayes' theorem, correlation, covariance, vectors, matrices, dot products, derivatives, and gradient descent.

## Preprocessing

The implemented pipeline performs HTML article extraction, canonical URL cleanup, stable-hash deduplication, labeling, word-window chunking, TF-IDF vectorization, and train/test splitting. Extend it with language detection, Unicode normalization, duplicate similarity, class balancing, feature selection, one-hot metadata, normalization, standardization, and SMOTE applied only inside training folds.

## Classical machine learning

The real classifier uses Logistic Regression. Compare it with K-Nearest Neighbors, Naive Bayes, Decision Tree, Random Forest, Support Vector Machine, Gradient Boosting, XGBoost, and LightGBM. Compare accuracy, precision, recall, F1, confusion matrix, and ROC-AUC using held-out articles.

Add regression exercises by predicting article quality, reading time, or section count with Linear, Multiple Linear, Polynomial, Ridge, Lasso, and ElasticNet regression. Evaluate with MAE, MSE, RMSE, and R-squared.

Use cross-validation, grid search, randomized search, regularization, and learning curves to study bias, variance, overfitting, and underfitting. Keep near-duplicate articles in the same fold to prevent leakage.

## Unsupervised learning

Cluster article vectors with K-Means, hierarchical clustering, and DBSCAN. Use PCA to visualize topics and Isolation Forest to detect unusual or malformed documents.

## Scikit-learn

`training.py` demonstrates Pipeline, TF-IDF preprocessing, model training, prediction, metrics, a stratified split, and Joblib persistence. Add `ColumnTransformer` when combining article text with numeric and categorical metadata.

## Deep learning

Turn chunks into PyTorch datasets and compare ANN, CNN text classification, RNN, LSTM, GRU, attention, and Transformer models. Study neurons, layers, weights, bias, activation functions, forward propagation, backpropagation, loss functions, SGD, Adam, batch size, epochs, dropout, and batch normalization.

## NLP

Wikipedia naturally supports tokenization, stop words, stemming, lemmatization,
Bag of Words, TF-IDF, Word2Vec, GloVe, embeddings, text classification, sentiment
analysis, named entity recognition, and Transformers. The baseline uses TF-IDF,
while AtlasMind's multimodal path uses the configured open-source vision-language
model with locally retrieved evidence.

## Generative AI and LLMs

The knowledge system implements chunking, TF-IDF retrieval, local hashing
embeddings, PostgreSQL `pgvector`, HNSW indexing, hybrid ranking, and source URLs.
Compare this with FAISS, Chroma, or Qdrant running locally. Add reranking,
structured outputs, function calling, zero-shot and few-shot prompts, evaluation,
and hallucination mitigation.

Compare context windows, temperature, and top-p. Experiment with fine-tuning, LoRA, QLoRA, and quantization only after defining a task and an evaluation set. Do not describe retrieval-index fitting as LLM fine-tuning.

## AI frameworks

The active implementation uses local PyTorch, PostgreSQL full-text search, and pgvector. Keep source loading, embeddings, vector search, reranking, and generation behind narrow project-owned interfaces.

## Computer vision

Collect Wikimedia Commons images referenced by selected articles in an optional licensed pipeline. Explore OpenCV preprocessing, CNN classification, YOLO object detection, OCR, image embeddings, and Vision Transformers. Store all image data under the configured Expansion root.

## MLOps and deployment

The CLI provides crawling, training, classification, database initialization, and local search. The project contains Docker packaging, environment configuration, atomic artifact writes, model metadata, external storage boundaries, and backend conversation persistence. Study retention schedules, deletion workflows, privacy notices, per-request opt-out behavior, pseudonymous identifiers, and the difference between identification and authentication. Extend it with Git workflows, MLflow, dataset/model versioning, data drift, model drift, CI/CD, cloud deployment on GCP/AWS/Azure, and GPU inference.

## AI system design

Study offline collection versus online search, training versus inference, batch versus real-time retrieval, CPU versus GPU, model serving, caching, vector-search architecture, RAG, agents, tool calling, conversation memory, session identifiers, streaming persistence, multi-agent systems, guardrails, privacy, and cost/latency optimization. See [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md).

## Recommended progression

1. Collect three balanced Wikipedia topics.
2. Inspect and clean the corpus.
3. Train and evaluate the classifier.
4. Search the local knowledge index.
5. Compare PostgreSQL search with the local TF-IDF knowledge index.
6. Expand the crawl from reviewed Wikipedia topic names.
7. Compare hashing vectors with locally provisioned transformer embeddings.
8. Add an LLM answer generator with citations and faithfulness evaluation.
9. Evaluate conversation naming, history opt-out, retention, and deletion behavior.
10. Add deep-learning and vision experiments.
11. Version, monitor, and deploy the system.
