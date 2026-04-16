# Questor Unified Pipeline Inference: Comprehensive Evaluation Report

This report provides a detailed breakdown of how inference works within the Questor unified pipeline. It covers the inner workings of the structured and unstructured pipelines, scoring aggregation formulas, RAG/LLaMA implementations, and the multi-agent system. Use this guide to answer detailed technical questions during project evaluations.

## 1. Unified Pipeline Orchestration
The primary entry point for inference is `run_unified.py`. This orchestrator is responsible for executing both the **Structured Pipeline** (processing tabular and numerical data) and the **Unstructured Pipeline** (processing textual documents). Once both pipelines generate their respective risk assessments, their outputs are routed to `score_combiner.py` to calculate a final unified risk score. Based on the unified risk, the pipeline determines agent routing, prioritization, and whether immediate investigation is required.

## 2. Structured Pipeline
### How it Works
The structured pipeline ingests tabular financial data (e.g., CSV, JSON), transforms it into feature vectors using `StandardScaler` or specific preprocessing pipelines, and passes it through an extensive ensemble of machine learning models.

### Models Used
The pipeline utilizes **18 different machine learning models** ranging from simple classifiers to deep learning constructs to capture different dimensions of fraud (linear, non-linear, sequential, and outlier anomalies):
- **Supervised Learning**: XGBoost, LightGBM, CatBoost, Random Forest, Decision Tree, Logistic Regression, SVM, Deep Neural Networks (DNN), CNN, and LSTM.
- **Unsupervised / Anomaly Detection**: Isolation Forest, OneClassSVM, LocalOutlierFactor, DBSCAN, Autoencoder, PCA Anomaly, KMeans, and Gaussian Mixture Models (GMM).

The structured pipeline calculates its risk score based on the consensus of these models (e.g., how many models flag the data point as fraudulent over the total number of models) and their predictive probabilities.

## 3. Unstructured Pipeline
### How it Works
The unstructured pipeline analyzes textual financial documents (e.g. 10-K filings, news, reports) through natural language processing. 

1. **Entity and Relationship Extraction**: It uses the memory-optimized `spaCy` NLP model (typically `en_core_web_sm`) to chunk text and extract entities such as Operations, Persons, Monetary Amounts, and predefined Financial Terms/Fraud Indicators. It also builds subject-predicate-object relationships (e.g., "Company X" -> CONCEALED -> "$1M").
2. **Knowledge Graph and Vector Database**: Extracted chunks and relationships are embedded into a vector database to retain contextual search capabilities for Retrieval-Augmented Generation (RAG).
3. **Risk Scoring**: It calculates an unstructured risk score (out of 100) using four weighted components:
   - **Fraud Indicators (35%)**: Keyword occurrences like "fictitious", "manipulation", "restatement".
   - **Entity Risk (25%)**: Density of financial terms and organizational complexities.
   - **Financial Anomalies (25%)**: Recognition of unusually large monetary amounts.
   - **Relationship Risk (15%)**: Suspicious linkages detected in text (e.g., concealment relationships).

## 4. RAG Implementation and LLaMA Model Usage
The unstructured pipeline significantly relies on Retrieval-Augmented Generation (RAG) driven by the **LLaMA 3.1 8B Instant** model via the Groq API (`llama-3.1-8b-instant`).

### Process Flow (RagAnalyzer)
- **What is sent to LLaMA**: The pipeline takes the raw document text (truncated to a maximum of 12,000 characters to fit the context window limits) alongside predefined fraud patterns loaded from `fraud_patterns.yaml`.
- **System Instructions**: The model receives a system prompt assigning it the persona of an *expert forensic accountant and financial risk analyzer*.
- **Output Requirements**: The prompt enforces a strict JSON-only output rule. If LLaMA detects any conditions outlined in the YAML definitions, it returns an array containing the matched query name, specific text indicators found, and a `metadata` object specifying the Risk Level (CRITICAL, HIGH, MEDIUM, LOW, MINIMAL) and SEC Rule violations.
- **Why this model?**: LLaMA is employed specifically to catch complex semantics, obfuscated language, and advanced contextual fraud patterns that standard NLP entity-matching/regex rules are likely to miss.

## 5. Score Aggregation Formula (Structured + Unstructured)
When both pipeline scores are passed to `score_combiner.py`, they are merged using the following rules engine:

### Base Formula
```python
combined_score = (structured_score * 0.6) + (unstructured_score * 0.4)
```
- **Weights**: The structured score is assigned a **60% (0.6)** weight because tabular data (hard numbers) typically serves as a more reliable and quantifiable indicator of fraud. The unstructured score adds crucial context but carries slightly less weight at **40% (0.4)**.

### Conflict Handling & Penalties
- **Score Conflicts**: If the difference between the structured and unstructured scores exceeds a threshold (`>30` points), the system flags a "score conflict warning", lowers the confidence score `max(0.5, 1.0 - (score_diff / 100))`, and adopts the *higher* of the two respective risk levels for safety.
- **Missing Data Single Source Penalty**: If only one pipeline returns data (e.g., missing documents or missing tabular records), a `0.8` multiplier penalty is applied to the available score to reflect the lack of cross-validation.

## 6. Multi-Agent System 
Once the pipeline combined score is formulated, the data can be further evaluated by a swarm of specialized domain agents orchestrated by `AgentOrchestrator`.

### How Multi-Agents Work
Agents are tailored to specific forensic accounting strategies (e.g., Benford's Law Agent, Beneish M-Score Agent, Altman Z-Score Agent, Expense Padding Agent). The orchestrator routes the financial data to these agents (ignoring offline or explicitly disabled agents). Each active agent generates its own localized `score` and `confidence` metric based on specialized fraud indicators metrics.

### Agent Aggregation Formula
When agents finish executing, the orchestrator combines their outputs using a normalized weighted average:

1. **Weight Retrieval**: The orchestrator grabs the assigned significance weight for each successful agent.
2. **Normalization**: Weights are normalized so their sum precisely equals `1.0`:
   ```python
   normalized_weight = individual_weight / total_weight_of_all_successful_agents
   ```
3. **Agent Combined Score & Confidence**:
   ```python
   agent_score = Sum(agent.score * normalized_weight for all successful agents)
   agent_confidence = Sum(agent.confidence * normalized_weight for all successful agents)
   ```

### Final Integration (Pipeline + Agents)
If agent results are available, `score_combiner.py` merges the unified pipeline score and the combined agent score:
1. `pipeline_weight` = structured_weight (0.6) + unstructured_weight (0.4) = `1.0`
2. `agent_weight_sum` is computed as the total base/assigned weights of the executed agents.
3. The formula re-normalizes these two meta-weights:
   ```python
   norm_pipeline_weight = 1.0 / (1.0 + agent_weight_sum)
   norm_agent_weight = agent_weight_sum / (1.0 + agent_weight_sum)

   final_score = (pipeline_score * norm_pipeline_weight) + (agent_score * norm_agent_weight)
   final_confidence = (pipeline_confidence * norm_pipeline_weight) + (agent_confidence * norm_agent_weight)
   ```
This final score categorizes the overall threat (MINIMAL, LOW, MEDIUM, HIGH, CRITICAL) and the system maps out final agent routing heuristics for subsequent human and automated review (e.g., assigning a Case to a `fraud_investigation_agent`).
