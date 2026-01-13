# README

## Quick Start

This repository provides a reproducible evaluation framework for
quantitative analysis of hallucination behavior in large language models (LLMs),
as reported in *Quantitative Evaluation of Hallucination Behavior in Large Language Models*.

### 1. Requirements
- Python 3.9+
- Required libraries: `torch`, `transformers`, `numpy`, `pandas`
- Access to target LLM backends (e.g., GPT, Gemini, Qwen, LLaMA, Mistral)
- Backend-specific SDKs may be required (e.g., OpenAI or Google GenAI clients)

### 2. Run evaluation (Task B)
The following example runs Task B under the *abstain-allowed* regime:

```bash
python hallucination_evaluation.py \
  --task taskB \
  --regime abstain \
  --backend gpt \
  --model GPT5 \
  --repeat 4
```

### 3. Extract quantitative metrics
After execution, aggregate results into a CSV file:
```bash
python extract_evaluation_metrics.py logs/ > metrics.csv
```

### 4. Reproducibility
All experiments use explicitly controlled random seeds.

Execution environments are documented in environment_runpod.txt and
environment_windows_wsl.txt.

Full command histories are provided in execution_procedure.txt.

For experimental design, evaluation metrics, and interpretation,
see the sections below.


## 1. Experimental Objective (Purpose)

The goal of this experiment was to establish a **quantitative and reproducible framework** for evaluating hallucination behavior in large language models.

Instead of treating hallucination as a binary failure, we aimed to decompose it into **measurable components** such as forced answering errors, abstention behavior, ambiguity sensitivity, and order dependence under controlled conditions.

In particular, the experiment was designed to test whether hallucination can be evaluated as a **model-specific behavioral tendency**, rather than as a purely stochastic artifact.

---

## 2. Experimental Design (High-level)

We constructed a controlled task (**Task B**) where models are required to select a **single valid code** from a set of notes.

The task includes:
- **True notes**
- **Ambiguous fake notes** sharing the same context
- **Systematic variation of ambiguity levels**

Two regimes were evaluated:

- **Forced**  
  The model must output a code.  
  `UNKNOWN` is forbidden.

- **Abstain**  
  The model may output `UNKNOWN` when appropriate.

To ensure reproducibility, **all randomness was fixed via explicit seeds**, and experiments were repeated across multiple ambiguity levels and repetitions.


### Task Definition

Two task variants were originally considered:

- **Task A**: a baseline single-note selection task without ambiguity.
- **Task B**: a multi-note selection task with controlled ambiguity.

Preliminary analysis showed that Task A produces near-trivial behavior
with minimal hallucination signals.
Therefore, this repository focuses exclusively on Task B,
which is necessary to elicit and quantify hallucination behavior.

---

## 3. Evaluation Criteria (What is measured)

The evaluation framework explicitly separates different failure modes:

- **Accuracy**  
  Correctness when a valid answer exists.

- **Unknown rate**  
  Frequency of abstention.

- **Unknown correctness**  
  Whether abstention occurs only when appropriate.

- **Hallucination (H_any)**  
  Producing an incorrect but well-formed answer.

- **Forced-unknown violation**  
  Producing `UNKNOWN` when it is disallowed.

This allows hallucination to be **measured quantitatively**, rather than inferred qualitatively.

---

## 4. Key Findings (Results & Insights)

The results show that hallucination behavior is **highly model-dependent and systematic**.

- Some models (e.g. **GPT-class models**) exhibit strong abstention discipline, consistently choosing `UNKNOWN` under ambiguity and achieving near-perfect safety at the cost of lower forced accuracy.

- Other models (e.g. **Gemini-class models**) show high accuracy when input order is fixed, but a significant drop when note order is randomized, revealing strong **order sensitivity**.

- Open-source models (e.g. **LLaMA, Qwen, Mistral**) occupy intermediate regimes, with varying trade-offs between robustness and hallucination rate.

Importantly, these behaviors are **stable across repeated runs**, indicating that they reflect **intrinsic model characteristics** rather than sampling noise.

---

## 5. Significance (Why this matters)

This experiment demonstrates that hallucination can be:

- **Quantitatively measured**
- **Compared across models**
- **Interpreted as a behavioral or “personality-like” property of models**

These properties are influenced by constraints such as **forced answering** and **input ordering**.

The framework provides a **practical basis** for evaluating hallucination robustness in a **reproducible and model-agnostic** manner.

---

## 6. Implementation

**One-line summary (for collaborators):**

The full evaluation code and all raw logs are available, with fixed seeds and complete execution traces, enabling **exact reproduction** of all reported results.


## Citing this work
If you use this repository, please cite:
Chino, H. (2026). Quantitative Evaluation of Hallucination Behavior in Large Language Models.
Zenodo. https://doi.org/10.5281/zenodo.17756906