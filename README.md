# AI Applications Project Documentation 

Use this template to document your project concisely and completely.
Fill in all required fields. Keep answers short and precise.

## Documentation Hint

Important:
When possible, reference the corresponding code location directly in your description.

### Example: Reference to a notebook section
Reference to the header `## Data Preprocessing` in the notebook `analysis.ipynb`:

> See *Data Preprocessing* in
> [`analysis.ipynb`](analysis.ipynb#data-preprocessing)

### Example: Reference to Python code

Reference to a single line in `model.py`, line 42:
> [`model.py`, line 42](model.py#L42)

Reference to multiple lines in `train.py`, lines 15-38:
> [`train.py`, lines 15-38](train.py#L15-L38)

## Project Metadata

- Project title:
- Student:
- GitHub repository URL:
- Deployment URL:
- Submission date:

### Mandatory Setup Checks

- [ ] At least 2 blocks selected
- [ ] Multiple and different data sources used
- [ ] Deployment URL provided
- [ ] Required GitHub users added to repository (`jasminh`, `bkuehnis`)

## Selected AI Blocks

- [ ] ML Numeric Data
- [ ] NLP
- [ ] Computer Vision

Primary blocks used for core solution (choose 2):
- Primary block 1:
- Primary block 2:

If a third block is selected, it is documented and graded separately as extra work.

Guidance hint: Keep the project idea short and consistent. Focus most details on the selected blocks.
Evidence hint: Show where each selected block contributes to the final system.

---

## 1. Project Foundation (Short)

### 1.1 Problem Definition
- Problem statement:
- Goal:
- Success criteria:

### 1.2 Integration Logic
- How the selected blocks interact:
- Data and output flow between blocks:

Guidance hint: This section should be short. The detailed work belongs in block sections.
Evidence hint: Include one clear pipeline overview.

---

## 2. Block Documentation

Complete only selected blocks. Mark non-selected block sections as N/A.

### 2A. ML Numeric Data (If selected)

#### 2A.1 Data Source(s)
List every usage of a data source as a separate entry. If the same source is used twice for different roles, add it twice.

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |

#### 2A.2 Preprocessing and Features
- Cleaning steps:
- Preprocessing steps:
- Feature engineering and selection:

#### 2A.3 Model Selection
- Models tested:
- Why these models were chosen:

#### 2A.4 Model Comparison and Iterations
| Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |

#### 2A.5 Evaluation and Error Analysis
- Metrics used:
- Final results:
- Error patterns and likely causes:

#### 2A.6 Integration with Other Block(s)
- Inputs received from other block(s):
- Outputs provided to other block(s):

Guidance hint: Keep entries practical and evidence-based.
Evidence hint: Add values, not only claims.

### 2B. NLP (If selected)

#### 2B.1 Data Source(s)
List every usage of a data source as a separate entry. If the same source is used twice for different roles, add it twice.

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |

#### 2B.2 Preprocessing and Prompt Design
- Text preprocessing:
- Prompt design or retrieval setup:

#### 2B.3 Approach Selection
- Approach used (classical NLP, transformer, RAG, prompt engineering):
- Alternatives considered:

#### 2B.4 Comparison and Iterations
| Iteration | Objective | Key changes | Model or prompt setup | Main metric or qualitative check | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |

#### 2B.5 Evaluation and Error Analysis
- Evaluation strategy:
- Results:
- Error patterns and likely causes:

#### 2B.6 Integration with Other Block(s)
- Inputs received from other block(s):
- Outputs provided to other block(s):

Guidance hint: Show concrete prompt or retrieval decisions.
Evidence hint: Include representative outputs or failure cases.

### 2C. Computer Vision (If selected)

#### 2C.1 Data Source(s)
List every usage of a data source as a separate entry. If the same source is used twice for different roles, add it twice.

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |

#### 2C.2 Preprocessing and Augmentation
- Image preprocessing:
- Augmentation strategy:

#### 2C.3 Model Selection
- Vision model(s) used:
- Why these model(s) were chosen:

#### 2C.4 Model Comparison and Iterations
| Iteration | Objective | Key changes | Model(s) used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |

#### 2C.5 Evaluation and Error Analysis
- Metrics and/or visual checks:
- Final results:
- Error patterns and limitations:

#### 2C.6 Integration with Other Block(s)
- Inputs received from other block(s):
- Outputs provided to other block(s):

Guidance hint: Use concise examples from real predictions.
Evidence hint: Include sample outputs and observed failure cases.

---

## 3. Deployment

- Deployment URL:
- Main user flow:
- Screenshot or short demo:

Guidance hint: Deployment must be usable.
Evidence hint: Add screenshots or short demo references.

---

## 4. Execution Instructions

- Environment setup:
- Data setup:
- Training command(s):
- Inference/run command(s):
- Reproducibility notes:

Guidance hint: Another person should be able to run your project from this section.
Evidence hint: Include exact commands and versions.

---

## 5. Optional Bonus Evidence

Use this section for exceptional work beyond the core requirements.

- [ ] Third selected block implemented with strong quality
- [ ] More than two data sources used with clear added value
- [ ] A core section is done exceptionally well
- [ ] Extended evaluation
- [ ] Ethics, bias, or fairness analysis
- [ ] Creative or exceptional use case

Evidence for selected bonus items:
