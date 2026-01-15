# Context-Aware Fintech User Feedback Classification & Urgency Detection
## Overview

Fintech companies receive thousands of user feedback entries daily, many of which contain critical operational issues such as failed transactions, account access problems, fraud, or compliance-related complaints. 

This project focuses on moving beyond sentiment analysis to build a decision-driven feedback classification system that helps fintech teams:

- Understand what issue a user is facing
- Determine how urgent the issue is
- Prioritize responses based on operational risk rather than emotion

The system is designed with real-world fintech use cases in mind (e.g. Revolut, Wise, Monzo).

## Problem Statement

Traditional sentiment analysis labels feedback as positive or negative, but in fintech, this is often misleading. A review can be polite or appreciative while still describing a high-risk issue such as fraud, blocked funds, or account compromise.

**Example:** *“I love the app, but my account was hacked and I can’t access my money.”*

- **Sentiment:** Positive
- **Operational risk:** Critical

This project reframes user feedback analysis to focus on issue understanding and urgency detection, enabling better triage and escalation decisions.

## Objectives
The system aims to:

- **Classify user feedback into issue categories**
  - Transaction issues
  - Account access & security
  - Refunds & reversals
  - KYC
  - App performance
  - Customer support experience
  - Financial products
  - Product feedback
  - General inquiries
- **Detect urgency**
  - Identify feedback that requires immediate attention
  - Reduce the risk of delayed responses to critical issues
- **Support operational decision-making**
  - Prioritization
  - Routing to the right teams
  - Risk-aware triage

## Dataset & Annotation Process
The dataset was manually constructed to reflect real-world fintech user feedback, with labeling driven by operational risk rather than sentiment.

### Data Collection
- User reviews were **manually collected** from:
   - Google Play Store
   - Apple App Store
- Reviews were copied sequentially and curated to ensure:
   - Relevance to fintech use cases
   - Coverage across multiple issue types and multiple star ratings
   - Realistic user language and phrasing

### Data Preparation
- Raw reviews were cleaned and normalized
- Duplicates and low-signal entries were removed
- Text preprocessing included lowercasing and basic normalization

### Labeling & Guidelines
Each feedback entry was annotated with:
- **Issue Category**
- **Urgency Level**
- **clean_feedback_text word count**

A **primary labeling guideline** was created before annotation, inspired by:
- Fintech industry complaint taxonomies
- Operational support workflows
- Prior hands-on experience in data annotation and evaluation

Labels were assigned using a **human-in-the-loop process**, prioritizing:
- Operational risk
- User intent
- Business impact over emotional tone

## Exploratory Data Analysis (EDA)

Initial exploratory analysis was conducted in Excel to:
- Inspect class distributions
- Identify label imbalance
- Validate urgency proportions
- Spot annotation inconsistencies early

These insights informed modeling decisions such as:
- Stratified train–test splitting
- Macro-averaged evaluation metrics
- The decision to model urgency separately

## Modeling Approach
### **Issue Classification**
- **Model:** Logistic Regression (multiclass)
- **Features:** TF-IDF (unigrams)
- **Reasoning:**
  - Interpretable
  - Strong baseline for text classification
  - Fast to train and deploy

### **Urgency Detection**
- **Model:** Logistic Regression (binary)
- **Classes:** High urgency vs Not-High urgency
- **Evaluation focus:** Recall on high-urgency cases
(Missing urgent issues is more costly than false alarms)

## Evaluation Results
### Urgency Classification Performance
- **Accuracy:** 70%
- **High Urgency Recall:** 57.7%

The urgency model was evaluated with an emphasis on recall for high-risk cases, reflecting real-world fintech triage priorities. While overall accuracy is reasonable, the key metric is high-urgency recall, which highlights the model’s ability to surface critical issues for escalation. Misclassification analysis was performed to identify:
- Ambiguous language
- Polite phrasing masking serious issues
- Contextual gaps (to be addressed in future work)

## Key Insights
- Sentiment alone is insufficient for fintech feedback analysis
- Urgency and issue type provide more actionable signals
- Classical NLP models can deliver meaningful value with proper framing
- Error analysis is essential for understanding real-world failure modes

## Limitations
- Small dataset limits generalization
- Bag-of-words TF-IDF does not fully capture context
- Class imbalance affects high-urgency recall

## Future Work (Version 2)
- Contextual embeddings (BERT / sentence transformers)
- Joint learning of issue category and urgency
- Cost-sensitive learning to further improve high-urgency recall
- Deployment-oriented pipeline (API / dashboard integration)

# Why This Matters
In fintech operations, delayed handling of high-risk issues can lead to:

- Financial loss
- Regulatory exposure
- User churn

This project demonstrates how NLP can support operational triage, not just sentiment reporting.

## Tech Stack
- Python
- scikit-learn
- pandas
- numpy
- Jupyter / VS Code
- Excel / Googlesheets
