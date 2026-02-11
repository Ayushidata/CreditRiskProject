# CreditRiskProject
Credit Risk Scoring using Graph Neural Networks (GNN) with Explainability and Bias Detection
Overview

This project implements a Credit Risk Scoring System using Graph Neural Networks (GNNs) to model borrower relationships and predict loan default risk. Unlike traditional machine learning models that treat borrowers independently, this system captures relational patterns by constructing a borrower similarity graph and applying Graph Convolutional Networks (GCN) / GraphSAGE.

The project also integrates:

Explainable AI (SHAP/LIME) for model transparency

Bias Detection (Fairlearn/AIF360) for fairness evaluation

Optional Streamlit-based interactive web application

This solution demonstrates how graph-based learning improves predictive performance while maintaining interpretability and fairness in financial risk modeling.

Problem Statement

Traditional credit scoring models fail to capture hidden relationships between borrowers (e.g., similar income profiles, employment types, credit history patterns). This project addresses:

Improved default prediction using graph-based learning

Transparent decision-making using explainability tools

Fairness assessment across sensitive attributes

Key Features

Graph construction based on borrower similarity

GCN / GraphSAGE model for default prediction

SHAP-based feature importance analysis

Fairness metrics evaluation

Bias detection using Fairlearn or AIF360

Interactive Streamlit dashboard (optional)

Tech Stack

Python

PyTorch

PyTorch Geometric

Scikit-learn

SHAP

Fairlearn / AIF360

Pandas, NumPy

NetworkX

Streamlit (for UI)

Dataset

The model is designed to work with structured credit datasets such as:

LendingClub dataset

Financial loan transaction datasets

Typical features include:

Loan amount

Interest rate

Annual income

Employment length

Credit score

Debt-to-income ratio

Loan status (target variable)

Project Structure
credit_risk_gnn_app/
│
├── data/                      # Raw and processed datasets
├── models/                    # GNN model definitions
├── utils/                     # Helper functions
├── explainability/            # SHAP/LIME analysis scripts
├── fairness/                  # Bias detection scripts
├── app.py                     # Streamlit application
├── train.py                   # Model training script
├── graph_builder.py           # Borrower similarity graph construction
├── requirements.txt
└── README.md

Methodology
1. Data Preprocessing

Handling missing values

Encoding categorical variables

Feature scaling

2. Graph Construction

Nodes represent borrowers

Edges represent similarity (e.g., cosine similarity, k-nearest neighbors)

Adjacency matrix creation

3. Model Architecture

Example GNN Architecture:

Input layer

Graph Convolution Layer (GCNConv or GraphSAGE)

ReLU activation

Dropout

Fully connected output layer

Sigmoid activation for binary classification

Loss Function:

Binary Cross Entropy

Evaluation Metrics:

Accuracy

Precision

Recall

F1-score

ROC-AUC

Explainability

To ensure transparency:

SHAP values are computed for feature importance

Global and local explanations are generated

Important drivers of default risk are visualized

This helps financial institutions justify lending decisions.

Fairness and Bias Detection

Fairness analysis includes:

Demographic Parity

Equalized Odds

Disparate Impact Ratio

Bias detection is performed across sensitive attributes such as:

Gender

Employment type

Income group

Mitigation strategies can be integrated if bias is detected.
