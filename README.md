Credit Risk Scoring using Graph Neural Networks (GNN)

This project develops a Credit Risk Scoring System using Graph Neural Networks (GNNs) to predict loan default risk by modeling relationships between borrowers. Unlike traditional machine learning models that treat borrowers independently, this system captures hidden similarities such as income patterns, employment type, and credit behavior through graph-based learning using GCN and GraphSAGE architectures.

The project also focuses on Explainable AI and Fairness in financial decision-making. SHAP and LIME are used to provide transparent explanations for predictions, while Fairlearn and AIF360 help detect bias across sensitive attributes like gender, income group, and employment type. Fairness metrics such as Demographic Parity, Equalized Odds, and Disparate Impact Ratio are used for evaluation.

The workflow includes data preprocessing, borrower similarity graph construction, GNN model training, performance evaluation, explainability analysis, and fairness assessment. The system is designed for datasets such as LendingClub and supports features like loan amount, annual income, interest rate, credit score, debt-to-income ratio, and loan status.

An optional Streamlit web application is included to visualize predictions, graph relationships, feature importance, and fairness metrics through an interactive dashboard.

Key Features
Borrower similarity graph construction
Credit default prediction using GCN/GraphSAGE
SHAP and LIME explainability integration
Bias detection using Fairlearn/AIF360
Interactive Streamlit dashboard
Scalable and modular project structure
Tech Stack
Python
PyTorch & PyTorch Geometric
Scikit-learn
SHAP
Fairlearn / AIF360
Pandas & NumPy
NetworkX
Streamlit
