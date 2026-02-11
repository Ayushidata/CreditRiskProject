#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv
from torch_geometric.explain import GNNExplainer
from fairlearn.metrics import MetricFrame, demographic_parity_difference
from torch_geometric.utils import from_networkx, to_networkx

import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

st.set_page_config(layout="wide")
st.title("💳 Credit Risk Scoring with GNN + Explainability + Bias Detection")

sample_size = st.sidebar.slider("Sample Size", 500, 10000, 1000, step=500)
k_neighbors = st.sidebar.slider("k-Nearest Neighbors", 3, 20, 10)
epochs = st.sidebar.slider("Training Epochs", 10, 100, 50, step=10)
graph_nodes = st.sidebar.slider("Max Graph Nodes to Visualize", 10, 500, 100)

file_path = r"C:\\Users\\Ayushi\\Downloads\\credit_risk_gnn\\data\\accepted_2007_to_2018Q4.csv"
data = None


# In[2]:


def preprocess_lendingclub_data(path):
    df = pd.read_csv(path, low_memory=False)
    df = df[['loan_amnt', 'int_rate', 'annual_inc', 'fico_range_high', 'loan_status', 'title']]
    df.dropna(inplace=True)
    df = df.sample(n=10000, random_state=42)
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
    df['loan_status'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})
    return df.reset_index(drop=True)



# In[3]:


def build_similarity_graph(df, k=10):
    features = ['loan_amnt', 'int_rate', 'annual_inc', 'fico_range_high']
    X = StandardScaler().fit_transform(df[features])
    sim_matrix = cosine_similarity(X)
    np.fill_diagonal(sim_matrix, 0)

    G = nx.Graph()
    for idx in df.index:
        G.add_node(idx, x=torch.tensor(X[idx], dtype=torch.float), y=int(df.loc[idx, 'loan_status']))

    for i in range(len(df)):
        top_k = np.argsort(sim_matrix[i])[-k:]
        for j in top_k:
            G.add_edge(i, j)

    data = from_networkx(G)
    data.x = torch.stack([data.x[i] for i in range(len(data.x))])
    data.y = torch.tensor([data.y[i] for i in range(len(data.y))], dtype=torch.long)
    return data


# In[4]:


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x





# In[5]:

def train(model, data, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()

    data.train_mask = torch.rand(len(data.y)) < 0.8
    data.test_mask = ~data.train_mask

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    return model, data



# In[6]:


def detect_bias(y_true, y_pred, sensitive_attr):
    frame = MetricFrame(
        metrics=accuracy_score,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_attr
    )
    dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_attr)
    return frame.by_group.to_frame("Accuracy"), dpd


# In[8]:
def draw_graph(data, max_nodes=100):
    G_nx = to_networkx(data)
    G_small = nx.subgraph(G_nx, list(range(min(max_nodes, data.num_nodes))))
    pos = nx.spring_layout(G_small, seed=42)
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw(G_small, pos, with_labels=False, node_size=50, ax=ax)
    st.pyplot(fig)

def show_interactive_graph(data, max_nodes=100):
    G_nx = to_networkx(data)
    G_small = nx.subgraph(G_nx, list(range(min(max_nodes, data.num_nodes))))
    net = Network(height="600px", width="100%", notebook=False)
    net.from_nx(G_small)
    net.write_html("graph.html")
    with open("graph.html", "r", encoding='utf-8') as f:
        html_content = f.read()
        components.html(html_content, height=600, scrolling=True)




if st.button("Run Model Pipeline"):
    with st.spinner(" Loading and preprocessing data..."):
        df = preprocess_lendingclub_data(file_path)
        st.success("Data loaded!")
        st.write("Sample data:", df.head())

        with st.spinner("Building similarity graph..."):
            data = build_similarity_graph(df)
            st.success("Graph created!")
            st.write(f" Nodes: {data.num_nodes} | Edges: {data.num_edges}")

        with st.spinner("Training GCN model..."):
            model = GCN(input_dim=data.x.shape[1])
            model, data = train(model, data, epochs=50)
            st.success("Model trained!")

        model.eval()
        out = model(data)
        pred = out.argmax(dim=1)

        st.write("Predictions on test set:")
        st.write(pred[data.test_mask].numpy().tolist())

        st.write("---")
        st.subheader("Bias Detection Results (using `title` column as gender proxy):")
        sensitive_attr = df.loc[data.test_mask.numpy(), 'title']
        acc_by_group, dp_diff = detect_bias(data.y[data.test_mask], pred[data.test_mask], sensitive_attr)

        st.write("Accuracy by Title Group:", acc_by_group)
        st.write(f"Demographic Parity Difference: `{dp_diff:.4f}`")

        st.write("---")
        st.subheader("GNN Node Explanation (subgraph view)")
        if st.button("Explain Node 10"):
            explainer = GNNExplainer(model, epochs=100)
            node_feat_mask, edge_mask = explainer.explain_node(10, data.x, data.edge_index)
            explainer.visualize_subgraph(10, data.edge_index, edge_mask, y=data.y)
            st.pyplot(plt.gcf())
            

    st.download_button("Download Predictions CSV", pd.DataFrame({"prediction": pred.cpu().numpy()}).to_csv(index=False), file_name="predictions.csv")

    st.write("---")
    st.subheader("Borrower Similarity Graph")
    draw_graph(data, max_nodes=graph_nodes)

    st.write("---")
    st.subheader("Interactive Graph")
    show_interactive_graph(data, max_nodes=graph_nodes)
    




# %%


