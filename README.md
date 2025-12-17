
# 📊 Benchmarking Graph Neural Network Architectures on Cora

## 1. Project Overview

This project benchmarks multiple **Graph Neural Network (GNN)** architectures on a **citation network dataset (Cora)** to understand:

* Why **graph structure matters**
* How different **GNN families** behave
* How GNNs compare against a **non-graph baseline (MLP)**

The models implemented and compared are:

* **MLP** (baseline, no graph)
* **GCN** (Graph Convolutional Network — convolutional GNN)
* **GraphSAGE** (Message-passing GNN)
* **GAT** (Graph Attention Network — attentional GNN)

All models are evaluated on the **same dataset, splits, loss, and training protocol** to ensure a **fair architectural comparison**.

---

## 2. Dataset: Cora Citation Network

The **Cora** dataset is a standard benchmark for node classification tasks.

### Dataset properties:

* **Nodes**: Research papers
* **Edges**: Citation links between papers
* **Node features**: 1,433-dimensional bag-of-words vectors
* **Labels**: 7 research topic classes
* **Task**: Node-level classification

The dataset exhibits **high homophily**, meaning connected nodes tend to share labels — an important property for GCN-style models.

---

## 3. Task Definition

### Problem type:

**Node classification**

Each node (paper) must be assigned a class label based on:

* Its own features
* The structure of the graph
* Information from neighboring nodes (for GNNs)

---

## 4. Encoder–Decoder Framework (Unified View)

All models in this project follow the same **Enc / Dec / Gt / L** formulation.

### Encoder (Enc)

Maps input features (and graph structure for GNNs) to node embeddings:


### Decoder (Dec)

Maps embeddings to class probabilities:


### Ground Truth (Gt)

True node labels provided by the dataset.

### Loss (L)

Categorical cross-entropy between predictions and ground truth.

---

## 5. Models Implemented

### 5.1 MLP Baseline (No Graph)

**Notebook:** `02_mlp_baseline.ipynb`

#### Description

A standard multilayer perceptron that uses **only node features**, completely ignoring graph structure.

#### Encoder

Stack of dense layers applied independently to each node.
#### Decoder

Softmax classifier
#### Key limitation

* Nodes are treated as **independent samples**
* No neighborhood information
* Cannot exploit relational structure

#### Purpose

Acts as a **lower bound baseline**.

---

### 5.2 GCN — Graph Convolutional Network

**Notebook:** `03_gcn_model.ipynb`
**GNN family:** Convolutional GNN

#### Encoder

GCN layers perform **neighborhood feature smoothing**:


* Fixed normalization based on graph structure
* Strong inductive bias toward homophily

#### Decoder

Final GCN layer with softmax activation.

#### Why GCN works well on Cora

* Cora has high label homophily
* Neighbor averaging reinforces correct labels

---

### 5.3 GraphSAGE — Message Passing GNN

**Notebook:** `04_graphsage_model.ipynb`
**GNN family:** Message-passing GNN

#### Encoder

GraphSAGE explicitly separates:

* Node’s own representation
* Aggregated neighbor representation


* Aggregation (mean) is abstracted by Spektral
* More flexible than GCN

#### Decoder

Final GraphSAGE layer with softmax.

#### Important note

GraphSAGE **does not guarantee higher accuracy** than GCN on homophilic graphs.
Its strength lies in **flexibility and inductive learning**, not raw performance.

---

### 5.4 GAT — Graph Attention Network

**Notebook:** `05_gat_model.ipynb`
**GNN family:** Attentional GNN

#### Encoder

Learns attention weights for each neighbor:


* Attention coefficients are learned
* Model decides which neighbors matter more

#### Decoder

Final attention layer with softmax.

#### Why GAT performs best

* Can down-weight noisy or irrelevant neighbors
* More expressive than fixed-weight GCN

---

## 6. Experimental Setup (Fair Comparison)

To ensure fairness across models:

* Same dataset (Cora)
* Same train / validation / test splits
* Same loss function (categorical cross-entropy)
* Same optimizer (Adam)
* Same number of epochs
* Similar hidden dimensions

No model-specific tuning was done during the main benchmark.

---

## 7. Results

### 7.1 Quantitative Results

From `results/metrics.csv`:

| Model     | Test Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) |
| --------- | ------------- | ----------------- | -------------- | ---------- |
| MLP       | 0.5660        | 0.5426            | 0.5810         | 0.5514     |
| GCN       | 0.8080        | 0.7916            | 0.8201         | 0.8025     |
| GraphSAGE | 0.7920        | 0.7708            | 0.7950         | 0.7809     |
| GraphSAGE | 0.7940        | 0.7776            | 0.7994         | 0.7865     |
| GAT       | **0.8170**    | **0.7960**        | **0.8222**     | **0.8074** |

---

## 8. Analysis and Discussion

### 8.1 Why MLP Performs Poorly

* Ignores graph structure
* Cannot exploit label homophily
* Treats nodes as i.i.d. samples

### 8.2 Why GCN Outperforms MLP

* Aggregates neighbor information
* Enforces smoothness over the graph
* Matches the homophilic nature of Cora

### 8.3 Why GraphSAGE Is Slightly Lower Than GCN

* More flexible but less biased toward homophily
* Designed for generalization, not necessarily peak accuracy on static graphs

### 8.4 Why GAT Performs Best

* Learns which neighbors are important
* Reduces noise from less relevant connections
* More expressive aggregation mechanism

---

## 9. Key Takeaways

* Graph structure dramatically improves performance
* Different GNN families embody different inductive biases
* No single GNN is universally best — performance depends on graph properties
* Fair benchmarking is essential for meaningful comparison

---

## 10. Future Work

* Hyperparameter tuning (layers, hidden size, dropout)
* Compare different GraphSAGE aggregators
* Study oversmoothing effects
* Extend to other datasets (Citeseer, PubMed)

---

## 11. Conclusion

This project demonstrates, both conceptually and empirically, how **Graph Neural Networks leverage relational data** and how different GNN families trade off **bias, flexibility, and expressiveness**.
The encoder–decoder framework provides a unified lens for understanding all models, from MLPs to attention-based GNNs.


