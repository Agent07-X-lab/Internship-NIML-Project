# Graph Transformer-based Predictive Maintenance System

An end-to-end spatiotemporal Graph Deep Learning platform designed to monitor home appliance degradation, simulate gradual wear-and-tear failures, forecast Remaining Useful Life (RUL), identify failures via Root Cause Analysis (RCA), and render an interactive web control dashboard. The platform utilizes a **Graph Transformer Autoencoder (GTAE)** to process multi-appliance behavioral interactions.

---

## 📖 Project Overview

Modern smart homes contain multiple electrical appliances whose operating cycles are often co-dependent or correlated (e.g., washing machines followed by tumble dryers, or water heaters activating during morning routines). Traditional anomaly detection models treat each appliance as an isolated timeseries, ignoring these relational dynamics. 

This project solves this limitation by framing a household's appliances as a **dynamic behavioral graph**. By analyzing both temporal energy fluctuations (node features) and co-activation patterns (edge connectivities), the Graph Transformer Autoencoder (GTAE) models the normal structural and features baseline of a household. When an appliance begins to degrade, its power consumption behavior or duty cycle drifts, yielding elevated reconstruction errors that are utilized to flag anomalies, predict severity, and estimate RUL.

---

## 📂 Project Directory Structure

```text
├── 1_raw_data/                        # Raw REFIT smart home datasets
│   ├── CLEAN_House1.csv               # Household 1 raw power readings (CSV format)
│   └── CLEAN_House2.csv ...           # Household 2-21 raw power readings
├── 3_processed_outputs/               # Compiled datasets, model weights, and reports
│   ├── Anomaly_Report_PM.json         # Master database of all processed households
│   ├── PM_Report_House_[ID].json      # Multi-metric details for individual household
│   ├── PM_Report_House_[ID].html      # Styled, self-contained interactive audit report
│   ├── House_[ID]_Processed.csv       # Standardized resampled timeseries per house
│   ├── House_[ID]_Dynamic_Graphs.pt   # Serialized PyTorch graph sequence tensors
│   └── House_[ID]_GTAE_PM.pth         # Saved GTAE model state-dict weights
├── .venv/                             # Local Python virtual environment
├── .vscode/                           # IDE preferences (points to python.exe path)
├── pm_config.json                     # Threshold parameters and RUL degradation configurations
├── pm_dynamic_graph.py                # Multi-feature graph generator and Graph Drift calculator
├── pm_fault_injector.py               # Progressive wear-and-tear degradation simulator
├── pm_analytics.py                    # Health Index, Severity, RUL forecasting, and RCA algorithms
├── pm_xai.py                          # Backpropagation feature saliency and edge attention tracker
├── pm_report_exporter.py              # Self-contained JSON and HTML report generators
├── pm_pipeline.py                     # Master execution runner (supports single or batch processing)
├── server.py                          # Local HTTP backend server exposing routes for dashboard and reports
├── graph_transformer.py               # GTAE PyTorch neural network architecture
├── refit_processor.py                 # Timeseries resampling and cleaning preprocessor
├── Logo.png                           # Primary UI header logo asset
├── aegis_logo.png                     # Secondary alternate logo asset
└── Predictive_Maintenance_Dashboard.html # Final compiled dashboard file
```

---

## 🚀 How to Run the Project

All execution commands must be run from the project root directory using the local Python virtual environment interpreter:

### 1. Batch Process All Houses
To sequentially preprocess datasets, extract graphs, train models, inject gradual faults, forecast RUL, and output HTML/JSON reports for all 21 households, execute:
```bash
.venv\Scripts\python pm_pipeline.py --house all --epochs 15
```

### 2. Process an Individual House (or Selected Houses)
To execute the pipeline for a single house (e.g. House 1):
```bash
.venv\Scripts\python pm_pipeline.py --house 1 --epochs 15
```
To run a specific list of households (e.g. House 1, 3, and 5):
```bash
.venv\Scripts\python pm_pipeline.py --house 1,3,5 --epochs 15
```

### 3. Launch the Interactive Dashboard
To launch the backend API and serve the dark-glassmorphism control dashboard:
```bash
.venv\Scripts\python server.py
```
Open your browser and navigate to: **[http://localhost:8000](http://localhost:8000)**

---

## 📐 Complete Mathematical Framework

Below is the complete mathematical notation and formulation of the preprocessing, graph building, neural network architecture, and forecasting layers:

### 1. Timeseries Preprocessing & Normalization
For a household with $N$ appliances, raw power readings are sampled. To clean and normalize amplitude scales across devices with different power profiles (e.g., a $3000\text{W}$ kettle vs a $100\text{W}$ television), the raw power $P_{n, t}$ for appliance $n$ at time $t$ is normalized against its historical maximum:

$$X_{n, t}^{(0)} = \frac{P_{n, t}}{\max_{t} (P_n) + \epsilon}$$

Where:
*   $P_{n, t}$: Raw power reading of appliance $n$ at timestamp $t$.
*   $\max_{t} (P_n)$: Maximum power observed for appliance $n$ in the training baseline.
*   $\epsilon = 10^{-9}$: Small constant to prevent division-by-zero.
*   $X_{n, t}^{(0)}$: Normalized power value $\in [0, 1]$.

An appliance is classified as active ($S_{n, t} = 1$) if its power exceeds a configured threshold $\theta_n$:

$$S_{n, t} = \begin{cases} 1 & \text{if } P_{n, t} \ge \theta_n \\ 0 & \text{otherwise} \end{cases}$$

### 2. Dynamic Spatiotemporal Graph Construction
For a given sliding window $w$ of length $W$ steps ($W=256$), we construct a graph $\mathcal{G}_w = (\mathcal{V}, \mathcal{E}_w)$ where $\mathcal{V}$ is the set of $N$ appliance nodes, and $\mathcal{E}_w$ represents their connections.

#### Node Feature Extraction
Each node $n \in \mathcal{V}$ is represented by a 9-dimensional feature vector $\mathbf{x}_{n, w} \in \mathbb{R}^9$:

$$\mathbf{x}_{n, w} = \left[ P_{norm}, S, \mu, \sigma^2, D, L, E, \sin(\phi), \cos(\phi) \right]^T$$

Where:
1.  **Normalized Power ($P_{norm}$):** Mean normalized power in the window: $\frac{1}{W} \sum_{t \in w} X_{n, t}^{(0)}$
2.  **Binary State ($S$):** Current activation state at the end of the window: $S_{n, W}$
3.  **Rolling Mean ($\mu$):** Mean raw power in the window: $\frac{1}{W} \sum_{t \in w} P_{n, t}$
4.  **Rolling Variance ($\sigma^2$):** Variance of raw power: $\frac{1}{W} \sum_{t \in w} (P_{n, t} - \mu)^2$
5.  **Duty Cycle ($D$):** Active ratio: $\frac{1}{W} \sum_{t \in w} S_{n, t}$
6.  **Running Duration ($L$):** Total active seconds: $\left( \sum_{t \in w} S_{n, t} \right) \cdot \Delta t$ (where $\Delta t = 8\text{s}$)
7.  **Energy Wh ($E$):** Total energy consumed: $\sum_{t \in w} P_{n, t} \cdot \frac{\Delta t}{3600}$
8.  **Sine Positional Encoding ($\sin(\phi)$):** Time of day mapping: $\sin\left(\frac{2\pi \cdot \text{hour}}{24}\right)$
9.  **Cosine Positional Encoding ($\cos(\phi)$):** Time of day mapping: $\cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$

#### Multi-Channel Edge Connectivity
The adjacency tensor $\mathbf{A}_w \in \mathbb{R}^{C \times N \times N}$ incorporates $C=4$ feature similarity channels:

*   **Channel 1: Jaccard Similarity (Co-activation overlap):**
    $$A_{i,j}^{(1)} = \frac{\sum_{t \in w} (S_{i,t} \cdot S_{j,t})}{\sum_{t \in w} S_{i,t} + \sum_{t \in w} S_{j,t} - \sum_{t \in w} (S_{i,t} \cdot S_{j,t})}$$

*   **Channel 2: Pearson Correlation Coefficient (Amplitude variance):**
    $$A_{i,j}^{(2)} = \frac{\sum_{t \in w} (P_{i,t} - \bar{P}_i)(P_{j,t} - \bar{P}_j)}{\sqrt{\sum_{t \in w} (P_{i,t} - \bar{P}_i)^2 \sum_{t \in w} (P_{j,t} - \bar{P}_j)^2}}$$

*   **Channel 3: Mutual Information (Entropy correlation):**
    $$A_{i,j}^{(3)} = I(S_i; S_j) = \sum_{y_i \in \{0,1\}} \sum_{y_j \in \{0,1\}} p(y_i, y_j) \log_2 \frac{p(y_i, y_j)}{p(y_i)p(y_j)}$$
    Where $p(y_i, y_j)$ represents joint probability distributions of the active states in the window.

*   **Channel 4: Co-occurrence Frequency (Raw probability):**
    $$A_{i,j}^{(4)} = \frac{1}{W} \sum_{t \in w} (S_{i,t} \cdot S_{j,t})$$

### 3. Graph Transformer Autoencoder (GTAE) Architecture

The GTAE compresses and reconstructs the node feature tensor $\mathbf{X} \in \mathbb{R}^{B \times N \times W \times F}$ and adjacency matrix $\mathbf{A} \in \mathbb{R}^{B \times C \times N \times N}$ (where $B$ is batch size).

#### Encoder GAT Layer
For layer $l$, the node embeddings $\mathbf{h}_i^{(l)}$ are updated by aggregating neighbor embeddings weighted by attention coefficients $\alpha_{i,j}$:

$$\mathbf{h}_i^{(l+1)} = \sigma \left( \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}^{(l)} \mathbf{W}^{(l)} \mathbf{h}_j^{(l)} \right)$$

$$\alpha_{i,j}^{(l)} = \frac{\exp \left( \text{LeakyReLU} \left( \mathbf{a}^{(l)T} [ \mathbf{W}^{(l)} \mathbf{h}_i^{(l)} \,\|\, \mathbf{W}^{(l)} \mathbf{h}_j^{(l)} ] \right) \right)}{\sum_{k \in \mathcal{N}(i)} \exp \left( \text{LeakyReLU} \left( \mathbf{a}^{(l)T} [ \mathbf{W}^{(l)} \mathbf{h}_i^{(l)} \,\|\, \mathbf{W}^{(l)} \mathbf{h}_k^{(l)} ] \right) \right)}$$

Where:
*   $\mathbf{h}_i^{(l)}$: Embedding of node $i$ at layer $l$.
*   $\mathbf{W}^{(l)}$: Learnable linear weight transformation matrix.
*   $\mathbf{a}^{(l)}$: Attention parameter vector.
*   $\mathcal{N}(i)$: Neighborhood set of node $i$ (including self-loops).
*   $\| \mathrel{\cdot} \|$: Vector concatenation operator.

#### Transformer Self-Attention Layer
Following GAT embeddings, a Transformer layer aggregates global context:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

Where Query $Q$, Key $K$, and Value $V$ matrices are generated from the node embedding projections:

$$Q = \mathbf{H}\mathbf{W}_Q, \quad K = \mathbf{H}\mathbf{W}_K, \quad V = \mathbf{H}\mathbf{W}_V$$

#### Joint Optimization Loss Function
The network is optimized using a weighted multi-task loss composed of Mean Squared Error (MSE) for node feature reconstruction and Binary Cross-Entropy (BCE) for adjacency structure reconstruction:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{X} + \beta \cdot \mathcal{L}_{A}$$

$$\mathcal{L}_{X} = \frac{1}{B \cdot N \cdot W \cdot F} \sum_{b=1}^B \sum_{n=1}^N \sum_{w=1}^W \sum_{f=1}^F \left(X_{b,n,w,f} - \hat{X}_{b,n,w,f}\right)^2$$

$$\mathcal{L}_{A} = -\frac{1}{B \cdot C \cdot N \cdot N} \sum_{b=1}^B \sum_{c=1}^C \sum_{i=1}^N \sum_{j=1}^N \left[ A_{b,c,i,j} \log\left(\hat{A}_{b,c,i,j}\right) + \left(1 - A_{b,c,i,j}\right) \log\left(1 - \hat{A}_{b,c,i,j}\right) \right]$$

Where:
*   $X_{b,n,w,f}$: Original feature value for batch $b$, node $n$, window step $w$, and feature index $f$.
*   $\hat{X}_{b,n,w,f}$: Reconstructed feature value.
*   $A_{b,c,i,j}$: True adjacency weight for channel $c$ between nodes $i$ and $j$.
*   $\hat{A}_{b,c,i,j}$: Reconstructed adjacency link.
*   $\beta = 0.2$: Regularization factor scaling the structure reconstruction loss.

---

## 🔍 Fault Simulation & Maintenance Analytics

### Progressive Wear-and-Tear Fault Models
To evaluate predictive maintenance capabilities, synthetic faults are gradually introduced from Week 12 to Week 20 of the timeseries:

*   **Fridge Compressor Gasket Leak:** Simulates thermal loss. The active duration is extended over time:
    $$P_{n,t}' = P_{n,t} \cdot \left(1.0 + \gamma_{leak} \cdot \max(0, w - 12)\right)$$
    (where $\gamma_{leak} = 0.02$, increasing power requirements by up to $16\%$).

*   **Washing Machine Motor Bearing Degradation:** Simulates mechanical wear, introducing random current micro-spikes during cycles:
    $$P_{n,t}' = P_{n,t} + \eta_{spike} \cdot \delta_t \cdot P_{\max}$$
    (where $\delta_t \sim \text{Bernoulli}(0.1)$ and $\eta_{spike} = 0.15 \cdot \frac{w - 12}{8}$).

### Health Index and Remaining Useful Life (RUL)
The anomaly score is calculated using the **Reconstruction Error Drift Ratio**:

$$\text{Drift Ratio}_n(w) = \frac{\text{MSE}_{recon, n}(w)}{\bar{\text{MSE}}_{normal, n}}$$

Where $\bar{\text{MSE}}_{normal, n}$ is the baseline reconstruction error for appliance $n$ trained under normal conditions. 

The **Health Index ($H$)** is computed exponentially:

$$H_n(w) = 100 \cdot e^{-\alpha \cdot \max\left(0, \text{Drift Ratio}_n(w) - 1.0\right)}$$

(where $\alpha = 0.1$).

#### RUL Estimation
For a rolling window of historical health values $\mathbf{H} = [H(w-K), \dots, H(w)]$, the pipeline fits an exponential decay regression model:

$$H(t) = H_0 \cdot e^{-\lambda t}$$

Where $H_0$ and $\lambda$ are estimated via least-squares regression on the log-transformed health values. The RUL is defined as the time step $t_{fail}$ when the projected health drops below the failure threshold ($50\%$):

$$t_{fail} = \frac{\ln(50) - \ln(H_0)}{-\lambda}$$

$$\text{RUL (Days)} = \max\left(5, (t_{fail} - t_{curr}) \cdot \text{stride\_days}\right)$$

---

## 💡 Explainable AI (XAI)
To provide diagnostic insights for operators, GTAE computes **Feature Saliency** maps. Saliency is calculated as the absolute gradient of the reconstruction loss with respect to the input features, isolating exactly which parameter caused the anomaly trigger:

$$\text{Saliency}_{n,w,f} = \left| \frac{\partial \mathcal{L}_{recon, n}}{\partial X_{n,w,f}} \right|$$

If the saliency gradient for the power feature is dominant, the system diagnoses a mechanical power surge. If the active duration feature gradient dominates, it diagnoses a duty-cycle control failure (e.g., thermostat failure).
