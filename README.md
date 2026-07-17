# Graph Transformer-based Predictive Maintenance System

An end-to-end spatiotemporal Graph Deep Learning platform designed to monitor home appliance degradation, simulate gradual wear-and-tear failures, forecast Remaining Useful Life (RUL), identify failures via Root Cause Analysis (RCA), and render an interactive web control dashboard. The platform utilizes a **Graph Transformer Autoencoder (GTAE)** to process multi-appliance behavioral interactions.

---

## 📖 Project Overview

Modern smart homes contain multiple electrical appliances whose operating cycles are often co-dependent or correlated (e.g., washing machines followed by tumble dryers, or water heaters activating during morning routines). Traditional anomaly detection models treat each appliance as an isolated timeseries, ignoring these relational dynamics. 

This project solves this limitation by framing a household's appliances as a **dynamic behavioral graph**. By analyzing both temporal energy fluctuations (node features) and co-activation patterns (edge connectivities), the Graph Transformer Autoencoder (GTAE) models the normal structural and features baseline of a household. When an appliance begins to degrade, its power consumption behavior or duty cycle drifts, yielding elevated reconstruction errors that are utilized to flag anomalies, predict severity, and estimate RUL.

---

## 📂 Project Directory Structure

```text
├── 1_raw_data/                        # Raw REFIT smart home datasets (Aggregate & Appliance columns)
│   ├── CLEAN_House1.csv               # Household 1 raw power readings (CSV format)
│   ├── CLEAN_House2.csv ...           # Household 2-21 raw power readings
│   └── CLEAN_House19_Test_Sample.csv  # 1,000-row lightweight sample of House 19 optimized for fast web dashboard upload testing
├── 3_processed_outputs/               # Compiled datasets, model weights, and reports
│   ├── Anomaly_Report_PM.json         # Master database of all processed households
│   ├── PM_Report_House_[ID].json      # Multi-metric details for individual household
│   ├── PM_Report_House_[ID].html      # Styled, self-contained interactive audit report
│   ├── House_[ID]_Processed.csv       # Standardized resampled timeseries per house
│   ├── House_[ID]_Dynamic_Graphs.pt   # Serialized PyTorch graph sequence tensors
│   └── House_[ID]_GTAE_PM.pth         # Saved GTAE model state-dict weights
├── benchmarks/                        # Benchmark and evaluation suite scripts
│   ├── evaluate_appliance_wise.py     # Appliance-wise detailed metrics (Accuracy, Precision, Recall, F1)
│   ├── evaluate_benchmark_cli.py      # Standard multi-model benchmark evaluation CLI (including ablation & seeds)
│   └── evaluate_per_house_split.py    # Per-house train/test split evaluator before/after fault
├── results/                           # Evaluation benchmark output reports
│   ├── ablation_results.json          # Results of GCN vs GTAE vs Hypergraph ablation studies
│   ├── appliance_wise_metrics.json    # Detailed appliance-wise evaluation metrics
│   ├── benchmark_results.json         # Overall model accuracy & lead time comparisons
│   ├── house_split_detailed.json      # Detailed appliance-wise train/test split metrics
│   ├── per_house_split_results.json   # General train/test split benchmark outcomes
│   └── seed_results.json              # Model robustness results across multiple random seeds
├── .venv/                             # Local Python virtual environment
├── .vscode/                           # IDE preferences (points to python.exe path)
├── compile_dashboard.py               # Compiles base dashboard templates
├── compile_pm_dashboard.py            # Compiles dashboard_pm_template.html to Predictive_Maintenance_Dashboard.html
├── dashboard_pm_template.html         # Front-end dark-glassmorphism HTML dashboard template
├── dashboard_template.html            # Base front-end HTML dashboard template
├── fault_injector.py                  # Standard fault injection utility
├── generate_synthetic_refit.py        # Generates synthetic REFIT dataset for testing
├── graph_builder.py                   # Graph construction utilities
├── graph_transformer.py               # GTAE, GCN, and LSTM PyTorch autoencoder architectures
├── observation.md                     # Markdown file detailing observations and notes
├── pm_analytics.py                    # Health Index, Severity, RUL forecasting, and RCA algorithms
├── pm_config.json                     # Threshold parameters and RUL degradation configurations
├── pm_dynamic_graph.py                # Multi-feature graph generator and Graph Drift calculator
├── pm_fault_injector.py               # Progressive wear-and-tear degradation simulator
├── pm_pipeline.py                     # Master execution runner (supports single/batch processing)
├── pm_report_exporter.py              # Self-contained JSON and HTML report generators
├── pm_xai.py                          # Backpropagation feature saliency and edge attention tracker
├── predict.py                         # Standalone prediction script
├── refit_processor.py                 # Timeseries resampling, gap mitigation, and signature-based NILM disaggregation
├── report_template.html               # Template for generating HTML reports
├── run_pipeline.py                    # Alternate execution runner for pipelines
├── server.py                          # Local HTTP backend server exposing routes for dashboard and reports
├── test_upload.py                     # Test script for upload functionality
├── train_eval.py                      # Training and evaluation loop script
├── verify_dashboard.py                # Script to verify dashboard functionalities
├── Logo.png                           # Primary UI header logo asset
├── aegis_logo.png                     # Secondary alternate logo asset
└── Predictive_Maintenance_Dashboard.html # Final compiled master dashboard file
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

*   **Testing Custom Uploads:** To test the custom CSV upload feature, click the **Upload Custom CSV** button in the sidebar and select the lightweight sample dataset: `1_raw_data/CLEAN_House19_Test_Sample.csv`. The backend will resample, extract dynamic graph features, and perform predictive diagnostics in less than 2 seconds.

### 4. Run Benchmark & Evaluation Suite
To perform model performance benchmarking, ablation studies, and multi-seed sensitivity analysis, run any of the following commands:

*   **Overall Benchmark CLI (Evaluates all models, outputs ablation results, and runs a multi-seed experiment):**
    ```bash
    .venv\Scripts\python benchmarks/evaluate_benchmark_cli.py --houses 19,20,21 --epochs 5
    ```
*   **Appliance-Wise Detailed Metrics (Evaluates Accuracy, Precision, Recall, and F1 for every single appliance model):**
    ```bash
    .venv\Scripts\python benchmarks/evaluate_appliance_wise.py --test_houses 19,20,21 --epochs 2
    ```
*   **Per-House Train/Test Split Evaluation (Evaluates Accuracy, Precision, Recall, and F1 under train/test splits before/after fault):**
    ```bash
    .venv\Scripts\python benchmarks/evaluate_per_house_split.py --houses 19,20,21 --epochs 3
    ```

---

## 📐 Complete Mathematical Framework

### Mathematical Formula and Parameter Table

Below is the consolidated mathematical formulary detailing the core computations, exact variables, parameter domains, and physical meanings:

| Component / Phase | Mathematical Formulation | Parameters & Variables | Physical Domain / Meaning |
| :--- | :--- | :--- | :--- |
| **Power Normalization** | $$X_{n, t}^{(0)} = \frac{P_{n, t}}{\max_{t} (P_n) + \epsilon}$$ | <ul><li>$P_{n, t} \ge 0$: Raw wattage value of appliance $n$</li><li>$\max_{t} (P_n)$: Baseline peak wattage</li><li>$\epsilon = 10^{-9}$: Safeguard divisor</li></ul> | Normalizes appliance power curves into the unit interval $[0, 1]$ to align low-power and high-power appliances. |
| **Appliance State Detection** | $$S_{n, t} = \begin{cases} 1 & \text{if } P_{n, t} \ge \theta_n \\ 0 & \text{otherwise} \end{cases}$$ | <ul><li>$\theta_n > 0$: Power activation threshold in Watts</li><li>$S_{n, t} \in \{0, 1\}$: Binary operational state</li></ul> | Establishes whether appliance $n$ is actively performing work at timestamp $t$. |
| **Spatiotemporal Co-activation** | $$A_{i,j}^{(1)} = \frac{\sum_{t \in w} (S_{i,t} S_{j,t})}{\sum_{t} S_{i,t} + \sum_{t} S_{j,t} - \sum_{t} (S_{i,t} S_{j,t})}$$ | <ul><li>$S_{i,t}, S_{j,t} \in \{0,1\}$: Appliance states</li><li>$A_{i,j}^{(1)} \in [0, 1]$: Jaccard overlap weight</li></ul> | Quantifies overlapping activation routines between appliances $i$ and $j$ in window $w$. |
| **Graph Transformer Gating** | $$\text{Attn}_{\text{gated}} = \frac{\mathbf{A} \odot \exp\left(\frac{Q K^T}{\sqrt{d_k}}\right)}{\sum \left[ \mathbf{A} \odot \exp\left(\frac{Q K^T}{\sqrt{d_k}}\right) \right]}$$ | <ul><li>$Q, K \in \mathbb{R}^{N \times d_k}$: Query and Key node projections</li><li>$\mathbf{A} \in \mathbb{R}^{N \times N}$: Adjacency gate matrix</li><li>$\odot$: Hadamard element-wise product</li></ul> | Gates multi-head attention scores by the co-occurrence weights to route representation updates along valid graph paths. |
| **Anomaly Scoring (Drift)** | $$\text{Drift Ratio}_n(w) = \frac{\text{MSE}_{recon, n}(w)}{\bar{\text{MSE}}_{normal, n}}$$ | <ul><li>$\text{MSE}_{recon, n}$: Current reconstruction error</li><li>$\bar{\text{MSE}}_{normal, n}$: Baseline trained error</li><li>$\text{Drift Ratio}_n \ge 0$: Ratio threshold</li></ul> | Computes the magnitude of signal deviation for appliance $n$ relative to its baseline healthy profile. |
| **Health Index Conversion** | $$H_n(w) = 100 \cdot e^{-\alpha \cdot \max\left(0, \text{Drift Ratio}_n(w) - 1.0\right)}$$ | <ul><li>$H_n(w) \in [0, 100]$: Percent health status</li><li>$\alpha = 0.1$: Exponential decay parameter</li></ul> | Translates signal reconstruction drift into an intuitive, bounded physical health indicator. |
| **Exponential RUL Decay** | $$H(t) = H_0 \cdot e^{-\lambda t}$$ | <ul><li>$H_0$: Health intercept parameter</li><li>$\lambda > 0$: Log-regression degradation rate</li><li>$t$: Cumulative sliding window index</li></ul> | Fits the degradation path of appliance $n$ to predict when its health will cross the failure boundary. |
| **Remaining Useful Life** | $$\text{RUL (Days)} = \frac{\ln(50) - \ln(H_0)}{-\lambda}$$ | <ul><li>$H_{\text{fail}} = 50\%$: Crucial degradation limit</li><li>$\text{RUL} \ge 5$: Estimated days before failure</li></ul> | Computes the time step intersection where health reaches the $50\%$ failure threshold. |

---

### Detailed Formulary Details

Below is the complete mathematical notation of the preprocessing, graph building, neural network architecture, and forecasting layers:

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
