<div align="center">

# 🚀 Federated NO3-CDR
### 🔒 Privacy-Preserving Cross-Domain Recommendation System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch">
  <img src="https://img.shields.io/badge/Federated-Learning-success?style=for-the-badge">
  <img src="https://img.shields.io/badge/Recommendation-System-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Privacy-Preserving-purple?style=for-the-badge">
</p>

<p align="center">
  <b>A modern implementation of SNO3-CDR with a novel Federated Learning architecture</b>
</p>

</div>

---

# 📌 Overview

This repository presents a **privacy-preserving Cross-Domain Recommendation System (CDR)** built using:

- 🎯 Matrix Factorization
- 🔄 Sinkhorn-based Soft User Alignment
- 🌐 Federated Learning
- 🔐 Privacy-Aware Distributed Training

The project is based on the paper:

> **SNO3-CDR: Soft Matching via Optimal Transport for Cross-Domain Recommendation**

and extends it with a novel architecture called:

# ✨ Federated NO3-CDR

Unlike traditional centralized recommender systems, this implementation allows multiple domains to collaboratively learn user preferences **without sharing raw user data**.

---

# 🧠 Main Idea

Traditional Cross-Domain Recommendation systems improve recommendations by transferring knowledge between domains:

- 🎬 Movies
- 📚 Books
- 🎵 Music
- 🛒 E-commerce

However, centralized training creates major privacy issues because all user interactions must be collected in one server.

This project solves that problem using **Federated Learning**.

---

# 🔥 Proposed Innovation

## ✅ Federated NO3-CDR

### Key Contributions

### 🔒 Privacy Preservation
Raw ratings and interaction histories never leave local clients.

### 🧠 Local Specialized Models
Each domain trains its own recommendation model independently.

### 🔄 Sinkhorn-Based Knowledge Transfer
Only latent user embeddings are exchanged and aligned using Optimal Transport.

### 📈 Better Performance
The federated architecture reduces negative transfer between heterogeneous domains and improves RMSE.

---

# 🏗️ Architecture

<div align="center">

```text
                +----------------------+
                |   Coordinator Server |
                | Sinkhorn Alignment   |
                +----------+-----------+
                           |
          -----------------------------------------
          |                                       |
+-------------------+             +-------------------+
| Amazon Client     |             | Book Client       |
| Local MF Model    |             | Local MF Model    |
| Private Data      |             | Private Data      |
+-------------------+             +-------------------+
```

</div>

---

# 📚 Datasets

This project uses two real-world datasets:

| Domain | Dataset |
|---|---|
| 🎬 Movies & TV | Amazon Reviews |
| 📚 Books | Book-Crossing |

---

# ⚙️ Preprocessing Pipeline

The following preprocessing steps are applied:

- ✅ Top active users selection
- ✅ 5-core filtering
- ✅ User/item indexing
- ✅ Dense interaction generation
- ✅ Train/Test split

---

# 🧪 Experimental Settings

| Parameter | Value |
|---|---|
| Embedding Dimension | 64 |
| Global Rounds | 40 |
| Local Epochs | 5 |
| Learning Rate | 0.005 |
| Gamma | 0.1 |
| Sinkhorn Epsilon | 0.1 |

---

# 📊 Results

| Method | Amazon RMSE | Books RMSE | Average RMSE |
|---|---|---|---|
| Centralized NO3-CDR | 0.8819 | 1.63 | 1.25 |
| 🚀 Federated NO3-CDR | Improved | Improved | **1.20** |

---

# 💡 Why Federated Learning Improved Performance

The centralized model struggles because:

- Amazon ratings use scale **1–5**
- Book-Crossing ratings use scale **1–10**

This creates optimization conflicts and negative transfer.

The federated approach solves this by:

✅ Keeping models domain-specific  
✅ Aligning only abstract preference distributions  
✅ Preserving local specialization  

---

# 🛠️ Tech Stack

<div align="center">

| Technology | Usage |
|---|---|
| 🐍 Python | Core Language |
| 🔥 PyTorch | Deep Learning |
| 📊 Pandas | Data Processing |
| 🔢 NumPy | Numerical Operations |
| 📉 Matplotlib | Visualization |
| 🤖 Federated Learning | Distributed Training |
| 🚚 Optimal Transport | Sinkhorn Alignment |

</div>

---

# 📂 Project Structure

```text
📦 Federated-NO3-CDR
 ┣ 📂 datasets
 ┃ ┣ 📄 Movies_and_TV.csv
 ┃ ┗ 📄 Ratings.csv
 ┣ 📄 federated_no3_cdr.py
 ┣ 📄 no3_cdr_implementation.py
 ┣ 📄 data_loader.py
 ┣ 📄 notebook.ipynb
 ┣ 📄 requirements.txt
 ┗ 📄 README.md
```

---

# 🚀 Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/federated-no3-cdr.git
cd federated-no3-cdr
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ How to Run

## 📌 Step 1 — Prepare Datasets

Place datasets inside the `datasets/` directory:

```text
datasets/
├── Movies_and_TV.csv
└── Ratings.csv
```

Update dataset paths inside notebook if needed.

---

## 📌 Step 2 — Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
notebook.ipynb
```

and run all cells sequentially.

---

# 🧩 Code Workflow

The notebook is organized into multiple sections:

| Section | Description |
|---|---|
| Import Libraries | Load required packages |
| Settings | Configure hyperparameters |
| Data Loading | Load datasets |
| Preprocessing | Filter & map users/items |
| Centralized Training | Original SNO3-CDR |
| Federated Training | Proposed method |
| Evaluation | RMSE & MAE comparison |

---

# 🔍 Core Components

---

## 📌 Matrix Factorization

Each domain uses an independent MF model:

```python
MatrixFactorization(
    n_users,
    n_items,
    n_factors=64
)
```

---

## 📌 Sinkhorn Alignment

User embedding distributions are aligned using:

- Optimal Transport
- Sinkhorn Distance

This enables soft cross-domain knowledge transfer.

---

## 📌 Federated Training

The federated process works as follows:

```text
1. Train local models
2. Extract user embeddings
3. Send embeddings to coordinator
4. Compute Sinkhorn alignment
5. Send gradients back
6. Update local models
```

---

# 📈 Example Training Logs

```text
Device: cuda
Global rounds: 40

Amazon Ready: 71,864 rows
Books Ready: 51,410 rows

Amazon (Fed) - RMSE: 0.85
Books  (Fed) - RMSE: 1.56
Average (Fed)- RMSE: 1.20
```

---


# 📖 Citation

If you use this repository in academic research, please cite:

```bibtex
@article{sno3cdr,
  title={SNO3-CDR: Soft Matching via Optimal Transport for Cross-Domain Recommendation},
  author={...},
  journal={...},
  year={...}
}
```

---

# ⭐ Support

If you found this project useful:

🌟 Star the repository  
🍴 Fork the project  
📢 Share it with others  

---

<div align="center">

# ❤️ Thank You

### Built with PyTorch + Federated Learning

</div>
