# Loan Approval Optimization using Deep Learning and Offline Reinforcement Learning

This project explores two modeling techniques to optimize loan approval decisions using LendingClub data:
- A **supervised deep learning classifier (MLP)** to predict loan default risk.
- An **offline reinforcement learning agent (DQN)** to learn a policy that maximizes financial return.

---

## Repository Contents
- `eda + supervised_model.ipynb`: Contains data preprocessing, feature engineering, and training of the MLP classifier.
- `eda + offline_reinforcement.ipynb`: Contains environment setup, reward engineering, and training of the RL agent using DQN.
- `requirements.txt`: Python dependencies for running the notebooks.
- `mlp_best1.pt`: Saved weights of the trained MLP model.
- `cleaned_loan_data_with_target.csv`: Preprocessed dataset used for both models.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone [https://github.com/Medha-glitch/LoanApprovalProject.git](https://github.com/Medha-glitch/LoanApprovalProject.git)
cd LoanApprovalProject
```

### 2. Create a virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

# How to Run

## Run the Deep Learning Model

1. Open **`eda + supervised_model.ipynb`** in **Jupyter Notebook** or **VS Code**.  
2. Run all cells to:
   - Preprocess the dataset  
   - Train the **MLP (Multi-Layer Perceptron)** classifier  
   - Evaluate performance metrics  
3. The notebook will output:
   - **AUC**
   - **F1-score**
   - Visualizations: **ROC curve** and **Confusion Matrix**

---

##  Run the RL Agent

1. Open **`eda + offline_reinforcement.ipynb`**.  
2. Run all cells to:
   - Set up the **offline reinforcement learning environment**
   - Define **rewards**
   - Train the **DQN (Deep Q-Network)** agent  
3. The notebook will output:
   - Estimated **policy value**
   - **Approval / Denial statistics**

---

## Results Summary

| Model | Metric | Value |
|:------|:--------|:------:|
| **MLP Classifier** | AUC | **0.9990** |
| **MLP Classifier** | F1-Score | **0.9891** |
| **RL Agent (DQN)** | Estimated Policy Value | **0.8235** |

---

## Notes

- **EDA** and cleaning are integrated within each notebook.  
- The **RL agent** uses **synthetic denied loans** to simulate a complete action space for offline learning.  
- All models are trained on the **cleaned LendingClub dataset**.  

---

## Contact

For questions, feedback, or collaboration:  
Email — *medha.sharma2901@gmail.com*  
GitHub — [https://github.com/Medha-glitch/LoanApprovalProject](https://github.com/Medha-glitch/LoanApprovalProject)

---

## `requirements.txt`

> Copy and paste the following into a file named **`requirements.txt`**:

```text
pandas
numpy
scikit-learn
matplotlib
torch
d3rlpy
```
---
