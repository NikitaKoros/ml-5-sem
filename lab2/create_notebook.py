#!/usr/bin/env python3
import json

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [text]}

cells = [
    md("# Лабораторная работа 2: Ансамблевые методы и бустинг\n\n**Цель**: Предсказать выдачу кредита\n\n**Метрика**: ROC-AUC >= 0.75"),
    
    md("## 1. Импорт библиотек"),
    code("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier\nfrom sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve\nimport lightgbm as lgb\nimport xgboost as xgb\nimport catboost as cb\nimport optuna\nimport warnings\nwarnings.filterwarnings('ignore')\nnp.random.seed(42)"),
    
    md("## 2. Загрузка данных"),
    code("train_df = pd.read_csv('datasets/train.csv')\ntest_df = pd.read_csv('datasets/test.csv')\nprint(f'Train: {train_df.shape}, Test: {test_df.shape}')\ntrain_df.head()"),
]

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('lab2.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Notebook created!")
