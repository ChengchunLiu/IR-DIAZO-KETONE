# IR_DIAZO_KETONE

This repository corresponds to the [Infrared Spectra Prediction for Diazo Groups Utilizing a Machine Learning Approach with Structural Attention Mechanism](https://arxiv.org/abs/2402.03112).

## Python Version

This project requires Python (Version 3.11.4).

## Core Libraries

- **RDKit (Version 2023.9.5)**: RDKit is a collection of cheminformatics and machine learning tools that provide functionalities for handling chemical informatics tasks such as molecule generation, molecular descriptors, and visualization.

- **scikit-learn (Version 1.4.1)**: A robust library for machine learning in Python, providing simple and efficient tools for data mining and data analysis. It is built on NumPy, SciPy, and matplotlib.

- **matplotlib (Version 3.8.4)**: A comprehensive library for creating static, animated, and interactive visualizations in Python. It is highly customizable and used for generating publication-quality plots.

- **SHAP (SHapley Additive exPlanations) (Version 0.45.0)**: A library that provides interpretability for machine learning models by quantifying the contribution of each feature to the model's predictions using Shapley values.

- **XGBoost (Version 2.0.3)**: An optimized gradient boosting library designed to be highly efficient, flexible, and portable. It implements machine learning algorithms under the Gradient Boosting framework.

- **CatBoost (Version 1.2.3)**: A gradient boosting library on decision trees, providing fast and accurate solutions for both classification and regression problems. It is particularly effective with categorical features.

- **LightGBM (Version 4.3.0)**: A gradient boosting framework that uses tree-based learning algorithms, known for its efficiency and speed. It is designed to be distributed and efficient with large-scale data.

## Installation

To install these libraries, use the following pip commands:

```bash
pip install rdkit==2023.9.5
pip install scikit-learn==1.4.1
pip install matplotlib==3.8.4
pip install shap==0.45.0
pip install xgboost==2.0.3
pip install catboost==1.2.3
pip install lightgbm==4.3.0
