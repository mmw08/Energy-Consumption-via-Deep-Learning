# Energy Consumption Prediction with GRU and LSTM Models

## Introduction

This project aims to provide an **hourly energy consumption prediction service** using the **PJM Interconnection LLC Energy Consumption Dataset**. By leveraging advanced time-series models like **GRU (Gated Recurrent Unit)** and **LSTM (Long Short-Term Memory)** networks, implemented using the **PyTorch** framework, the project strives to achieve high prediction accuracy for energy usage trends.

In this project, we:

- Utilize GRU and LSTM models for time-series prediction tasks.
- Aim to predict the next hour's energy usage based on historical data.
- Follow a structured approach including feature selection, data preprocessing, model definition, training, and evaluation.
- Compare the performance of GRU and LSTM models using metrics like **sMAPE (Symmetric Mean Absolute Percentage Error)**.

The ultimate goal is to build robust models capable of capturing patterns and cyclical trends in energy consumption data.

---

## Dataset

The dataset contains hourly power consumption data across different regions in the United States, provided by PJM Interconnection LLC. It is available on Kaggle: [Hourly Energy Consumption Dataset](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption).

### Key Details:

- **Source**: Kaggle
- **Files**: 12 `.csv` files
- **Frequency**: Hourly data
- **Features**: Energy consumption trends across various regions

This dataset serves as the foundation for training and testing our GRU and LSTM models.

---

## Code Structure

The project comprises the following files and scripts:

1. **Notebook**: The main code is executed in a Jupyter Notebook, which serves as the central hub for running the project.

2. **Python Scripts**:

   - `utils.py`: Contains utility functions, such as the computation of sMAPE and other helper functions.
   - `training.py`: Implements the training pipeline for both GRU and LSTM models.
   - `lstm_model.py`: Defines the `LSTMNet` class, which represents a basic LSTM network architecture in PyTorch. This class can be adapted for various sequence prediction tasks.
   - `gru_model.py`: Defines the `GRUNet` class, which represents a GRU-based network architecture for similar tasks.

These scripts are modular, ensuring easy maintenance and scalability.

---

## Results

The results demonstrate that both GRU and LSTM models are capable of predicting energy consumption trends effectively:

- **Model Comparison**:

  - The GRU model exhibited slightly better performance in terms of **sMAPE**, but the difference is not statistically significant.
  - Both models successfully captured patterns and cyclical trends in the data.

- **Observations**:

  - The predictions closely follow the actual energy consumption values in the test set.
  - Delays in predicting abrupt changes, such as drops in consumption, highlight areas for potential improvement.

- **Implications**:

  - Energy consumption data is highly patterned, making it suitable for these types of models.
  - Other time-series tasks, such as stock price prediction, may present more challenges due to less predictable data.

### Model Saving:

The trained models for both GRU and LSTM are saved as `.pt` files. These files can be loaded for further testing or deployment.

---

By combining robust models and structured workflows, this project provides an effective solution for energy consumption prediction, enabling better decision-making and potential cost and CO2 savings.

