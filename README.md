# 🛍️ Customer Segmentation App

A web-based dashboard built with Streamlit and Scikit-learn to perform K-Means clustering on mall customer data. This app allows a user to explore existing customer segments and predict the segment for a new customer based on their income and spending score.

This project was built as a portfolio piece for technical placements.



## ✨ Features

* **Exploratory Data Analysis (EDA):** Interactive plots (histograms, scatter plots) to explore the customer dataset.
* **K-Means Clustering:** A 2D visualization of the 5 customer segments.
* **Live Prediction:** Sliders to input a new customer's 'Annual Income' and 'Spending Score'.
* **Persona Analysis:** The app predicts the customer's cluster and provides a "Persona" (e.g., "Target Customer," "Careful Spender") and a recommended marketing strategy.

## 🚀 Live Demo

**[Link to your deployed Streamlit app will go here]**

## 🛠️ Tech Stack

* **Python:** The core programming language.
* **Streamlit:** For building the interactive web app.
* **Pandas:** For data manipulation and analysis.
* **Scikit-learn:** For K-Means clustering and data scaling.
* **Matplotlib & Seaborn:** For data visualization.
* **Jupyter:** For the initial model development and experimentation.

## 🏃 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/customer-segmentation-app.git](https://github.com/YourUsername/customer-segmentation-app.git)
    cd customer-segmentation-app
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```