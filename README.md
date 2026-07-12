# Smart Ride AI Dashboard 🚕

An interactive Machine Learning web application and operational forecasting tool for smart mobility and taxi fleet management. This project utilizes historical ride data and an XGBoost prediction engine to forecast taxi demand, recommend dynamic surge pricing, and intelligently allocate vehicle fleet mixes across different zones in Dubai.

## Features

- **Demand Forecasting:** Predicts exact outbound ride demand for 20 distinct zones across Dubai (e.g., Downtown Dubai, JLT, Business Bay) using an XGBRegressor model.
- **Dynamic Surge Recommendations:** Automatically suggests surge pricing multipliers based on anticipated demand tiers.
- **Fleet Allocation Engine:** Intelligently splits the required fleet into Standard Sedans, Premium SUVs, and Luxury VIP vehicles based on the predicted volume.
- **Interactive 3D Map (PyDeck):** Visualizes zone locations with a fully interactive map that smoothly transitions when a new zone is selected. 
- **7-Day Trend Analysis:** Projects demand for the upcoming week at the specified hour to assist in long-term fleet planning.
- **Beautiful UI:** A modern, glassmorphic UI built with Streamlit and custom CSS for smooth animations and hover interactions.

## Project Structure

- `streamlit_app.py`: The main Streamlit web application providing the interactive dashboard.
- `Smart Mobility.py`: A command-line script for training the model, evaluating performance (R2 Score), and running interactive terminal-based predictions.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/smart-mobility.git
   cd smart-mobility
   ```

2. **Install dependencies:**
   Make sure you have Python installed, then install the required packages:
   ```bash
   pip install pandas numpy scikit-learn xgboost streamlit pydeck plotly
   ```

3. **Run the Streamlit Dashboard:**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Run the Command Line Interface (CLI):**
   If you prefer to run predictions via the terminal:
   ```bash
   python "Smart Mobility.py"
   ```

## Technologies Used
- **Python** for core logic
- **XGBoost & Scikit-Learn** for Machine Learning
- **Streamlit** for the web application UI
- **PyDeck & Plotly** for interactive data visualization
- **Pandas & NumPy** for data manipulation

## Preview
*(Add a screenshot of your beautiful dashboard here!)*

---
*Developed as part of an ML Portfolio / Internship project.*
