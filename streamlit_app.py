import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from streamlit.components.v1 import html

st.set_page_config(page_title="Chocolate Sales Revenue Prediction", layout="centered")
st.title("Chocolate Sales Revenue Prediction")
st.caption("Predict expected sales amount based on product, country, sales person, shipment quantity, and time period")

@st.cache_data
def load_data():
    df = pd.read_csv("Chocolate Sales (2).csv")

    df["Amount"] = (
        df["Amount"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year

    df = df.dropna(subset=["Amount", "Boxes Shipped", "Month", "Year", "Sales Person", "Country", "Product"])
    df = df.drop_duplicates()
    return df

@st.cache_resource
def load_model():
    return joblib.load("chocolate_sales_model.pkl")

df = load_data()
model = load_model()

# =========================
# SIDEBAR CONTROLS
# =========================
with st.sidebar:
    st.header("Controls")

    show_data = st.toggle("Show dataset preview", False)

    st.subheader("Inputs")
    sales_person = st.selectbox("Sales Person", sorted(df["Sales Person"].unique()))
    country = st.selectbox("Country", sorted(df["Country"].unique()))
    product = st.selectbox("Product", sorted(df["Product"].unique()))

    boxes_shipped = st.number_input(
        "Boxes Shipped",
        min_value=int(df["Boxes Shipped"].min()),
        max_value=int(df["Boxes Shipped"].max()),
        value=int(df["Boxes Shipped"].median()),
        step=1
    )

    month = st.slider("Month", 1, 12, int(df["Month"].median()))
    year = st.slider(
        "Year",
        int(df["Year"].min()),
        int(df["Year"].max()),
        int(df["Year"].median())
    )

    predict = st.button("Predict Sales Amount")

# =========================
# MAIN AREA (for graphs later)
# =========================
st.subheader("Prediction")

if predict:
    input_df = pd.DataFrame([{
        "Sales Person": sales_person,
        "Country": country,
        "Product": product,
        "Boxes Shipped": boxes_shipped,
        "Month": month,
        "Year": year
    }])

    pred = model.predict(input_df)[0]
    st.success(f"Predicted Sales Amount: ${pred:,.2f}")

if show_data:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))


def render_chartjs_line(labels, values, title="Sales Trend"):
    labels_json = json.dumps(labels)
    values_json = json.dumps(values)

    chart_html = f"""
    <div style="width: 100%; height: 420px;">
      <canvas id="salesChart"></canvas>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
      const ctx = document.getElementById('salesChart').getContext('2d');

      const data = {{
        labels: {labels_json},
        datasets: [{{
          label: {json.dumps(title)},
          data: {values_json},
          tension: 0.25,
          pointRadius: 0,
          borderWidth: 2
        }}]
      }};

      const config = {{
        type: 'line',
        data,
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{
            legend: {{ display: true }},
            tooltip: {{
              callbacks: {{
                label: (ctx) => '$' + Number(ctx.parsed.y).toLocaleString()
              }}
            }}
          }},
          scales: {{
            y: {{
              ticks: {{
                callback: (v) => '$' + Number(v).toLocaleString()
              }}
            }}
          }}
        }}
      }};

      new Chart(ctx, config);
    </script>
    """
    html(chart_html, height=460)

# ---- Example usage with your df ----
# df must already contain Date (datetime) and Amount (numeric)

trend_monthly = (
    df.dropna(subset=["Date", "Amount"])
      .set_index("Date")
      .sort_index()
      .resample("MS")["Amount"]
      .sum()
      .reset_index()
)

labels = trend_monthly["Date"].dt.strftime("%Y-%m").tolist()
values = trend_monthly["Amount"].round(2).tolist()

st.subheader("Overall Sales Trend (Monthly)")
render_chartjs_line(labels, values, title="Total Sales Amount")