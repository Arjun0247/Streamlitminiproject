import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="📊 Insights Explorer", layout="wide")
st.title("📊 Streamlit Insights Application")

st.sidebar.header("1. Upload CSV")
file = st.sidebar.file_uploader("Upload your dataset (CSV only)", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.success("Data loaded successfully!")
    
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    st.markdown(f"**Shape:** `{df.shape}` | **Columns:** `{list(df.columns)}`")
    
    with st.expander("ℹ️ Column Info"):
        st.dataframe(df.dtypes.astype(str).rename("Data Type"))

    st.sidebar.header("2. Data Cleaning")
    if df.isnull().values.any():
        st.warning("Missing values detected")
        st.write(df.isnull().sum())
        drop_na = st.sidebar.checkbox("Drop rows with missing values?")
        if drop_na:
            df.dropna(inplace=True)
            st.success("Missing rows dropped")
    else:
        st.success("No missing values detected")

    st.sidebar.header("3. Exploratory Analysis")
    st.subheader("📈 Distributions & Correlations")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if numeric_cols:
        with st.expander("📊 Histogram Explorer"):
            col = st.selectbox("Select numeric column", numeric_cols)
            fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("📉 Correlation Heatmap"):
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    if categorical_cols:
        with st.expander("📊 Category Counts"):
            col = st.selectbox("Select categorical column", categorical_cols)
            fig = px.bar(df[col].value_counts().reset_index(),
                     x='index', y='count',
                     labels={'index': col, 'count': 'Count'},
                     title=f"Value Counts for {col}")
        st.plotly_chart(fig, use_container_width=True)

    st.sidebar.header("4. Insights")
    st.subheader("💡 Key Insights")

    # Insight 1: Most correlated features
    st.markdown("**🔗 Most Correlated Feature Pair:**")
    cor_pairs = corr.abs().unstack().sort_values(ascending=False)
    cor_pairs = cor_pairs[cor_pairs < 1]
    top_pair = cor_pairs.idxmax()
    st.write(f"{top_pair[0]} and {top_pair[1]} — Correlation: {cor_pairs.max():.2f}")

    # Insight 2: Most missing data
    if df.isnull().sum().max() > 0:
        st.markdown("**🚫 Column with Most Missing Data:**")
        miss_col = df.isnull().sum().idxmax()
        st.write(f"{miss_col} — Missing: {df[miss_col].isnull().sum()} rows")

    # Insight 3: Feature Imbalance
    st.markdown("**⚖️ Most Imbalanced Category:**")
    if categorical_cols:
        cat_counts = {col: df[col].value_counts(normalize=True).max()
                      for col in categorical_cols}
        most_imbalanced = max(cat_counts, key=cat_counts.get)
        st.write(f"{most_imbalanced} — Dominance: {cat_counts[most_imbalanced]*100:.1f}%")

    # Insight 4: Outlier detection
    st.markdown("**🚨 Outlier Detection (IQR Method):**")
    outlier_counts = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
        outlier_counts[col] = outliers
    most_outliers = max(outlier_counts, key=outlier_counts.get)
    st.write(f"{most_outliers} — {outlier_counts[most_outliers]} potential outliers")

    # Insight 5: Time-based trend
    datetime_cols = df.select_dtypes(include=['datetime', 'object']).columns
    for col in datetime_cols:
        try:
            df[col] = pd.to_datetime(df[col])
            with st.expander("📅 Time Trend Insight"):
                time_col = col
                if numeric_cols:
                    y = numeric_cols[0]
                    df_time = df[[time_col, y]].dropna()
                    df_time = df_time.groupby(time_col)[y].mean().reset_index()
                    fig = px.line(df_time, x=time_col, y=y, title=f"{y} over time ({time_col})")
                    st.plotly_chart(fig, use_container_width=True)
                break
        except Exception:
            continue

else:
    st.info("Please upload a CSV file to get started.")
