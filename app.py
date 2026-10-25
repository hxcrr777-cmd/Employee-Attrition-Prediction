import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

st.set_page_config(page_title="Employee Attrition Dashboard", page_icon="📊", layout="wide")

st.title("📊 Employee Attrition Prediction System")
st.caption("Explore workforce data, review model performance, and estimate attrition risk.")

DEFAULT_FILE = "employee_attrition_synthetic_1000.csv"

@st.cache_data
def load_csv(uploaded_file, local_path):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    try:
        return pd.read_csv(local_path)
    except Exception:
        return None

uploaded = st.sidebar.file_uploader("Upload employee dataset (CSV)", type=["csv"])
df = load_csv(uploaded, DEFAULT_FILE)

if df is None:
    st.info(
        f"Upload your CSV dataset in the sidebar, or place `{DEFAULT_FILE}` in the same folder as app.py."
    )
    st.stop()

if df.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

df.columns = [str(c).strip() for c in df.columns]
st.sidebar.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

# Choose target column, preferring common attrition labels.
target_candidates = [c for c in df.columns if c.lower().replace("_", " ").strip() in
                     ["attrition", "left", "employee attrition", "turnover", "target", "label"]]
default_target_index = df.columns.get_loc(target_candidates[0]) if target_candidates else len(df.columns) - 1
target = st.sidebar.selectbox("Target column (what to predict)", df.columns, index=default_target_index)

data = df.dropna(subset=[target]).copy()
if data[target].nunique(dropna=True) < 2:
    st.error("The selected target column must contain at least two different classes.")
    st.stop()

# Summary KPIs
n_employees = len(df)
n_features = len(df.columns) - 1
target_values = data[target].astype(str).str.strip().str.lower()
positive_labels = {"yes", "1", "true", "left", "attrition", "leave", "leaving"}
positive_mask = target_values.isin(positive_labels)
if positive_mask.any():
    observed_rate = 100 * positive_mask.mean()
else:
    observed_rate = 100 * (target_values == target_values.value_counts().index[0]).mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Employee Records", f"{n_employees:,}")
c2.metric("Features", f"{n_features:,}")
c3.metric("Target Classes", f"{data[target].nunique()}")
c4.metric("Most common target class", str(data[target].mode(dropna=True).iloc[0]))

st.divider()
overview_tab, model_tab, predict_tab, data_tab = st.tabs(
    ["📈 Workforce Overview", "🧠 Model Performance", "🔮 Predict Attrition", "🗂️ Dataset"]
)

with overview_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Target class distribution")
        counts = data[target].astype(str).value_counts()
        fig, ax = plt.subplots()
        counts.plot(kind="bar", ax=ax)
        ax.set_xlabel(target)
        ax.set_ylabel("Number of employees")
        ax.set_title("Distribution of target classes")
        plt.xticks(rotation=0)
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Attrition by department")
        dept_col = next((c for c in df.columns if c.lower().replace("_", " ").strip() in
                         ["department", "dept", "business unit"]), None)
        if dept_col:
            cross = pd.crosstab(data[dept_col].fillna("Unknown"), data[target].astype(str))
            fig, ax = plt.subplots()
            cross.plot(kind="bar", stacked=True, ax=ax)
            ax.set_xlabel(dept_col)
            ax.set_ylabel("Number of employees")
            ax.set_title("Target classes by department")
            plt.xticks(rotation=35, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No department column found. Add a column named Department to see this chart.")

    st.subheader("Numeric feature overview")
    numeric_cols = [c for c in data.select_dtypes(include=np.number).columns if c != target]
    if numeric_cols:
        chosen_num = st.selectbox("Choose a numeric feature", numeric_cols)
        fig, ax = plt.subplots()
        data[chosen_num].dropna().plot(kind="hist", bins=20, ax=ax)
        ax.set_xlabel(chosen_num)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of {chosen_num}")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No numeric features are available to plot.")

@st.cache_resource
def train_model(frame, target_col):
    model_df = frame.dropna(subset=[target_col]).copy()
    X = model_df.drop(columns=[target_col])
    y_raw = model_df[target_col].astype(str)

    # Remove obvious identifier columns to reduce accidental memorization.
    id_cols = [c for c in X.columns if c.lower().strip() in
               ["id", "employeeid", "employee_id", "employee number", "employeenumber"]]
    X = X.drop(columns=id_cols, errors="ignore")

    # Remove columns with no useful variation.
    constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    X = X.drop(columns=constant_cols, errors="ignore")

    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after removing identifiers/constants.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    if len(np.unique(y)) < 2:
        raise ValueError("The target column must contain at least two classes.")

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)
    classifier = RandomForestClassifier(
        n_estimators=250, random_state=42, class_weight="balanced"
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("classifier", classifier)])

    class_counts = pd.Series(y).value_counts()
    stratify = y if class_counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "f1": f1_score(y_test, predictions, average="weighted", zero_division=0),
        "y_test": y_test,
        "predictions": predictions,
        "class_names": encoder.classes_,
        "label_encoder": encoder,
        "feature_columns": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "dropped_columns": id_cols + constant_cols,
        "X": X
    }
    return pipeline, metrics

try:
    pipeline, metrics = train_model(data, target)
except Exception as e:
    st.error(f"Could not train the model: {e}")
    st.stop()

with model_tab:
    st.subheader("Random Forest classification results")
    st.caption("Metrics are calculated on a held-out test set (20%). Results depend on your dataset and target column.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    m2.metric("Precision (weighted)", f"{metrics['precision']:.1%}")
    m3.metric("Recall (weighted)", f"{metrics['recall']:.1%}")
    m4.metric("F1-score (weighted)", f"{metrics['f1']:.1%}")

    st.subheader("Confusion matrix")
    cm = confusion_matrix(metrics["y_test"], metrics["predictions"])
    labels = metrics["class_names"]
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    report = classification_report(
        metrics["y_test"], metrics["predictions"],
        labels=np.arange(len(labels)), target_names=[str(x) for x in labels],
        output_dict=True, zero_division=0
    )
    st.subheader("Classification report")
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

    st.caption("This is a baseline Random Forest model trained by the dashboard. Its score may differ from the models in your Colab notebook.")

with predict_tab:
    st.subheader("Estimate attrition for an employee")
    st.caption("Enter values for the features used by the model. This estimate is not a decision about an employee.")
    feature_cols = metrics["feature_columns"]
    input_values = {}
    X_source = metrics["X"]
    with st.form("prediction_form"):
        cols = st.columns(2)
        for i, col in enumerate(feature_cols):
            with cols[i % 2]:
                series = X_source[col]
                if col in metrics["numeric_features"]:
                    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
                    default_val = float(numeric_series.median()) if not numeric_series.empty else 0.0
                    input_values[col] = st.number_input(
                        col, value=default_val, key=f"input_{col}"
                    )
                else:
                    options = series.dropna().astype(str).value_counts().index.tolist()
                    if not options:
                        options = ["Unknown"]
                    input_values[col] = st.selectbox(col, options, key=f"input_{col}")
        submitted = st.form_submit_button("Predict Attrition", type="primary")

    if submitted:
        sample = pd.DataFrame([input_values], columns=feature_cols)
        pred_encoded = int(pipeline.predict(sample)[0])
        predicted_label = metrics["label_encoder"].inverse_transform([pred_encoded])[0]
        st.markdown("### Prediction")
        st.success(f"Predicted class: **{predicted_label}**")
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(sample)[0]
            prob_df = pd.DataFrame({
                "Class": metrics["class_names"],
                "Estimated probability": probabilities
            }).sort_values("Estimated probability", ascending=False)
            st.dataframe(prob_df.style.format({"Estimated probability": "{:.1%}"}), use_container_width=True)
        st.warning("Use predictions only as a prompt for supportive review. Do not make employment decisions solely from this model.")

with data_tab:
    st.subheader("Dataset preview")
    st.dataframe(df.head(100), use_container_width=True)
    st.write("**Missing values by column**")
    missing = df.isna().sum().sort_values(ascending=False)
    st.dataframe(missing[missing > 0].rename("Missing values").to_frame(), use_container_width=True)
    st.download_button(
        "Download dataset as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="employee_attrition_dataset.csv",
        mime="text/csv"
    )

st.sidebar.markdown("---")
st.sidebar.caption("Educational project • Verify model quality before real-world use.")
