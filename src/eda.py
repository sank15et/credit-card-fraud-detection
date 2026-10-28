
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# 1. Project configuration
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "creditcard.csv"
REPORT_DIR = ROOT_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def main():
    # -----------------------------
    # 2. Load dataset
    # -----------------------------
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Download creditcard.csv and put it inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)

    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column.")

    if not set(df["Class"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Class must contain only 0 (legitimate) and 1 (fraud).")

    print("\n" + "=" * 55)
    print("CREDIT CARD FRAUD DETECTION - EDA")
    print("=" * 55)

    print(f"\nDataset shape: {df.shape}")
    print(f"Total transactions: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    print("\nFirst five rows:")
    print(df.head().to_string())

    # -----------------------------
    # 3. Data quality analysis
    # -----------------------------
    print("\n--- Data types ---")
    print(df.dtypes.to_string())

    missing = df.isnull().sum().sort_values(ascending=False)
    print("\n--- Missing values ---")
    print(missing[missing > 0].to_string() if missing.any() else "None")

    duplicate_count = int(df.duplicated().sum())
    print(f"\nDuplicate rows: {duplicate_count:,}")

    if df["Class"].isna().any():
        raise ValueError("Target column 'Class' contains missing values.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df.isnull().all(axis=1).any():
        print("Warning: Some rows contain only missing values.")

    # -----------------------------
    # 4. Class imbalance analysis
    # -----------------------------
    counts = df["Class"].value_counts().reindex([0, 1], fill_value=0)
    percentages = counts / len(df) * 100

    print("\n--- Transaction class distribution ---")
    print(f"Legitimate transactions: {counts[0]:,} ({percentages[0]:.4f}%)")
    print(f"Fraudulent transactions: {counts[1]:,} ({percentages[1]:.4f}%)")

    if counts[1] > 0:
        print(f"Legitimate-to-fraud ratio: {counts[0] / counts[1]:.2f}:1")

    # -----------------------------
    # 5. Amount statistics
    # -----------------------------
    if "Amount" in df.columns:
        print("\n--- Transaction amount statistics ---")
        print(df.groupby("Class")["Amount"].describe().round(2).to_string())

    # -----------------------------
    # 6. Chart 1: Class distribution
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        x=["Legitimate", "Fraud"],
        y=[counts[0], counts[1]],
        ax=ax,
    )

    ax.set_title("Legitimate vs Fraudulent Transactions")
    ax.set_xlabel("Transaction type")
    ax.set_ylabel("Number of transactions")

    for index, count in enumerate([counts[0], counts[1]]):
        ax.text(index, count, f"{count:,}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(REPORT_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)

    # -----------------------------
    # 7. Chart 2: Amount distribution
    # -----------------------------
    if "Amount" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))

        # Limit the display to the 99th percentile to make
        # the main distribution easier to inspect.
        amount_limit = df["Amount"].quantile(0.99)
        plot_df = df[df["Amount"] <= amount_limit]

        sns.histplot(
            data=plot_df,
            x="Amount",
            hue="Class",
            bins=50,
            element="step",
            stat="count",
            common_norm=False,
            ax=ax,
        )

        ax.set_title("Transaction Amount Distribution (Up to 99th Percentile)")
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "amount_distribution.png", dpi=150)
        plt.close(fig)

        # -----------------------------
        # 8. Chart 3: Amount by class
        # -----------------------------
        fig, ax = plt.subplots(figsize=(8, 5))

        sns.boxplot(data=df, x="Class", y="Amount", ax=ax)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_title("Transaction Amount by Class")
        ax.set_xlabel("Class: 0 = Legitimate, 1 = Fraud")
        ax.set_ylabel("Amount (symmetric log scale)")

        fig.tight_layout()
        fig.savefig(REPORT_DIR / "amount_by_class.png", dpi=150)
        plt.close(fig)

    # -----------------------------
    # 9. Chart 4: Transaction time
    # -----------------------------
    if "Time" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))

        sns.histplot(
            data=df,
            x="Time",
            hue="Class",
            bins=50,
            element="step",
            stat="count",
            common_norm=False,
            ax=ax,
        )

        ax.set_title("Transaction Time Distribution")
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "transaction_time.png", dpi=150)
        plt.close(fig)

    # -----------------------------
    # 10. Chart 5: Correlation heatmap
    # -----------------------------
    numeric_df = df.select_dtypes(include=np.number)
    correlations = numeric_df.corr()["Class"].drop("Class", errors="ignore")

    if not correlations.empty:
        top_features = correlations.abs().nlargest(10).index.tolist()
        heatmap_columns = top_features + ["Class"]

        fig, ax = plt.subplots(figsize=(11, 8))

        sns.heatmap(
            numeric_df[heatmap_columns].corr(),
            cmap="coolwarm",
            center=0,
            annot=False,
            ax=ax,
        )

        ax.set_title("Correlation Heatmap: Top Features Related to Class")
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "correlation_heatmap.png", dpi=150)
        plt.close(fig)

        print("\n--- Features most correlated with Class ---")
        print(correlations.reindex(
            correlations.abs().sort_values(ascending=False).index
        ).head(10).round(4).to_string())

    # -----------------------------
    # 11. Save summary report
    # -----------------------------
    summary = pd.DataFrame({
        "metric": [
            "total_transactions",
            "total_columns",
            "legitimate_transactions",
            "fraudulent_transactions",
            "legitimate_percentage",
            "fraud_percentage",
            "missing_values",
            "duplicate_rows",
        ],
        "value": [
            len(df),
            len(df.columns),
            int(counts[0]),
            int(counts[1]),
            round(float(percentages[0]), 6),
            round(float(percentages[1]), 6),
            int(df.isnull().sum().sum()),
            duplicate_count,
        ],
    })

    summary.to_csv(REPORT_DIR / "eda_summary.csv", index=False)

    # Save a feature-level missing-value report.
    missing_report = missing.rename("missing_count").reset_index()
    missing_report.columns = ["column", "missing_count"]
    missing_report.to_csv(
        REPORT_DIR / "missing_values.csv",
        index=False,
    )

    print("\n--- Generated reports ---")
    for report in sorted(REPORT_DIR.iterdir()):
        print(f"  {report.name}")

    print("\nEDA completed successfully!")


if __name__ == "__main__":
    main()
