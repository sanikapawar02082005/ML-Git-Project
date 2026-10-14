
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer

# Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "students.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "processed_students.csv"

def run_pipeline():
    print("Starting End-to-End Data Pipeline")

    # 1. Data Collection
    df = pd.read_csv(DATA_FILE)
    print("Data collected:", df.shape)

    # 2. Data Preprocessing
    df = df.drop_duplicates()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Study_Hours"] = pd.to_numeric(
        df["Study_Hours"], errors="coerce"
    )
    df["Attendance"] = pd.to_numeric(
        df["Attendance"], errors="coerce"
    )
    df["Marks"] = pd.to_numeric(
        df["Marks"], errors="coerce"
    )

    # Fill missing numeric values with column medians
    numeric_columns = [
        "Age", "Study_Hours", "Attendance", "Marks"
    ]
    imputer = SimpleImputer(strategy="median")
    df[numeric_columns] = imputer.fit_transform(
        df[numeric_columns]
    )
    print("Preprocessing completed")

    # 3. Feature Engineering
    df["Study_Efficiency"] = (
        df["Marks"] / df["Study_Hours"].clip(lower=1)
    ).round(2)
    print("Feature engineering completed")

    # 4. Data Validation
    required_columns = [
        "Student_ID", "Age", "Study_Hours",
        "Attendance", "Marks", "Study_Efficiency"
    ]

    assert all(
        col in df.columns for col in required_columns
    ), "Required columns are missing"

    assert df["Student_ID"].is_unique, (
        "Student IDs must be unique"
    )
    assert df[required_columns].notna().all().all(), (
        "Missing values detected"
    )
    assert df["Attendance"].between(0, 100).all(), (
        "Attendance must be between 0 and 100"
    )
    assert df["Marks"].between(0, 100).all(), (
        "Marks must be between 0 and 100"
    )
    print("Data validation successful")

    # 5. Save processed data
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print("Processed data saved:", OUTPUT_FILE)

    # 6. Pipeline summary
    print("Final dataset shape:", df.shape)
    print("End-to-End Pipeline completed successfully")

if __name__ == "__main__":
    run_pipeline()
