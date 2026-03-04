from pathlib import Path
import joblib
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay


def main():
    # Ensure outputs folder exists
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    # Load Iris dataset (no external data files needed)
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model (Decision Tree like your notebook)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save trained model
    joblib.dump(model, outputs_dir / "model.joblib")

    # Save confusion matrix as PNG
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.tight_layout()
    plt.savefig(outputs_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    print("Saved outputs:")
    print(outputs_dir / "model.joblib")
    print(outputs_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()