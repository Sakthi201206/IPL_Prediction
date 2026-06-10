from pathlib import Path
import argparse
import pickle
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

DATA_FILE = Path("matches.csv")
MODEL_FILE = Path("model.pkl")
ENCODER_FILE = Path("encoders.pkl")

FEATURE_COLUMNS = [
    "city",
    "match_type",
    "venue",
    "team1",
    "team2",
    "toss_winner",
    "toss_decision",
]

CATEGORICAL_COLUMNS = FEATURE_COLUMNS.copy()
TARGET_COLUMN = "team1_won"


def load_matches(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. Place matches.csv in the project root."
        )

    df = pd.read_csv(csv_path)
    df[TARGET_COLUMN] = (df["winner"] == df["team1"]).astype(int)
    df = df.dropna(subset=FEATURE_COLUMNS + ["winner", "team1"])
    return df


def prepare_features(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    X[CATEGORICAL_COLUMNS] = X[CATEGORICAL_COLUMNS].astype(str).fillna("Unknown")
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[CATEGORICAL_COLUMNS] = encoder.fit_transform(X[CATEGORICAL_COLUMNS])

    return X, y, encoder


def train_and_save_model():
    df = load_matches(DATA_FILE)
    X, y, encoder = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained. Test accuracy: {accuracy:.4f}")

    with open(MODEL_FILE, "wb") as model_file:
        pickle.dump(model, model_file)

    with open(ENCODER_FILE, "wb") as encoder_file:
        pickle.dump(encoder, encoder_file)

    return model, encoder


def load_saved_model():
    if not MODEL_FILE.exists() or not ENCODER_FILE.exists():
        return None, None

    with open(MODEL_FILE, "rb") as model_file:
        model = pickle.load(model_file)

    with open(ENCODER_FILE, "rb") as encoder_file:
        encoder = pickle.load(encoder_file)

    return model, encoder


def prompt_for_input():
    print("Enter match details to predict whether team1 will win.")
    user_data = {}

    for col in FEATURE_COLUMNS:
        prompt = f"{col.replace('_', ' ').title()}: "
        value = input(prompt).strip()
        if value == "":
            print(f"Value required for {col}.")
            sys.exit(1)
        user_data[col] = value

    return pd.DataFrame([user_data])


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict whether team1 will win based on match metadata."  
    )
    for col in FEATURE_COLUMNS:
        parser.add_argument(
            f"--{col}",
            type=str,
            help=f"Value for {col.replace('_', ' ')}",
        )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train and save the model, then exit without predicting.",
    )
    return parser


def prompt_for_input(existing: dict | None = None):
    print("Enter match details to predict whether team1 will win.")
    user_data = existing.copy() if existing else {}

    for col in FEATURE_COLUMNS:
        if col in user_data and user_data[col] is not None:
            continue

        prompt = f"{col.replace('_', ' ').title()}: "
        while True:
            value = input(prompt).strip()
            if value:
                user_data[col] = value
                break
            print(f"Value required for {col}. Please try again.")

    return pd.DataFrame([user_data])


def encode_user_input(user_df: pd.DataFrame, encoder: OrdinalEncoder):
    user_df = user_df[FEATURE_COLUMNS].astype(str).fillna("Unknown")
    user_df[CATEGORICAL_COLUMNS] = encoder.transform(user_df[CATEGORICAL_COLUMNS])
    return user_df


def main():
    parser = get_parser()
    args = parser.parse_args()

    model, encoder = load_saved_model()
    if model is None or encoder is None:
        model, encoder = train_and_save_model()
        if args.train_only:
            return

    if args.train_only:
        print("Model is already trained and saved.")
        return

    user_data = {
        col: getattr(args, col)
        for col in FEATURE_COLUMNS
        if getattr(args, col) is not None
    }

    user_df = prompt_for_input(user_data if user_data else None)
    encoded_df = encode_user_input(user_df, encoder)

    prediction = model.predict(encoded_df)[0]
    result = "Team1 wins" if prediction == 1 else "Team2 wins"
    print(f"Prediction: {result}")


if __name__ == "__main__":
    main()
