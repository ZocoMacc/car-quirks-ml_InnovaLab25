from src.data.load import load_raw
from src.data.clean import clean_df
from src.features.build import add_features
from src.models.train import train_model
from src.utils.io import save_model

def main():
    df = load_raw("data/raw/cars.csv")
    df = clean_df(df)
    df = add_features(df)
    X, y = df.drop("quality", axis=1), df["quality"]
    model = train_model(X, y)
    save_model(model, "models/baseline.joblib")

if __name__ == "__main__":
    main()