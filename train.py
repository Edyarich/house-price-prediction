import argparse
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.append('./code/')

from data_processing import preprocess_train_data
from model import fit_model, predict_using_model, r2_score_custom, rmsle_custom


def main():
    parser = argparse.ArgumentParser(description="Process paths for train and test dataframes.")

    # Adding arguments
    parser.add_argument('train_df_path', type=str, help='Path to the training dataframe')
    parser.add_argument('output_path', type=str, help='Path to save model')

    # Parsing arguments
    args = parser.parse_args()

    # Now you can use args.train_df and args.test_df as the paths to your dataframes
    df_train = pd.read_csv(args.train_df_path)

    X_train, y_train, metadata = preprocess_train_data(df_train, return_metadata=True)
    model = fit_model(X_train, y_train)
    pred_train = predict_using_model(model, X_train)

    print("R^2 on training dataset:")
    print(r2_score_custom(y_train, pred_train))

    print("RMSLE on training dataset:")
    print(rmsle_custom(y_train, pred_train))

    with open(args.output_path, 'wb') as fd:
        pickle.dump((model, metadata), fd)


if __name__ == "__main__":
    main()
