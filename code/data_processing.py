from typing import Tuple, Optional, Any
import numpy as np
import pandas as pd

import statsmodels.api as sm
from sklearn.impute import SimpleImputer


def transform_ecology_ctgry(eco: str) -> int:
    """
    Convert ecology category to number
    :param eco: ecology quality category
    :return: encoded category
    """
    if eco == 'no data':
        return 0
    elif eco == 'poor':
        return 1
    elif eco == 'satisfactory':
        return 2
    elif eco == 'good':
        return 3
    elif eco == 'excellent':
        return 4


def filter_by_correlation(dataset: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Remove highly-correlated features
    :param dataset: dataframe
    :param threshold: correlation cutoff
    :return:
    """
    col_corr = set()  # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname]  # deleting the column from the dataset

    return dataset


def preprocess_train_data(
        df: pd.DataFrame,
        trg_corr_cutoff: float = 0.05,
        mutual_corr_cutoff: float = 0.9,
        ohe_max_features: int = 5,
        lin_reg_pvalue_cutoff: float = 0.1,
        return_metadata: bool = False
) -> Tuple[pd.DataFrame, pd.Series, Optional[dict]]:
    """
    Training data processing pipeline
    :param df_train: train dataframe
    :param df_test: test dataframe
    :param trg_corr_cutoff: lower cutoff for correlation between feature and target to exclude features
    :param mutual_corr_cutoff: upper cutoff for correlation between feature_X and feature_Y to exclude features
    :param ohe_max_features: `unique_values <= ohe_max_features` -- use OHE, otherwise -- target encoding
    :param lin_reg_pvalue_cutoff: lower pvalue cutoff of F-test to exclude features
    :param return_metadata if True, return metadata for processing test dataframe
    :return: (X_train, y_train, metadata)
    """
    # Transforming the target variable and timestamp
    target = np.log(df['price_doc'] + 1)
    df['log_price_doc'] = target
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Dropping the original target and other specified columns
    columns_to_drop = [
        'ID_metro',
        'ID_railroad_station_walk',
        'ID_railroad_station_avto',
        'ID_big_road1',
        'ID_big_road2',
        'ID_railroad_terminal',
        'ID_bus_terminal',
        'id',
        'price_doc'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Calculate correlation between features and target
    numeric_columns_raw = list(df.loc[:, df.dtypes != np.object_].columns)
    numeric_columns_raw.remove('timestamp')
    corr_dct = dict()

    for col in numeric_columns_raw:
        notna_mask = df[col].notna()
        corr_dct[col] = np.corrcoef(df[col][notna_mask], target[notna_mask])[1, 0]

    # Filter features by correlation with target
    k_alive_features = sum(abs(value) >= trg_corr_cutoff for value in corr_dct.values())
    numeric_columns = [feature for feature, corr in
                       sorted(corr_dct.items(), key=lambda item: abs(item[1]), reverse=True)[:k_alive_features]]

    # Filter features by pairwise correlations
    df_num = filter_by_correlation(df[numeric_columns], mutual_corr_cutoff)
    numeric_columns_final = list(df_num.columns)

    # Convert to OHE
    df_cat = None
    categorical_columns = df.loc[:, df.dtypes == np.object_].columns
    cat_columns_ohe = []
    cat_columns_final_ohe = []
    cat_columns_trg_enc_dct = dict()

    for col in categorical_columns:
        if col in ('timestamp', 'ecology'):
            continue

        if df[col].nunique() <= ohe_max_features:
            one_hot = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df_cat = pd.concat((df_cat, one_hot), axis=1)

            cat_columns_ohe.append(col)
            cat_columns_final_ohe.extend(one_hot.columns)
        else:
            mean_target = df.groupby(col)['log_price_doc'].mean()
            df_cat[col] = df[col].map(mean_target)

            cat_columns_trg_enc_dct[col] = mean_target

    # Unite processed categorical and numerical features
    df_filt = pd.concat((df_cat, df_num), axis=1)
    df_filt['ecology_rng'] = df['ecology'].apply(transform_ecology_ctgry)

    # Fill the NaN values
    imp_median = SimpleImputer(strategy='median')
    df_filt_vals = imp_median.fit_transform(df_filt)
    df_filt_cols = df_filt.columns
    del df_filt
    df_filt = pd.DataFrame(df_filt_vals, columns=df_filt_cols)

    # Clip some features
    df_filt['full_sq'] = np.clip(df_filt['full_sq'], a_min=0, a_max=1000)
    df_filt['life_sq'] = np.clip(df_filt['life_sq'], a_min=0, a_max=1000)

    # Extract year from timestamp
    df_filt['year'] = df.timestamp.dt.year

    # Drop target
    df_filt = df_filt.drop('log_price_doc', axis=1)

    # Filter using Linear Regression
    linr_regr_mod = sm.OLS(
        endog=target,
        exog=sm.add_constant(df_filt)
    ).fit()
    bad_columns = linr_regr_mod.pvalues[np.flatnonzero(linr_regr_mod.pvalues > lin_reg_pvalue_cutoff)].index
    df_filt_new = df_filt.drop(columns=bad_columns)

    metadata = {
        'numeric_columns': numeric_columns_final,
        'categorical_ohe_columns': cat_columns_ohe,
        'categorical_final_ohe_columns': cat_columns_final_ohe,
        'categorical_trg_enc_values': cat_columns_trg_enc_dct,
        'median_imputer': imp_median,
        'bad_lin_reg_columns': bad_columns
    } if return_metadata else None

    return df_filt_new, target, metadata


def preprocess_test_data(
        df: pd.DataFrame,
        metadata: dict
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Data processing pipeline for test dataframe
    :param df: test dataframe
    :param metadata: dictionary with information from train dataframe
    :return: X_test, y_test
    """
    # Transforming the target variable and timestamp
    has_target = 'price_doc' in df.columns

    if has_target:
        target = np.log(df['price_doc'] + 1)
        df['log_price_doc'] = target
    else:
        df['log_price_doc'] = np.zeros(df.shape[0])

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Dropping the original target and other specified columns
    columns_to_drop = [
        'ID_metro',
        'ID_railroad_station_walk',
        'ID_railroad_station_avto',
        'ID_big_road1',
        'ID_big_road2',
        'ID_railroad_terminal',
        'ID_bus_terminal',
        'id',
        'price_doc'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    df_num = df[metadata['numeric_columns']]

    # Convert to OHE
    df_ohe = pd.get_dummies(
        df,
        prefix=metadata['categorical_ohe_columns'],
        columns=metadata['categorical_ohe_columns']
    )
    
    for col in metadata['categorical_final_ohe_columns']:
        if col not in df_ohe.columns:
            df_ohe[col] = 0
    
    df_ohe = df_ohe[metadata['categorical_final_ohe_columns']]

    trg_enc_columns = list(metadata['categorical_trg_enc_values'].keys())
    df_trg_enc = pd.DataFrame()

    for col in trg_enc_columns:
        df_trg_enc[col] = df[col].map(metadata['categorical_trg_enc_values'][col])

    # Unite processed categorical and numerical features
    df_filt = pd.concat((df_num, df_ohe, df_trg_enc), axis=1)
    df_filt['ecology_rng'] = df['ecology'].apply(transform_ecology_ctgry)

    # Fill the NaN values
    df_filt = df_filt[metadata['median_imputer'].get_feature_names_out()]
    df_filt_vals = metadata['median_imputer'].transform(df_filt)
    df_filt_cols = df_filt.columns
    del df_filt
    df_filt = pd.DataFrame(df_filt_vals, columns=df_filt_cols)

    # Clip some features
    df_filt['full_sq'] = np.clip(df_filt['full_sq'], a_min=0, a_max=1000)
    df_filt['life_sq'] = np.clip(df_filt['life_sq'], a_min=0, a_max=1000)

    # Extract year from timestamp
    df_filt['year'] = df.timestamp.dt.year

    # Drop target
    df_filt = df_filt.drop('log_price_doc', axis=1)

    # Filter using Linear Regression
    df_filt_new = df_filt.drop(columns=metadata['bad_lin_reg_columns'])

    if has_target:
        return df_filt_new, target
    else:
        return df_filt_new, None


def preprocess_data(
        df_train: pd.DataFrame,
        df_test: Optional[pd.DataFrame] = None,
        trg_corr_cutoff: float = 0.05,
        mutual_corr_cutoff: float = 0.9,
        ohe_max_features: int = 5,
        lin_reg_pvalue_cutoff: float = 0.1,
) -> Any:
    """
    Full data processing pipeline
    :param df_train: train dataframe
    :param df_test: test dataframe
    :param trg_corr_cutoff: lower cutoff for correlation between feature and target to exclude features
    :param mutual_corr_cutoff: upper cutoff for correlation between feature_X and feature_Y to exclude features
    :param ohe_max_features: `unique_values <= ohe_max_features` -- use OHE, otherwise -- target encoding
    :param lin_reg_pvalue_cutoff: lower pvalue cutoff of F-test to exclude features
    :return: (X_train, y_train) if df_test is None, otherwise (X_train, X_test, y_train, y_test)
    """
    df_train.reset_index(inplace=True)
    if df_test is not None:
        df_test.reset_index(inplace=True)

    X_train_filt, y_train, metadata = preprocess_train_data(
        df_train,
        trg_corr_cutoff,
        mutual_corr_cutoff,
        ohe_max_features,
        lin_reg_pvalue_cutoff,
        return_metadata=False if df_test is None else True
    )

    if df_test is None:
        return X_train_filt, y_train
    else:
        X_test_filt, y_test = preprocess_test_data(
            df_test,
            metadata
        )

        return X_train_filt, X_test_filt, y_train, y_test
