import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

train_df = pd.read_csv("train.csv").drop("season", axis=1)
test_df = pd.read_csv("test.csv").drop("season", axis=1)
test_df.insert(loc=1, column="price_CHF", value=np.nan)

impute = KNNImputer(n_neighbors=8).fit(train_df)
pd.DataFrame.from_records(
    impute.transform(test_df.to_numpy()), columns=test_df.columns
)["price_CHF"].to_csv("results.csv", index=False)
