import pandas as pd
from sklearn.impute import KNNImputer

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, X_test = pd.read_csv("train.csv").drop("season", axis=1), pd.read_csv("test.csv").drop("season", axis=1)
    # concatate and impute missing entries with KNN
    conc = X_train.append(X_test, ignore_index = True)
    imputer = KNNImputer(n_neighbors=8)
    conc = pd.DataFrame(imputer.fit_transform(conc), columns= conc.keys())
    # extract back X_test and read out prediction for CHF
    X_test = conc.tail(len(X_test))
    y_pred = X_test["price_CHF"]
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

