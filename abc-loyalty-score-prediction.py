# Import packages
import pandas as pd
import pickle

# Import customers data
to_be_scored = pickle.load(open("/Data/Processed/regression_scoring.p", "rb"))

# Import model
regressor = pickle.load(open("/Data/Processed/random_forest_regression_model.p", "rb"))

# Drop unused columns
to_be_scored.drop(["customer_id"], axis=1, inplace=True)

# Drop missing values
to_be_scored.dropna(how="any", inplace=True)

# Make predictions
loyalty_predictions = regressor.predict(to_be_scored)
loyalty_predictions.head()
