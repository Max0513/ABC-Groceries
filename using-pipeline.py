#################################
# Import Pipeline object
#################################

# Import packages

import joblib
import pandas as pd
import numpy as np

# import Pipeline

clf = joblib.load("/Data/Model/classification-model.joblib")

# Create new data

new_data = pd.DataFrame(
    {
        "age": [25, np.nan, 50],
        "gender": ["M", "F", np.nan],
        "credit_score": [200, 100, 500],
    }
)

clf.predict(new_data)
