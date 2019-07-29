import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
df = pd.read_csv("myFile.csv", header=None)
print(df)
imp = SimpleImputer(missing_values=np.nan,strategy="mean")
print(imp.fit_transform(df.values))