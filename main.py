import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import src.preprocessing as prep
import src.feature_engineering as feat
import src.modeling as model
import src.evaluation as eval

# Load + Clean
df = prep.load_and_clean_data("data/ola_driver.csv")
df = prep.knn_impute(df)

# Feature Engineering
df = feat.add_target_column(df)
df = feat.add_income_rating_trends(df)

# Drop unneeded columns
df = pd.get_dummies(df, columns=['Gender', 'City', 'Education_Level'], drop_first=True)
df = df.drop(['Driver_ID', 'MMMM-YY', 'Date Of Joining', 'LastWorkingDate'], axis=1)

# Split
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
rf_model = model.train_random_forest(X_train, y_train)
eval.evaluate_model(rf_model, X_test, y_test)

xgb_model = model.train_xgboost(X_train, y_train)
eval.evaluate_model(xgb_model, X_test, y_test)
