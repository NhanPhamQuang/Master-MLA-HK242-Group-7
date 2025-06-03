
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# Load Excel file (update the filename and sheet name if needed)
file_path = 'House_price.xlsx'
df = pd.read_excel(file_path, sheet_name=0)  # Default is the first sheet


# Columns to encode
label_cols = ['Compass', 'Street', 'Ward', 'District', 'Category']

# Convert columns to string and apply Label Encoding
label_encoders = {}
for col in label_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # store encoder if you need to inverse later


# # Remove outlier
cols_to_check = ['Price (tỷ VND)']  # Replace with relevant numeric columns

for col in cols_to_check:
    Q1 = df[col].quantile(0.2)
    Q3 = df[col].quantile(0.8)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2 * IQR
    upper_bound = Q3 + 2 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# df = df[df['Price (tỷ VND)'] <40]

### Add feature

# Price per square
df['Price per square'] = df['Price (tỷ VND)'] / df['Area (m²)']
# Balcony or not

def has_balcony(text: str) -> bool:
    text = str(text).lower()
    keywords = ["ban công"]
    return any(kw in text for kw in keywords)

df['Balcony'] = df['Description'].apply(has_balcony)

# Furniture
def detect_furniture(text) -> str:
    if not isinstance(text, str):
        text = str(text).lower()
    else:
        text = text.lower()

    if "nội thất đầy đủ" in text or "full nội thất" in text or "nội thất cao cấp" in text or "đầy đủ nội thất" in text or "nội thất hiện đại" in text or "full nội thất" in text:
        return 2
    elif "nội thất cơ bản" in text or "nội thất tiêu chuẩn" in text:
        return 1
    else:
        return 0  # In case nothing is detected
df['Furniture'] = df['Description'].apply(detect_furniture)


# Drop columns not used
df = df.drop(columns=['Index', 'Description', 'ID', 'City'])

X = df.drop(columns=['Price (tỷ VND)'])
Y = df['Price (tỷ VND)']


print(X, Y)

import matplotlib.pyplot as plt
import seaborn as sns
correlation = df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

"""#Random Forest Regression"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 1.0],  # Removed 'auto'
    'bootstrap': [True, False]
}

rf_model = RandomForestRegressor(random_state=2)
random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_grid,
    n_iter=30,            # Reasonable for 10k records
    scoring='neg_mean_squared_error',
    cv=5,                 # 5-fold CV is standard
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, Y_train)

print("Best params:", random_search.best_params_)

from sklearn import metrics
import matplotlib.pyplot as plt

best_model = random_search.best_estimator_

Y_pred_train = best_model.predict(X_train)
Y_pred_test = best_model.predict(X_test)

### EVALUATE TRAINING ###
R_square_error = metrics.r2_score(Y_train, Y_pred_train)
mean_absolute_error = metrics.mean_absolute_error(Y_train, Y_pred_train)

print('R square error:', R_square_error)
print('Mean absolute error:', mean_absolute_error)

### VISUALIZE TRAINING ###
plt.scatter(Y_train, Y_pred_train)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Random Forest Train Data')
plt.show()

# Residual for training
residuals_train = Y_train - Y_pred_train

plt.scatter(Y_pred_train, residuals_train, alpha=0.5, color='blue')
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Training Residual Plot - Random Forest")
plt.show()

### EVALUATE AND VISUALIZE TEST DATA ###
R_square_error = metrics.r2_score(Y_test, Y_pred_test)
mean_absolute_error = metrics.mean_absolute_error(Y_test, Y_pred_test)

print('R square error:', R_square_error)
print('Mean absolute error:', mean_absolute_error)

plt.scatter(Y_test, Y_pred_test)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Random Forest Test Data')
plt.show()

residuals_test = Y_test - Y_pred_test
plt.scatter(Y_pred_test, residuals_test, alpha=0.5, color='blue')
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Testing Residual Plot - Random Forest")
plt.show()

import joblib
export_model = joblib.dump(best_model,'rf.jb')
joblib.dump(label_encoders, 'label_encoders.joblib')

