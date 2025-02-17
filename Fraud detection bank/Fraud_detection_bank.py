# Importing required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r"C:\Users\calla\source\repos\Fraud detection bank\bank_transactions_data_2.csv")

# Display the first few rows and basic information
print(df.head())
print(df.info())

#Handle Missing Values and Duplicates
missing_values = df.isnull().sum()
print(f"Missing values:\n{missing_values}")

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"Number of duplicates: {duplicates}")


#Categorical Features Exploration
# Explore categorical columns
print("Unique values in 'TransactionType':", df['TransactionType'].unique())
print("Unique values in 'Location':", df['Location'].unique())
print("Unique values in 'CustomerOccupation':", df['CustomerOccupation'].unique())
print("Unique values in 'Channel':", df['Channel'].unique())

#We'll visualize the distributions of transaction amounts and account balances.
# Distribution of TransactionAmount (Numerical Feature)
plt.figure(figsize=(8, 6))
sns.histplot(df['TransactionAmount'], kde=True, color='blue')
plt.title('Transaction Amount Distribution')
plt.show()

# Distribution of AccountBalance
plt.figure(figsize=(8, 6))
sns.histplot(df['AccountBalance'], kde=True, color='green')
plt.title('Account Balance Distribution')
plt.show()

#Let's explore the TransactionDate column to see if there’s any trend or pattern over time.
# Convert TransactionDate to datetime
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])

# Plot transaction frequency over time
plt.figure(figsize=(10, 6))
df.groupby(df['TransactionDate'].dt.date).size().plot(kind='line')
plt.title('Transaction Frequency Over Time')
plt.ylabel('Number of Transactions')
plt.xlabel('Date')
plt.show()

#Creating Initial Fraud Labels
# Convert TransactionDate to datetime
# Define fraud detection rules

# Rule 1: High transaction amount (top 1% of all transactions)
high_amount_threshold = df['TransactionAmount'].quantile(0.997)
df['HighAmountFlag'] = (df['TransactionAmount'] >= high_amount_threshold).astype(int)

# Rule 2: High number of login attempts (greater than 3)
df['HighLoginAttemptsFlag'] = (df['LoginAttempts'] > 5).astype(int)

# Rule 3: Frequent location changes (check if same account has different locations)
df['LocationChangeFlag'] = df.groupby('AccountID')['Location'].transform(lambda x: x.nunique() > 7).astype(int)

# Rule 4: Fast consecutive transactions (time difference < 10 minutes for same account)
df = df.sort_values(['AccountID', 'TransactionDate'])
df['TimeDiff'] = df.groupby('AccountID')['TransactionDate'].diff().dt.total_seconds()
df['FastTransactionFlag'] = (df['TimeDiff'] < 600).astype(int)  # 600 seconds = 10 minutes

# Combine the rules to create an initial fraud flag
df['is_fraud'] = (
    df['HighAmountFlag'] | 
    df['HighLoginAttemptsFlag'] | 
    df['LocationChangeFlag'] | 
    df['FastTransactionFlag']
).astype(int)

# Check the distribution of fraud labels
fraud_distribution = df['is_fraud'].value_counts(normalize=True)
print(fraud_distribution)

from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Identify categorical columns (object type)
categorical_columns = df.select_dtypes(include=['object']).columns

# Apply LabelEncoder to each categorical column
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

from sklearn.model_selection import train_test_split

# Define features and target
X = df.drop(columns=['is_fraud', 'TransactionDate'])  # drop target and non-relevant columns
y = df['is_fraud']

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Model Selection and Training
from sklearn.ensemble import RandomForestClassifier

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Fraud", "Fraud"], yticklabels=["Non-Fraud", "Fraud"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
