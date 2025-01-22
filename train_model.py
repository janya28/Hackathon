import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset (Make sure to adjust the file path as needed)
df = pd.read_csv(r'C:\Users\Janya\Desktop\crop_recommendation\data\filtered_dataset.csv')

# Select only the relevant features
features = ['temperature', 'humidity', 'rainfall', 'Season', 'State']
target = 'Crop'  # The column containing the crop target

# Filter the dataset to only include the features you need
df = df[features + [target]]

# Convert categorical data into numeric using LabelEncoder
label_encoder_season = LabelEncoder()
label_encoder_state = LabelEncoder()

# Encode the 'Season' and 'State' columns
df['Season'] = label_encoder_season.fit_transform(df['Season'])
df['State'] = label_encoder_state.fit_transform(df['State'])

# Split the dataset into features (X) and target (y)
X = df[features]  # Features: Temperature, Humidity, Rainfall, Season, State
y = df[target]  # Target: Crop

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (using RandomForestClassifier for simplicity)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the model and encoders to files for later use
joblib.dump(model, 'models/crop_recommendation_model.pkl')
joblib.dump(label_encoder_season, 'models/season_encoder.pkl')
joblib.dump(label_encoder_state, 'models/state_encoder.pkl')

print("Model, encoders saved successfully!")
