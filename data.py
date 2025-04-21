import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib



# Load dataset
column_names = [
    'erythema', 'scaling', 'definite_borders', 'itching', 'koebner_phenomenon',
    'polygonal_papules', 'follicular_papules', 'oral_mucosal_involvement',
    'knee_and_elbow_involvement', 'scalp_involvement', 'family_history',
    'melanin_incontinence', 'eosinophils_in_the_infiltrate', 'PNL_infiltrate',
    'fibrosis_of_the_papillary_dermis', 'exocytosis', 'acanthosis',
    'hyperkeratosis', 'parakeratosis', 'clubbing_of_the_rete_ridges',
    'elongation_of_the_rete_ridges', 'thinning_of_the_suprapapillary_epidermis',
    'spongiform_pustule', 'munro_microabcess', 'focal_hypergranulosis',
    'disappearance_of_the_granular_layer', 'vacuolisation_and_damage_of_basal_layer',
    'spongiosis', 'saw_tooth_appearance_of_retes', 'follicular_horn_plug',
    'perifollicular_parakeratosis', 'inflammatory_monoluclear_inflitrate',
    'band_like_infiltrate', 'Age', 'Class'
]

df = pd.read_csv('dermatology.data', names=column_names)

# Replace missing values marked as '?' with NaN
df.replace('?', np.nan, inplace=True)

# Convert 'Age' column to numeric
df['Age'] = pd.to_numeric(df['Age'])

# Fill missing values with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Check for nulls again
print(df.isnull().sum())


# Basic statistics
print(df.describe())

# Class distribution
plt.figure(figsize=(8, 5))
df['Class'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Distribution of Skin Disease Classes')
plt.xlabel('Class')
plt.ylabel('Count')
plt.grid(axis='y')
plt.show()

# Age distribution
plt.hist(df['Age'], bins=20, color='coral', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Heatmap of correlations
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap='coolwarm', square=True)
plt.title('Feature Correlation Heatmap')
plt.show()

# Features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
label_decoder = {
    1: 'psoriasis',
    2: 'seborrheic dermatitis',
    3: 'lichen planus',
    4: 'pityriasis rosea',
    5: 'chronic dermatitis',
    6: 'pityriasis rubra pilaris'
}

# Save model
joblib.dump(clf, 'dermatology_rf_model.pkl')

# Save label decoder if using
joblib.dump(label_decoder, 'label_decoder.pkl')
