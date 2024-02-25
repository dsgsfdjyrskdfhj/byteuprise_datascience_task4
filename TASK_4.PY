import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import graphviz

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
bank_data = pd.read_csv(url, compression='zip', sep=';')


print("Dataset Information:")
print(bank_data.info())


label_encoder = LabelEncoder()
bank_data['y'] = label_encoder.fit_transform(bank_data['y'])


categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'poutcome']
numerical_features = ['age', 'balance', 'campaign', 'pdays', 'previous']


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])


model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', DecisionTreeClassifier(random_state=42))])


X = bank_data.drop('y', axis=1)
y = bank_data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


param_grid = {
    'classifier__max_depth': [5, 10, 15],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)


best_params = grid_search.best_params_
print(f"\nBest Hyperparameters: {best_params}")


best_model = grid_search.best_estimator_


y_pred = best_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2%}")


print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


tree_rules = export_text(best_model.named_steps['classifier'], feature_names=list(X.columns))
print("\nDecision Tree Rules:")
print(tree_rules)


plt.figure(figsize=(20, 10))
plot_tree(best_model.named_steps['classifier'], feature_names=list(X.columns), class_names=['No Purchase', 'Purchase'], filled=True, rounded=True)
plt.title('Decision Tree Visualization')
plt.show()


dot_data = export_graphviz(best_model.named_steps['classifier'], out_file=None, 
                           feature_names=list(X.columns), class_names=['No Purchase', 'Purchase'],
                           filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree", format="pdf", cleanup=True)
