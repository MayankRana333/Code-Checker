import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import ast
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import MaxAbsScaler
from flask import Flask, request, render_template

print("Loading datasets...")
human_df = pd.read_csv('human_code.csv')
ai_df = pd.read_csv('ai_generated_code.csv')

human_df['label'] = 0  
ai_df['label'] = 1     

print("Combining datasets...")
data_df = pd.concat([human_df, ai_df], ignore_index=True).sample(frac=1).reset_index(drop=True)

# TF-IDF Vectorization
print("Vectorizing with TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000, token_pattern=r"(?u)\b\w+\b")
tfidf_features = tfidf_vectorizer.fit_transform(data_df["code"])

# AST Feature Extraction
def extract_ast_features(code):
    try:
        tree = ast.parse(code)
        return {
            "num_for_loops": len([n for n in ast.walk(tree) if isinstance(n, ast.For)]),
            "num_if_statements": len([n for n in ast.walk(tree) if isinstance(n, ast.If)]),
            "num_func_defs": len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
        }
    except:
        return {"num_for_loops": 0, "num_if_statements": 0, "num_func_defs": 0}

print("Extracting AST features...")
ast_features = [extract_ast_features(code) for code in data_df["code"]]
ast_df = pd.DataFrame(ast_features)
ast_scaled = MaxAbsScaler().fit_transform(ast_df)
ast_sparse = csr_matrix(ast_scaled)

# Combine features
X = hstack([tfidf_features, ast_sparse])
y = data_df["label"]

# Train-test split
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train XGBoost with GridSearchCV
print("Training XGBoost with GridSearchCV...")
clf = xgb.XGBClassifier(objective='binary:logistic', random_state=42)

param_grid = {
    'n_estimators': [100],
    'max_depth': [10],
    'learning_rate': [0.1],
    'subsample': [1.0]
}

grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=1)
grid_search.fit(X_train, y_train)

best_clf = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Cross-validation
print("Cross-validating...")
cv_scores = cross_val_score(best_clf, X, y, cv=5)
print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Train final model
print("Training final model...")
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)

print("Evaluating...")
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check_code', methods=['POST'])
def check_code():
    code_input = request.form['code_input']
    tfidf_vec = tfidf_vectorizer.transform([code_input])
    ast_feat = extract_ast_features(code_input)
    ast_df = pd.DataFrame([ast_feat])
    ast_scaled = MaxAbsScaler().fit_transform(ast_df)
    combined = hstack([tfidf_vec, csr_matrix(ast_scaled)])
    prediction = best_clf.predict(combined)[0]
    result = "AI-generated" if prediction == 1 else "Human-written"
    return render_template("result.html", result=result)

if __name__ == '__main__':
    app.run(debug=True)
