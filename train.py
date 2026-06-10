print("\nSTAGE 4: Training and Comparing Models...")
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)
target_names = ['High Risk (0)', 'Medium Risk (1)', 'Low Risk (2)']

print("\nTraining AdaBoost Classifier...")
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_model.fit(X_train, y_train)
ada_preds = ada_model.predict(X_test)
print(" AdaBoost Accuracy:", f"{accuracy_score(y_test, ada_preds) * 100:.2f}%")
print("AdaBoost Confusion Matrix:\n", confusion_matrix(y_test, ada_preds))

print("\nTraining XGBoost Classifier...")
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
print(" XGBoost Accuracy:", f"{accuracy_score(y_test, xgb_preds) * 100:.2f}%")
print("XGBoost Confusion Matrix:\n", confusion_matrix(y_test, xgb_preds))

print("\nTraining CatBoost Classifier...")
cat_model = CatBoostClassifier(iterations=100, random_state=42, verbose=0)
cat_model.fit(X_train, y_train)
cat_preds = cat_model.predict(X_test)
print(" CatBoost Accuracy:", f"{accuracy_score(y_test, cat_preds) * 100:.2f}%")
print("CatBoost Confusion Matrix:\n", confusion_matrix(y_test, cat_preds))
