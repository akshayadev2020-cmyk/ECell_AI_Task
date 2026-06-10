# FINAL 
print(" DETAILED CLASSIFICATION REPORTS (Precision, Recall, F1) ")

print("\n--- ADABOOST METRICS ---")
print(classification_report(y_test, ada_preds, target_names=target_names, zero_division=0))

print("\n--- XGBOOST METRICS ---")
print(classification_report(y_test, xgb_preds, target_names=target_names, zero_division=0))

print("\n--- CATBOOST METRICS ---")
print(classification_report(y_test, cat_preds, target_names=target_names, zero_division=0))
