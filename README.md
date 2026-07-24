# Customer Churn Prediction (ANN)

A binary classification model built with Keras/TensorFlow to predict whether a bank customer will churn (exit), based on the [Churn Modelling dataset](https://www.kaggle.com/datasets/harkiratssingh/customer-churn-dataset).

## Dataset

- 10,000 customer records, 14 columns
- Target: `Exited` (1 = churned, 0 = retained) — imbalanced, ~79.6% / 20.4%
- Features: credit score, geography, gender, age, tenure, balance, number of products, credit card ownership, active member status, estimated salary

## Pipeline

1. **Cleaning** — dropped `RowNumber`, `CustomerId`, `Surname` (identifiers, no predictive value)
2. **Encoding** — one-hot encoded `Geography` and `Gender` with `drop_first=True`
3. **Split** — `train_test_split`, 80/20, `random_state=42`
4. **Scaling** — `StandardScaler` fit on training features
5. **Model** — `Sequential` ANN:
   - Dense(11, activation='sigmoid', input_dim=11)
   - Dense(11, activation='sigmoid')
   - Dense(1, activation='sigmoid')
   - Optimizer: Adam, Loss: binary_crossentropy, Metric: accuracy
6. **Training** — 100 epochs, batch_size=50, validation_split=0.2
7. **Evaluation** — accuracy_score on held-out test set

## Results

| Metric | Value |
|---|---|
| Test accuracy | *(recompute — see note below)* |

## Known issues / TODO

- [ ] `y_pred.argmax(axis=-1)` on a single-sigmoid-output column always returns 0 for every row — this silently turns evaluation into "always predict no-churn." Replace with `(y_pred > 0.5).astype(int)` and recompute accuracy.
- [ ] `scaler.fit_transform(X_test)` refits the scaler on test data — data leakage. Should be `scaler.transform(X_test)`, fit only on train.
- [ ] Scaled features (`X_train_scaled`, `X_test_scaled`) are computed but never passed into `model.fit` / `model.predict` — training currently runs on unscaled `X_train`.
- [ ] Hidden layers use `sigmoid`; consider `relu` for hidden layers (sigmoid saturates and slows training in deeper stacks).
- [ ] No class-imbalance handling (~80/20 split) — accuracy alone is a weak metric here; add precision/recall/F1/confusion matrix.

## Requirements

```
pandas
numpy
scikit-learn
tensorflow
matplotlib
```

