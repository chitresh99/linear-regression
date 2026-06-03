# Data Pipeline

## Code flow

High level:

```
                    prepare_dataset()
                             │
                             ▼
              dataset_name == "synthetic" ?
                       /             \
                    Yes               No
                     │                 │
                     ▼                 ▼
     generate_synthetic_data()   load_diabetes_dataset()
                     │
                     ▼
               Returns X, y
                     │
                     ▼
                split_data()
                     │
                     ▼
     X_train, X_test, y_train, y_test
                     │
                     ▼
         standardize_features()
                     │
                     ▼
    X_train_scaled, X_test_scaled
                     │
                     ▼
          Create output dictionary
                     │
                     ▼
                return data
```
---

Detailed function call

```
prepare_dataset()
│
├── generate_synthetic_data()
│      │
│      ├── np.random.seed(42)
│      │
│      ├── Create random X
│      │
│      ├── Create coefficients
│      │
│      ├── Compute y
│      │
│      └── return X,y
│
├── split_data(X,y)
│      │
│      ├── Shuffle data
│      │
│      ├── 80% → Train
│      │
│      ├── 20% → Test
│      │
│      └── return
│
├── standardize_features()
│      │
│      ├── scaler = StandardScaler()
│      │
│      ├── fit(X_train)
│      │       │
│      │       ├── Calculate means
│      │       └── Calculate stds
│      │
│      ├── transform(X_train)
│      │
│      ├── transform(X_test)
│      │
│      └── return scaled data
│
└── Return final dictionary
```
---
Mental Model

```
Raw Data
   │
   ▼
Generate / Load Dataset
   │
   ▼
Train-Test Split
   │
   ▼
Learn Scaling Parameters
(mean,std from train only)
   │
   ▼
Scale Train Data
   │
   ▼
Scale Test Data
(using same mean/std)
   │
   ▼
Package Everything
   │
   ▼
Ready For Machine Learning
```