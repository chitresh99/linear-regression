#######################################
# Steps
# 1.Creating datasets
# 2.Loading real-world data
# 3.Splitting into train/test sets
# 4.Feature scaling (standardization)
# 5.Returning everything in a clean dictionary
#######################################

import numpy as np
from typing import Tuple,Optional,Dict
from sklearn.datasets import load_diabetes,make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_synthetic_data(
        n_samples=200,
        n_features=5,
        noise=10.0,
        random_state=42
) -> Tuple[np.ndarray,np.ndarray]:
    np.random.seed(random_state)
    X = np.random.randn(n_samples,n_features) # create random numbers from standard distribution
    true_coefficient = np.array([3.5,-2.0,1.5,0.5,-1.0][:n_features])
    true_bias = 5.0

    y = X.dot(true_coefficient) + true_bias + np.random.randn(n_samples) * noise
    return X,y

def load_diabetes_dataset()->Tuple[np.ndarray,np.ndarray,Dict]:
    diabetes = load_diabetes()
    metadata = {
        'feature_names': diabetes.feature_names,
        'target_name': diabetes.target_names[0],
        'description':diabetes.DESCR[:500] + "..."
    }
    return diabetes.data,diabetes.target,metadata

def split_data(
        X: np.ndarray,
        y: np.ndarray,
        test_size:float = 0.2,
        random_state: int = 42
)->Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]: 
    return train_test_split(
        X,y,
        test_size=test_size,
        random_state=random_state
    )

def standardize_features( #basically here we scale the train data and using the same statistics we scale the test data correct ? 
    X_train:np.ndarray,
    X_test:Optional[np.ndarray] = None
) -> Tuple[np.ndarray,Optional[np.ndarray],StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled,X_test_scaled,scaler
    else:
        return X_train_scaled, None , scaler
    
def prepare_dataset(
    dataset_name: str = 'synthetic',
    test_size: float = 0.2,
    random_state: int = 42,
    **kwargs
) -> Dict:
    # Load data
    if dataset_name == 'synthetic':
        X, y = generate_synthetic_data(random_state=random_state, **kwargs)
        feature_names = [f"Feature_{i+1}" for i in range(X.shape[1])]
        metadata = {'feature_names': feature_names}
    elif dataset_name == 'diabetes':
        X, y, metadata = load_diabetes_dataset()
        feature_names = metadata['feature_names']
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )
    
    X_train_scaled, X_test_scaled, scaler = standardize_features(X_train, X_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'feature_names': feature_names,
        'metadata': metadata,
        'scaler': scaler,
        'n_samples': len(y),
        'n_features': X.shape[1]
    }