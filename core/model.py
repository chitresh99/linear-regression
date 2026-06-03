import numpy as np
from typing import Optional,Tuple,Dict,List

class LinearRegression:
    def __init__(
        self,
        method:str = 'gradient_descent',
        learning_rate: float = 0.01,
        n_iterations:int = 1000,
        regularization:Optional[str] = None,
        reg_lambda:float=0.0,
        batch_size:Optional[int] = None,
        random_state:int = 42
    ):
        if method not in ['normal_equation','gradient_descent']:
            raise ValueError("method must be 'normal_equation or 'gradient_descent'")
        if regularization is not None and regularization not in ['l1','l2']:
            raise ValueError("regularization must be None,'l1' or 'l2'")
        
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.batch_size = batch_size
        self.random_state = random_state

        self.theta = None
        self.cost_history = []
        self.is_fitted = False

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        return np.c_[np.ones((X.shape[0],1)),X] # concatenate column of ones with X
    
    def _compute_cost(
            self,
            X: np.ndarray,
            y: np.ndarray,
            theta: np.ndarray
    ) -> float:
        m = len(y)
        predictions = X.dot(theta)
        errors = predictions - y
        
        # mse
        mse = (1 / (2 * m)) * np.sum(errors ** 2)
        
        # add regularization term (don't regularize bias term theta[0])
        if self.regularization == 'l2':
            reg_term = (self.reg_lambda / (2 * m)) * np.sum(theta[1:] ** 2)
            mse += reg_term
        elif self.regularization == 'l1':
            reg_term = (self.reg_lambda / m) * np.sum(np.abs(theta[1:]))
            mse += reg_term
        
        return mse
    
    def _compute_gradient(
            self,
            X:np.ndarray,
            y:np.ndarray,
            theta: np.ndarray
    ) -> np.ndarray:
        m = len(y)
        predictions = X.dot(theta)
        errors = predictions - y
        
        gradient = (1 / m) * X.T.dot(errors)
        
        if self.regularization == 'l2':
            reg_gradient = (self.reg_lambda / m) * theta
            reg_gradient[0] = 0  
            gradient += reg_gradient
        elif self.regularization == 'l1':
            reg_gradient = (self.reg_lambda / m) * np.sign(theta)
            reg_gradient[0] = 0  
            gradient += reg_gradient
        
        return gradient
    
    def _normal_equation(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_features = X.shape[1]
        
        if self.regularization == 'l2':
            reg_matrix = self.reg_lambda * np.eye(n_features)
            reg_matrix[0, 0] = 0  
            theta = np.linalg.inv(X.T.dot(X) + reg_matrix).dot(X.T).dot(y)
        else:
            theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        
        return theta
    

    def _gradient_descent(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        optimize using gradient gescent (batch, mini-batch, or stochastic).
        """
        m, n = X.shape
        np.random.seed(self.random_state)
        theta = np.zeros(n)
        
        self.cost_history_ = []
        
        for i in range(self.n_iterations):
            if self.batch_size is None:
                # batch gradient descent
                X_batch, y_batch = X, y
            else:
                # mini-batch or stochastic gradient descent
                indices = np.random.choice(m, min(self.batch_size, m), replace=False)
                X_batch = X[indices]
                y_batch = y[indices]
            
            # compute gradient and update
            gradient = self._compute_gradient(X_batch, y_batch, theta)
            theta -= self.learning_rate * gradient
            
            # record cost (using full dataset for accurate tracking)
            if i % 10 == 0 or i == self.n_iterations - 1:
                cost = self._compute_cost(X, y, theta)
                self.cost_history_.append(cost)
        
        return theta
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        # validate inputs
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.ndim}D")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got {y.ndim}D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        
        # Add bias term
        X_bias = self._add_bias(X)
        
        # Choose optimization method
        if self.method == 'normal_equation':
            self.theta_ = self._normal_equation(X_bias, y)
            # Compute final cost
            final_cost = self._compute_cost(X_bias, y, self.theta_)
            self.cost_history_ = [final_cost]
        else:
            self.theta_ = self._gradient_descent(X_bias, y)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("model must be fitted before making predictions")
        
        X = np.asarray(X, dtype=np.float64)
        X_bias = self._add_bias(X)
        
        return X_bias.dot(self.theta_)
    
    def get_params(self) -> Dict:
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted first")
        
        return {
            'bias': self.theta_[0],
            'coefficients': self.theta_[1:],
            'theta': self.theta_.copy()
        }
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from utils.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
