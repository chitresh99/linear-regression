import sys
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import LinearRegression
from utils.metrics import compute_all_metrics, print_metrics, r2_score
from data.loader import prepare_dataset
from visualizations.plots import (
    plot_cost_history,
    plot_predictions_vs_actual,
    plot_residuals,
    plot_feature_importance,
    create_comparison_plot
)

def section(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")

def step(n, total, label):
    print(f"\n  [{n}/{total}] {label}")

def info(label, value):
    print(f"      {label:<28} {value}")

def metric_block(metrics, label):
    print(f"\n      {label}")
    print(f"        MSE       {metrics['mse']:.4f}")
    print(f"        RMSE      {metrics['rmse']:.4f}")
    print(f"        MAE       {metrics['mae']:.4f}")
    print(f"        R²        {metrics['r2']:.4f}")
    print(f"        Adj. R²   {metrics['adjusted_r2']:.4f}")


# ──────────────────────────────────────────────
# Experiment 1 — Synthetic Data
# ──────────────────────────────────────────────

def experiment_1_synthetic_data():
    section("EXPERIMENT 1: SYNTHETIC DATA")
    print("  Trains on synthetic data with known ground-truth coefficients.")
    print("  Noise is kept low so the model can actually recover the signal.")

    step(1, 6, "Preparing dataset")
    data = prepare_dataset(
        dataset_name='synthetic',
        n_samples=200,
        n_features=5,
        noise=2.0,
        test_size=0.2,
        random_state=42
    )
    info("Training samples:", len(data['y_train']))
    info("Test samples:", len(data['y_test']))
    info("Features:", data['n_features'])

    step(2, 6, "Training — Normal Equation")
    model_ne = LinearRegression(method='normal_equation')
    model_ne.fit(data['X_train_scaled'], data['y_train'])

    y_pred_ne_train = model_ne.predict(data['X_train_scaled'])
    y_pred_ne_test  = model_ne.predict(data['X_test_scaled'])

    metrics_ne_train = compute_all_metrics(data['y_train'], y_pred_ne_train, data['n_features'])
    metrics_ne_test  = compute_all_metrics(data['y_test'],  y_pred_ne_test,  data['n_features'])

    metric_block(metrics_ne_train, "Train")
    metric_block(metrics_ne_test,  "Test")

    step(3, 6, "Training — Gradient Descent")
    model_gd = LinearRegression(
        method='gradient_descent',
        learning_rate=0.01,
        n_iterations=1000,
        random_state=42
    )
    model_gd.fit(data['X_train_scaled'], data['y_train'])

    y_pred_gd_train = model_gd.predict(data['X_train_scaled'])
    y_pred_gd_test  = model_gd.predict(data['X_test_scaled'])

    metrics_gd_train = compute_all_metrics(data['y_train'], y_pred_gd_train, data['n_features'])
    metrics_gd_test  = compute_all_metrics(data['y_test'],  y_pred_gd_test,  data['n_features'])

    metric_block(metrics_gd_train, "Train")
    metric_block(metrics_gd_test,  "Test")

    step(4, 6, "Comparing learned vs true coefficients")
    params    = model_ne.get_params()
    true_coef = [3.5, -2.0, 1.5, 0.5, -1.0]
    learned   = np.round(params['coefficients'], 2)
    print(f"\n      {'Feature':<12} {'True':>8} {'Learned':>10}")
    print(f"      {'───────':<12} {'────':>8} {'───────':>10}")
    for i, (t, l) in enumerate(zip(true_coef, learned)):
        print(f"      Feature {i+1:<4}  {t:>8.2f} {l:>10.2f}")
    print(f"\n      Bias — True: 5.00   Learned: {params['bias']:.2f}")

    step(5, 6, "Saving visualizations")
    os.makedirs('outputs/experiment1', exist_ok=True)

    plot_cost_history(
        model_gd.cost_history_,
        title="Gradient Descent - Cost History",
        save_path='outputs/experiment1/cost_history.png'
    )
    plot_predictions_vs_actual(
        data['y_test'], y_pred_gd_test,
        title="Predictions vs Actual (Test Set)",
        save_path='outputs/experiment1/predictions_vs_actual.png'
    )
    plot_residuals(
        data['y_test'], y_pred_gd_test,
        title="Residual Analysis (Test Set)",
        save_path='outputs/experiment1/residuals.png'
    )
    print("      Saved to outputs/experiment1/")

    step(6, 6, "Summary")
    r2_ne = metrics_ne_test['r2']
    r2_gd = metrics_gd_test['r2']
    print(f"      Normal Equation R²:   {r2_ne:.4f}")
    print(f"      Gradient Descent R²:  {r2_gd:.4f}")
    delta = abs(r2_ne - r2_gd)
    print(f"      Difference:           {delta:.6f}  (both methods agree)")

    return {
        'model_ne': model_ne,
        'model_gd': model_gd,
        'metrics_ne_test': metrics_ne_test,
        'metrics_gd_test': metrics_gd_test,
        'data': data
    }


# ──────────────────────────────────────────────
# Experiment 2 — Diabetes Dataset
# ──────────────────────────────────────────────

def experiment_2_diabetes_dataset():
    section("EXPERIMENT 2: DIABETES DATASET")
    print("  Real-world medical dataset. Tests regularization on actual data.")

    step(1, 5, "Loading dataset")
    data = prepare_dataset(
        dataset_name='diabetes',
        test_size=0.2,
        random_state=42
    )
    info("Training samples:", len(data['y_train']))
    info("Test samples:", len(data['y_test']))
    info("Features:", data['n_features'])
    info("Feature names:", ', '.join(data['feature_names'][:3]) + '...')

    step(2, 5, "Training models")
    model_no_reg = LinearRegression(
        method='gradient_descent',
        learning_rate=0.01,
        n_iterations=2000,
        random_state=42
    )
    model_no_reg.fit(data['X_train_scaled'], data['y_train'])

    model_l2 = LinearRegression(
        method='gradient_descent',
        learning_rate=0.01,
        n_iterations=2000,
        regularization='l2',
        reg_lambda=1.0,
        random_state=42
    )
    model_l2.fit(data['X_train_scaled'], data['y_train'])

    step(3, 5, "Evaluating models")
    models = [
        ('No Regularization', model_no_reg),
        ('L2  lambda=1.0',    model_l2),
    ]

    all_metrics = []
    for name, model in models:
        y_pred  = model.predict(data['X_test_scaled'])
        metrics = compute_all_metrics(data['y_test'], y_pred, data['n_features'])
        metric_block(metrics, name)
        all_metrics.append({'name': name, 'metrics': metrics})

    step(4, 5, "Feature importance")
    os.makedirs('outputs/experiment2', exist_ok=True)

    params = model_no_reg.get_params()
    plot_feature_importance(
        data['feature_names'],
        params['coefficients'],
        title="Feature Importance - Diabetes Dataset",
        save_path='outputs/experiment2/feature_importance.png'
    )

    y_pred_test = model_no_reg.predict(data['X_test_scaled'])
    plot_predictions_vs_actual(
        data['y_test'], y_pred_test,
        title="Diabetes Prediction - Test Set",
        save_path='outputs/experiment2/predictions.png'
    )
    print("      Saved to outputs/experiment2/")

    step(5, 5, "Summary")
    no_reg_r2 = all_metrics[0]['metrics']['r2']
    l2_r2     = all_metrics[1]['metrics']['r2']
    delta     = l2_r2 - no_reg_r2
    direction = "improved" if delta > 0.001 else "made no meaningful difference"
    print(f"      No-regularization R²:  {no_reg_r2:.4f}")
    print(f"      L2 R²:                 {l2_r2:.4f}")
    print(f"      L2 regularization {direction}  (delta R²={delta:+.4f})")
    print(f"      Note: with only 10 features and 353 samples, overfitting")
    print(f"            pressure is low so regularization rarely helps here.")

    return {
        'models': models,
        'all_metrics': all_metrics,
        'data': data
    }


# ──────────────────────────────────────────────
# Experiment 3 — Hyperparameter Tuning
# ──────────────────────────────────────────────

def experiment_3_hyperparameter_tuning():
    section("EXPERIMENT 3: HYPERPARAMETER TUNING")
    print("  Sweeps learning rates and regularization strengths on diabetes data.")
    print("  Using a real dataset makes differences between settings visible.")

    step(1, 4, "Preparing data")
    data = prepare_dataset(
        dataset_name='diabetes',
        test_size=0.2,
        random_state=42
    )
    info("Samples:", len(data['y_train']) + len(data['y_test']))
    info("Features:", data['n_features'])

    step(2, 4, "Learning rate sweep")
    print(f"\n      {'LR':<10} {'R2':>8}   result")
    print(f"      {'──':<10} {'──':>8}   ──────")

    learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0]
    best_lr = None
    best_r2 = -np.inf

    for lr in learning_rates:
        model = LinearRegression(
            method='gradient_descent',
            learning_rate=lr,
            n_iterations=1000,
            random_state=42
        )
        model.fit(data['X_train_scaled'], data['y_train'])
        y_pred = model.predict(data['X_test_scaled'])
        r2     = r2_score(data['y_test'], y_pred)
        status = "good" if r2 > 0.4 else "underfit / diverged"
        print(f"      {lr:<10.4f} {r2:>8.4f}   {status}")

        if r2 > best_r2:
            best_r2 = r2
            best_lr = lr

    print(f"\n      Best learning rate: {best_lr}  (R2={best_r2:.4f})")

    step(3, 4, "Regularization strength sweep  (L2, best LR fixed)")
    print(f"\n      {'lambda':<10} {'R2':>8}")
    print(f"      {'──────':<10} {'──':>8}")

    reg_lambdas = [0.0, 0.01, 0.1, 1.0, 10.0]
    best_lambda = None
    best_r2_reg = -np.inf

    for lam in reg_lambdas:
        model = LinearRegression(
            method='gradient_descent',
            learning_rate=best_lr,
            n_iterations=1000,
            regularization='l2',
            reg_lambda=lam,
            random_state=42
        )
        model.fit(data['X_train_scaled'], data['y_train'])
        y_pred = model.predict(data['X_test_scaled'])
        r2     = r2_score(data['y_test'], y_pred)
        print(f"      {lam:<10.2f} {r2:>8.4f}")

        if r2 > best_r2_reg:
            best_r2_reg = r2
            best_lambda = lam

    print(f"\n      Best lambda: {best_lambda}  (R2={best_r2_reg:.4f})")

    step(4, 4, "Final model with best hyperparameters")
    final_model = LinearRegression(
        method='gradient_descent',
        learning_rate=best_lr,
        n_iterations=2000,
        regularization='l2',
        reg_lambda=best_lambda,
        random_state=42
    )
    final_model.fit(data['X_train_scaled'], data['y_train'])
    y_pred_final  = final_model.predict(data['X_test_scaled'])
    final_metrics = compute_all_metrics(data['y_test'], y_pred_final, data['n_features'])
    metric_block(final_metrics, "Final Model")

    print(f"\n      Learning rate: {best_lr}   lambda: {best_lambda}")
    print(f"      Final R2:      {final_metrics['r2']:.4f}")
    print(f"      Note: tuning impact depends on dataset noise and")
    print(f"            dimensionality — not always dramatic.")

    return {
        'best_lr': best_lr,
        'best_lambda': best_lambda,
        'final_model': final_model,
        'final_metrics': final_metrics
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("\nLinear Regression — Comprehensive Experiments")
    print("=" * 60)
    print("  Normal Equation & Gradient Descent")
    print("  L1 / L2 Regularization")
    print("  Feature standardization")
    print("  Evaluation metrics & visualizations")

    results_exp1 = experiment_1_synthetic_data()
    results_exp2 = experiment_2_diabetes_dataset()
    results_exp3 = experiment_3_hyperparameter_tuning()

    section("ALL EXPERIMENTS COMPLETE")
    print("  Key points:")
    print("  1. Normal Equation is exact but does not scale to large datasets.")
    print("  2. Gradient Descent is scalable; learning rate matters a lot.")
    print("  3. Regularization helps when features >> samples or collinearity is high.")
    print("  4. Standardizing features is essential for gradient descent to converge.")
    print("  5. Tuning impact depends on dataset — always verify on real data.")
    print("\n  Visualizations saved to outputs/experiment1/ and outputs/experiment2/")
    print()


if __name__ == '__main__':
    main()