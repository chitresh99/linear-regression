# Model

## Architecture

```
                 ┌─────────────────┐
                 │ User Data       │
                 │ X, y            │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ fit(X, y)       │
                 └────────┬────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
 ┌─────────────────┐            ┌─────────────────┐
 │ Validate Input  │            │ Add Bias Column │
 │ Shape Checks    │            │ _add_bias()     │
 └────────┬────────┘            └────────┬────────┘
          │                              │
          └──────────────┬───────────────┘
                         │
                         ▼
               ┌─────────────────┐
               │ Choose Method   │
               │ normal_equation │
               │ or GD           │
               └───────┬─────────┘
                       │
       ┌───────────────┴───────────────┐
       │                               │
       ▼                               ▼
┌─────────────────┐         ┌─────────────────┐
│ Normal Equation │         │ Gradient        │
│ _normal_equation│         │ Descent         │
└────────┬────────┘         └────────┬────────┘
         │                           │
         ▼                           ▼
    Compute θ                  Iterate many times
                                     │
                                     ▼
                             _compute_gradient()
                                     │
                                     ▼
                              Update θ
                                     │
                                     ▼
                             _compute_cost()
                                     │
                                     ▼
                               Cost History
                                   │
                                   ▼
                          Store self.theta_
                                   │
                                   ▼
                          is_fitted_ = True

```