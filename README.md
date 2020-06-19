# FIRES
This repository contains an implementation of the FIRES framework that is introduced in *Haug et al. (2020), Leveraging Model Inherent Variable Importance for Stable Online Feature Selection.*

The paper will be published in the proceedings of the KDD'20 conference. In the meantime, please refer to [arXiv](https://arxiv.org/abs/2006.10398).

## Apply FIRES to Your Project
The FIRES implementation provided here uses a Probit base model to select features in binary classification settings.
It can be used as follows:

```python
import numpy as np
from skmultiflow.data import FileStream
from skmultiflow.neural_networks import PerceptronMask
from fires import FIRES
from sklearn.metrics import accuracy_score

# Load data as scikit-multiflow FileStream
# NOTE: FIRES accepts only numeric values. Please one-hot-encode or factorize string/char variables
# Additionally, we suggest users to normalize all features, e.g. by using scikit-learn's MinMaxScaler()
stream = FileStream('yourData.csv', target_idx=0)
stream.prepare_for_use()

# Initial fit of the predictive model
predictor = PerceptronMask()
x, y = stream.next_sample(batch_size=100)
predictor.partial_fit(x, y, stream.target_values)

# Initialize FIRES
fires_model = FIRES(n_total_ftr=stream.n_features,          # Total no. of features
                    target_values=stream.target_values,     # Unique target values (class labels)
                    mu_init=0,                              # Initial importance parameter
                    sigma_init=1,                           # Initial uncertainty parameter
                    penalty_s=0.01,                         # Penalty factor for the uncertainty (corresponds to gamma_s in the paper)
                    penalty_r=0.01,                         # Penalty factor for the regularization (corresponds to gamma_r in the paper)
                    epochs=1,                               # No. of epochs that we use each batch of observations to update the parameters
                    lr_mu=0.01,                             # Learning rate for the gradient update of the importance
                    lr_sigma=0.01,                          # Learning rate for the gradient update of the uncertainty
                    scale_weights=True,                     # If True, scale feature weights into the range [0,1]
                    model='probit')                         # Name of the base model to compute the likelihood

# Prequential evaluation
n_selected_ftr = 10

while stream.has_more_samples():
    # Load a new sample
    x, y = stream.next_sample(batch_size=10)

    # Select features
    ftr_weights = fires_model.weigh_features(x, y)  # Get feature weights with FIRES
    ftr_selection = np.argsort(ftr_weights)[::-1][:n_selected_ftr]

    # Truncate x (retain only selected features, 'remove' all others, e.g. by replacing them with 0)
    x_reduced = np.zeros(x.shape)
    x_reduced[:, ftr_selection] = x[:, ftr_selection]

    # Test
    y_pred = predictor.predict(x)
    print(accuracy_score(y, y_pred))

    # Train
    predictor.partial_fit(x, y)

# Restart the FileStream
stream.restart()
```

## Use Your Own Predictive Model
To use FIRES with your own base model, 
you need to substitute the placeholders `### ADD YOUR OWN MODEL HERE ###` in *fires.py* accordingly.

If you have developed a new instantiation of FIRES that is worth sharing with others, feel free to submit a pull request. 
