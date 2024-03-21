# Simultaneous quantile regression with XGBoost

This repository contains a demonstration how multiple quantiles can be predicted
simultaneously with XGBoost. The key idea is to use a smooth approximation of the
pinball loss, the arctan pinball loss, that has a relatively large second derivative.

The approximation is given by: 
$$L^{(\text{arctan})}_{\tau, s}(u) = (\tau - 0.5 + \frac{\arctan (u/s)}{\pi})u	 + \frac{s}{\pi}$$.

Some important settings:

- The parameter $s$ in the loss function determines the amount of smoothing. A smaller
values gives a closer approximation but also a much smaller second derivative.
A larger value gives more conservative quantiles when $\tau$ is larger than 0.5,
the quantile becomes larger and vice versa. Values between 0.05 and 0.1
appear to work well. It may be a good idea to optimize this parameter.
- Set min-child-weight to zero. The second derivatives can be a lot smaller than 1
and this parameter may prevent any splits.
- Use a relatively small max-delta-step. We used a default of 0.5. This prevents
excessive steps that could happen due to the relatively small second derivative.
- For the same reason, we used a slightly lower learning rate of 0.05.

The file 'objective_functions.py' contains an implementation of the loss
arctan pinball loss. Note that this is written for the sklearn api and therefore
returns the negative gradient. This needs to be changed when working with
the default xgboost package. Also not that this implementation use a slow 
for loop in python and may slow down the model.

The file 'sin_example.ipynb' provides a simple 1d toy-example as a proof of concept.
The file 'UCI_experiment' runs a more in-depth comparison. In general, it is 
possible to predict multiple quantiles simultaneously with a single model, resulting 
in far fewer quantile crossings.