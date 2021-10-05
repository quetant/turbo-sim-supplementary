# Turbo-Sim supplementary material

This code is giving the results presented in the paper _Turbo-Sim: a generalised generative model with a
physical latent space_.  

## To run the default code

> `python run.py`  

## To pass custom options

> `python run.py --<option 1> xxx --<option 2> xxx ...`

See the `run.py` script for the list of available options.  

## Outputs

By default the outputs are saved in the `experiment/` folder. It contains a _log_ file, _checkpoints_ of the best models, a plot of the _losses_ and several histograms of the desired _observables_.