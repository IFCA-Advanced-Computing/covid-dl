covid-dl
========

Machine Learning models for COVID-19 risk prediction

> **WARNING** Keep in mind this is a project still in development, and the predictions of the models are not always reliable. Use at your own discretion.

[comment]: <> (Additional information on how we train/tested the different models, as well as the performance results, can be found on the [model's report]&#40;reports/summary.md&#41;.)

## Usage
 
First, copy the `provinces-incidence-mobility.csv` file (obtained from the [covid-risk-map](https://github.com/IFCA/covid-risk-map) repo) to the `data/raw` folder.

If you want to train a model on your data run:
```bash
python train.py
```


If you already have a trained model (in `models/feedforward`) you can go directly to the prediction step. To make a prediction into the future from the last available date run:
```bash
python predict.py
```
This will write a `predictions.csv` file in `data/predictions`.

To view the results interactively, run a plotly instance with:
```bash
python visualize.py
```

![](reports/figures/predictions.png)
