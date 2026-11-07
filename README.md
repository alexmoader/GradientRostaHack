# Gradient RTO Forecasting

A two-person team project on monthly retail revenue (РТО) forecasting. We built a model to estimate March revenue for stores from historical monthly records and store characteristics.

## What we explored

We compared multiple feature sets and forecasting approaches, including CatBoost, LightGBM, and robust linear models. The selected solution combines:

1. A cascade that flags extreme January and February changes and replaces them using the typical transition across stores.
2. A six-formula median ensemble built from weighted January and February revenue.
3. A modest Huber-regression correction using store characteristics and recent operational features.

The team did not reach the finalist stage. I recall that the MAPE gap to finalists was small, but the exact value and metric convention are not confirmed. Experimental scripts in the original project folder record candidate public scores of 90.39 and 90.46. The selected script separately calculates March 2024 validation MAPE as a percentage, but no run result is saved; the relationship between the candidate scores and MAPE is unverified.

## My contribution

I worked with my teammate across the project. My work included breaking down the problem, trying model families and parameter settings, comparing results, engineering features, and writing code. We shared much of the experimentation and implementation.

## Run the selected solution

The competition data is not included. Use this code only with a copy of the dataset that you are allowed to access and process. From the repository root:

```powershell
python -m pip install -r requirements.txt
python src/solution.py --input data/train.csv --output outputs/submission.csv
```

The CSV must contain the columns used by the script, including `new_id`, `Год`, `Месяц`, `РТО`, the listed store characteristics, and recent operational features. The output contains `new_id` and the forecast `rto`.

## Data and validation notes

Raw data, test files, notebooks with saved outputs, and generated plots are excluded. Store-level data may have redistribution restrictions, so check the event rules before sharing any dataset. The script reports a local March 2024 validation MAPE; this is not the same as a verified final leaderboard result. The historical result has not been independently reproduced from a clean checkout.

## Project status

Portfolio presentation of a team hackathon project. This repository does not claim finalist placement or a verified leaderboard rank.

