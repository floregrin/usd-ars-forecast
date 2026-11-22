# USD/ARS forecasts

Forecasts are Argentine pesos per US dollar for 1D, 7D, 14D, 1M, 3M, 6M, and 1Y. The final notebook uses Yahoo daily, hourly and minute closes archived by the worker. It selects XGBoost settings on matured historical forecasts for each horizon, then selects the fraction of the predicted price move to retain. The baseline carries the starting close forward. A weight of 0 uses the baseline; a weight of 1 uses the entire XGBoost move.

## Explore the model

Open **USD_ARS_Final.ipynb** on GitHub to see the executed notebook, selected settings, held-out price MAEs and plots. Each horizon has a separate choice from 13 base configurations and 101 blend weights. Test dates were excluded from selection. Each test uses at most four years of historical prices, subject to archive coverage. A fresh model is fit on each sampled date. The operational worker fits once per Buenos Aires day and issues every minute. All seven horizons use the first archived quote from target time through 48 hours later, inclusive. Forecasts without an eligible quote remain in the test export and are excluded from price MAE. On its first scoring cycle after this update, the worker rechecks previously unmatched live forecasts against the wider window. Long-horizon returns overlap. Compare performance on identical matched targets.

## Run locally

1. Clone this repository and open its folder. Create a Python 3.12 environment.

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements-training.txt
   ```

2. To repeat the experiment, place a consistent backup of your worker's `usd_ars.sqlite` alongside the notebook (or set `USD_ARS_DB` to its absolute path). Run `jupyter lab USD_ARS_Final.ipynb` and select **Run All Cells**. The notebook writes `tuned_config.json` and `snapshot/`. These files are already provided for the published run.
3. To issue forecasts using these settings, run `python usd_ars_worker.py --db usd_ars.sqlite` once. Then use `python usd_ars_worker.py --db usd_ars.sqlite --loop` for minute updates. Use a copy of your database when testing. Historical forecasts keep their original version and predictions. An empty database triggers the Yahoo bootstrap.
4. In a second terminal in this folder and environment run `streamlit run usd_ars_dashboard.py`. With the worker database present, the app shows live issues and scored forecasts for the current configuration family. With no database it displays the committed, dated snapshot and historical backtests. Historical snapshot forecasts are reconstructed at the last quote. The live table begins scoring once issued targets mature.

## Publish on Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
2. Select **Create app** and the option to deploy from GitHub. Choose repository **floregrin/usd-ars-forecast**, branch **main**, main file **usd_ars_dashboard.py**.
3. In advanced settings choose Python **3.12** if available. Deploy. Community Cloud installs the dashboard dependencies from `requirements.txt`.
4. Open the generated URL. Select 3M and 1Y and check the displayed snapshot date, blend weight and historical MAE. Copy the URL from **Share**.
5. To update the preview, rerun the notebook locally and commit the regenerated snapshot and configuration together. The public app reads committed files. For uninterrupted minute issuance, run the worker on persistent compute and connect your dashboard to its current data. Opening the public preview does not start a worker.

## Track experiments with MLflow

1. From the project virtual environment run this local tracking server:

   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
   ```

2. In a second terminal, activate the same environment and launch the notebook with tracking enabled:

   ```bash
   export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
   export LOG_MLFLOW=1
   jupyter lab USD_ARS_Final.ipynb
   ```

3. Run all cells. The final cell logs seven runs under **USD-ARS horizon tuning**. Each includes the horizon, base parameters, blend weight, holdout MAEs and scored counts, frozen JSON and test forecast table.
4. Open [the local MLflow UI](http://127.0.0.1:5000), select the experiment and filter on a horizon, e.g. `params.horizon = '3M'`. Compare repeated experiments on matching test dates. The public Streamlit app reads its snapshot independently of this local server.

[Streamlit deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) · [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
