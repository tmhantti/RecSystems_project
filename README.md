
# Project work for course Recommender Systems (DATA.ML.360), Autumn 2026

_Authors_: Tatu Anttila [@tmhantti](https://github.com/tmhantti), Arttu Kiviranta [@akiviranta](https://github.com/akiviranta), Patrik Salmensaari [@pasasalm](https://github.com/pasasalm)

##   Current stage of the work 

All required material are committed to the repository for Part I evaluation.

## Repository contents

### Subfolders

* **_report_and_results_** - contains report (*Recommender_Systems_Report.docx*) and results (*Recommender_Systems_Results.xlsm*) assignment files. 

* **_plots_**              - contains figures created by the code.

## Project setup

The project was done using Python 3.12 using package manager _**venv**_. After cloning the repository, navigate to the repository main folder and take the following steps (for Windows console/Powershell): 

 1. Verify that Python 3.12 is installed:
	```
	py -3.12 --version
	```
2. Create virtual environment: 
	```
	py -3.12 -m venv .venv
	```
3. Activate the environment: 
	```
	.\.venv\Scripts\Activate.ps1
	```
4. Install dependencies:
	```
	python -m pip install -r requirements.txt
	```

## Code execution

Run the project-level script from the repository root:
```
python run_all.py -d .\data-1ml
```

For example, to run an individual experiment from the repository root:
```
python content_recommender.py -d .\data-1ml
python knn_recommender.py -d .\data-1ml
python -m two_towers.train_tt -d .\data-1ml
```

On macOS/Linux, activate the environment with `source .venv/bin/activate` and use `/` in paths, e.g. `python run_all.py -d ./data-1ml`.

### Two-tower recommender (self-implemented, Part II)

`run_all.py` trains and evaluates the two-tower model by default, using the configuration selected on validation (BPR loss, all side features, embedding 64, hidden 128, 40 epochs). This takes roughly 1.5 hours on a laptop CPU. Pass `--skip-two-tower` to skip it. The individual steps are:
```
python -m two_towers.train_tt -d ./data-1ml --seed-count 10   # seeds 42-51, 95% CIs
python -m two_towers.two_tower_metrics -d ./data-1ml          # paired test vs item-kNN, disaggregation, sparsity ablation
```
* Seeds: model seeds 42–51 for the multi-seed run; sparsity subsamples use seeds 0–2 (`utils.RAND_STATES`); bootstrap intervals use seed 42.
* `--seed-count 10` saves the final seed-51 model to `two_towers/confidence_interval_last_seed.pt`, which `two_tower_metrics` loads. If the file is missing, it is retrained automatically. The checkpoint is not committed because it contains MovieLens ratings.
* `two_tower_metrics` writes its CSV files to `report_and_results/two_tower_metrics/`.

To see a script's command-line options, add `-h`, for example:
```
python run_all.py -h
```