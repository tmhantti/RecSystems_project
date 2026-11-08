
# Project work for course Recommender Systems (DATA.ML.360), Autumn 2026

_Authors_ : Tatu Anttila @tmhantti , Arttu Kiviranta @akiviranta , Patrik Salmensaari @pasasalm

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

To run the code, type:
```
python run_all.py -d .\data-1ml
```

For command line options, type
```
python run_all.py -h
```
