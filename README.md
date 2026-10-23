
# Project work for course Recommender Systems (DATA.ML.360), Autumn 2026

_Authors_ : Tatu Anttila, Arttu Kiviranta, Patrik Salmensaari

## Repository contents

### Subfolders

* *report_and_results* - contains report and results files 
* *plots*              - contains figures created by the code

## Notes

The same report is developed throughout all three parts: do not create a separate report for each part. Instead, extend and revise the existing report as the project progresses. All explanations, methodological decisions, justifications, analyses, and other report content requested in Parts 1–3 must be included in this report. 

## Project setup

The project was done using Python 3.12 using package manager venv. After cloning the repository, navigate to the repository main folder and take the following steps (in Windows console/Powershell): 

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
