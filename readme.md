# BPI data challenge 2020

## Table of contents

- [Getting Started](#getting-started)
- [Requirements](#Requirements)
- [Running the code](#running-the-code)
- [Authors](#authors)
  - [Adam Broniewski](#adam-broniewski)
  - [Himanshu Choudhary](#himanshu-choudhary)
  - [Tejaswini Dhupad](#tejaswini-dhupad)
- [License](#license)

## Getting Started

This project contains the technical part of Predicting Rejected Declarations in the Travel Management System: BPI data challenge 2020

## Requirements
- Python 3.9
- pandas
- sklearn
- numpy
- matplotlib
- plotly
- pm4py
- seaborn
- tabulate

## Running the Code

These steps will run the full data-preparation, model building, prediction generations.

1. Install the dependencies:
    ```bash
    pip install -r requirments.txt
    ```
   ***Note***: *The scripts have been developed and tested with `Python 3.9`*
   
    ***Note 2***: If using ARM32 or ARM64 machine and an error is encountered installing pm4py, follow pm4py install instructions [here](https://pm4py.fit.fraunhofer.de/install-page).

2. Once all dependencies are installed, we can run the data preperation python file (default length 6 with complex encoding). The outputs are saved in `training_data/encodings` directory.
    ```bash
    python3 src/data_preparation.py
    ```
    **Expected Outcome -** 
    ```
    train test and validation split times are - 2018-09-17 11:34:42 , 2018-11-08 15:42:36 
    train, val and test count
    4020 861 861
    preparing training data for prefix length 6
    preparing test data for prefix length 6
    preparing val data for prefix length 6
    Done!
    ```

3. The below code will run our machine learning model (Default Decision Trees) on generated encodings. The outputs are saved in `results/` directory.

    ```bash
    python3 src/model_building_and_evaluation.py
    ```
    **Expected Outcome -** 

| Model                | Encoding   |   Trace Length |   F-score |   Precision |   Recall |   Accuracy |
|----------------------|------------|----------------|-----------|-------------|----------|------------|
| Decision Tree (Test) | complex    |              6 |  0.544791 |    0.263736 | 0.421053 |   0.651568 |
| Decision Tree (Val)  | complex    |              6 |  0.510082 |    0.307167 | 0.357143 |   0.576074 |
| Baseline             | complex    |              6 |  0.481256 |    0.26484       | 0.230159   |   0.587689 |




## Authors

#### Adam Broniewski [GitHub](https://github.com/abroniewski) | [LinkedIn](https://www.linkedin.com/in/abroniewski/) | [Website](https://adambron.com)
#### Himanshu Choudhary
#### Tejaswini Dhupad [GitHub](https://github.com/tejaswinidhupad) | [LinkedIn](https://www.linkedin.com/in/tejaswinidhupad/) 

## License

`APM_BPI_2020` is open source software [licensed as MIT][license].
