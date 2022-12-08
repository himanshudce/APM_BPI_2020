# BPI data challenge 2020

![Current Version](https://img.shields.io/badge/version-v0.5-blue)
![GitHub contributors](https://img.shields.io/github/contributors/abroniewski/Child-Wasting-Prediction)
![GitHub stars](https://img.shields.io/github/stars/abroniewski/README-Template?style=social)
![GitHub activity](https://img.shields.io/github/commit-activity/w/abroniewski/Child-Wasting-Prediction?logoColor=brightgreen)

## Table of contents

- [Getting Started](#getting-started)
- [Requirements](#Requirements)
- [Running the code](#running-the-code)
- [Development](#Development)
- [Authors](#authors)
  - [Adam Broniewski](#adam-broniewski)
  - [Himanshu Choudhary](#himanshu-choudhary)
  - [Tejaswini Dhupad](#tejaswini-dhupad)
- [License](#license)

## Getting Started

This project contains the technical part of Data challenge 3: Improving Child Wasting Prediction for Zero Hunger Labs.

The project code follows the structure below:

```
	Child-Wasting-Prediction
	├── README.md
	├── LICENSE.md
	├── requirments.txt
	└── src
		├── all executbale scripts
	└── notebooks
		├── contains jupyter notebook
	└── results
		├── contains all results in csv files
	└── data
		└── contains raw and processed data
```
GitHublink - https://github.com/himanshudce/APM_BPI_2020

## Requirements
- Python 3.9
- pandas
- sklearn
- numpy
- matplotlib
- plotly

## Running the Code

These steps will run the full data-preparation, model building, prediction generation, and results comparison using the data provided in [data](https://github.com/abroniewski/Child-Wasting-Prediction.git/data).

1. Unzip the folder Child-Wasting-Prediction.zip and switch to the folder
    ```bash
    cd Child-Wasting-Prediction
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirments.txt
    ```
   ***Note***: *The scripts have been developed and tested with `Python 3.9`*

3. Once all dependencies are installed, we can run the ZHL Baseline model file **(model_1)**. The output his saved in `results/` directory. Running this script can take 15-20min. For convenience, the output has been saved already in `results/`.
    ```bash
    python3 src/Baseline.py
    ```
    **Expected Outcome -** 
    ```
    no. of trees: 74
    max_depth: 6
    columns: ['district_encoded']
    0.05629900026844118 0.849862258953168
    ```

4. The below code will extract the features from raw conflict data. The output data is saved in the `data/acled/` directory.

    ```bash
    python3 src/feature_engineering.py
    ```

5. The below code will run the baseline model training with adjusted date preparation (Model 2) and adjusted data preparation with new features (Model 3). The results will be saved to the `results/` directory.  
    - For model_2
    ```bash
    python3 src/dc3_main.py model_2
    ```
    - For model_3
    ```bash
    python3 src/dc3_main.py model_3
    ```
    ***Note***: *The parameters 'model_2' and 'model_3' can be passed with the above script to generate **the baseline with adjusted preparation** and **combined conflict data model** results respectively. By default, it is running on our combined conflict data model (model_3)*.  

    **Expected Outcome with model_2** 
    ```
    Total no of district after preproecssing are - 55 
    number of observations for training are - 275 and for testing are - 110 
    MAE(Mean Absolute Error) score for model_2 model on training data is - 0.021391660328296973
    MAE(Mean Absolute Error) score for model_2 model on test data is - 0.0512394083425881 
    ```
    **Expected Outcome with model_3** 
    ```
    Total no of district after preproecssing are - 55 
    number of observations for training are - 275 and for testing are - 110 
    MAE(Mean Absolute Error) score for model_3 model on training data is - 0.021117145120009326
    MAE(Mean Absolute Error) score for model_3 model on test data is - 0.05013554272083966 
    ```

6.  The below code combines all the results from model_1 (ZHL Baseline), model_2 (Adjusted Baseline) and model_3 (conflict data combined model) for comparison. 
    ```bash
    python3 src/combine_results.py
    ```
    
The results of all 3 models are saved to `results/combined_model_1_2_3_testresults.csv`. The CSV shows the actual next_prevalence for each district with the predictions of each model. model_2 and model_3 have values for more districts than model_1 due to the changes in data preparation. 

## Development

The objective of this project is to work with various ****stakeholders**** to understand their needs and the impact modeling choices have on them. Additionally, the design choices are assessed through a lens of **ethical impact**.

The objective of the **data analytics model** to explore whether a better (more accurate or more generally applicable) forecasting model for predicting child watage can be developed, by researching one of the following two questions:
1. Is the quality of the additional data sources sufficient to improve or expand the existing GAM forecasting model? Are there additional, public data sources that allow you to improve or expand the existing GAM forecasting model?
2. Are there other techniques, different than additional data sources, that would lead to an improved GAM forecasting model on the data used in the development of the original GAM forecasting model?


## Authors

#### Adam Broniewski [GitHub](https://github.com/abroniewski) | [LinkedIn](https://www.linkedin.com/in/abroniewski/) | [Website](https://adambron.com)
#### Himanshu Choudhary
#### Tejaswini Dhupad [GitHub](https://github.com/tejaswinidhupad) | [LinkedIn](https://www.linkedin.com/in/tejaswinidhupad/) 

## License

`APM_BPI_2020` is open source software [licensed as MIT][license].
