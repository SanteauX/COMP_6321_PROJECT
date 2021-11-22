# Bixi Bike Rental Forecasting Project

A machine learning project comparing a variety of forecasting methods to predict daily bike-sharing system demand.

## Setup Instructions
* Step 1: The data is **not included** in the repository. In order to use the notebooks, put all original `csv` files into `data/raw` directory. The data can be downloaded [here](https://bixi.com/en/open-data).
* Step 2: Run the **preprocessing.ipynb** notebook in order to process the raw data and create `csv` files for use in the other notebooks. You can either run it cell by cell to follow the notebook and see how it works or just run all cells.
* Step 3: Install required dependencies if you don't have them:
  * **statsmodels**: `conda install -c conda-forge statsmodels`
  * **sktime**:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`conda install -c conda-forge sktime`
  * **fbprophet**:&nbsp;&nbsp;`conda install -c conda-forge pystan fbprophet`

## Notebook overview
1. **preprocessing.ipynb**: loads the raw data and processes it into overall daily trip history, as well as daily trip history for a particular station. History for the most popular station 'Metro Mount-Royal' is created by default and the last cell in the notebook can be used to extract any station history into a `csv` file.
2. **analysis.ipynb**: contains some basic analysis and visualization of the station data. Running it is optional.
3. **forecasting.ipynb**: model training, cross validation and testing is done here. Add your models at the end of this notebook.