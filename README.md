# NLP Final Project

By Anish Shivram Gawde, Charles Ciampa, Daniel Alvarez

## Overview

For our NLP final project, the aim is to measure and understand the bias transfer when performing model distallation on an LLM.

## File Structure

We have split up the code into four folders: `data_prep`, `data`, `training`, `models`, and `analysis`.

The `data_prep` folder contains the code to obtain the log probilities of the labels from the teacher model. The label probalilites are then saved into a csv in the `data` folder.

The `data` folder will contain the data use to trained the student models. This is the data saved from the teachers model and used to teach the student models.

The `training` folder contains the code for the student models which are trained from the data from the teachers model, which can be found inside the `data` folder. After training the student models, be sure to save the models to the `models` folder to be later evaluated for bias.

The `models` folder contains the student models trained from the training folder. This folder is used by the code from the `training` and `analysis` folders.

The `analysis` folder contains the code where we analyze the bias in the student and teacher models. This also has a subfolder for the results from the IMDB testing dataset as well as a graphs folder with a variety of graphs created for the analysis.

The `training`, `models`, and `analysis` folders currently contain `temp.txt` files in each since they don't contain any code so far. When other files are added to these folders make sure to delete them as they only serve the purpose of giving the repository structure.

## Running Code for Preparing Distillation and Performing Analysis

This contains instructions on how to run the code for `data_prep/data_prep.ipynb` (This code gets the label probabilitys from the teacher model and saves it to the `data` folder) and `analysis/teacher_analysis.ipynb` (This code creates graphs and analyzes the results from the student models).

### Installing Packages

There are two methods described for getting the installation prepared. One is using pixi and the other is a manual installation. This installation is only usable for the `data_prep/data_prep.ipynb` and `analysis/teacher_analysis.ipynb` code.

#### Installation using Pixi

To install all the librarys, pixi was used for enviroment handling ([https://pixi.sh](https://pixi.sh)). Once you have pixi installed, go to parent directory containing this project and run `pixi install`. If you are using ubuntu this will install all of the librarys automatically.

#### Manual Install

If you are not able to install using pixi, the required packages are as follows.

- Python 3.11
- NumPy ≥1.26
- Pandas ≥2.2
- Scikit-learn ≥1.7.2
- Matplotlib
- Seaborn
- JupyterLab ≥4.0
- PyTorch ≥2.4.0 (CUDA 12.4)
- Transformers ≥4.38
- tqdm ≥4.65
<!-- - Datasets ≥2.14 -->

Feel free to install these packages using whatever enviroment manager you want to use.

### Running Distillation
Open file, `data_prep/data_prep.ipynb`. Then run all cells. If the installation was performed correctly with no errors, this should automatically download the training and testing data, as well as download the model. Additionally it will save the results to the `data/llama3.1` folder as `train.csv` and `test.csv`. Both of these will contain the probability the teacher model gave each of the labels.

### Running Analysis
Open file, `analysis/analysis.ipynb`, Then run all cells. If the installation was performed correctly with no errors, this should run all cells without error and additionally save all the graphs into the `graphs` subdirectory.