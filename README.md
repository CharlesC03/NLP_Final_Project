# NLP Final Project

By Anish Shivram Gawde, Charles Ciampa, Daniel Alvarez

## Overview

For our NLP final project, the aim is to measure 

## File Structure

We have split up the code into four folders: `data_prep`, `data`, `training`, `models`, and `analysis`.

The `data_prep` folder contains the code to obtain the log probilities of the labels from the teacher model. The label probalilites are then saved into a csv in the `data` folder.

The `data` folder will contain the data use to trained the student models. This is the data saved from the teachers model and used to teach the student models.

The `training` folder contains the code for the student models which are trained from the data from the teachers model, which can be found inside the `data` folder. After training the student models, be sure to save the models to the `models` folder to be later evaluated for bias.

The `models` folder contains the student models trained from the training folder. This folder is used by the code from the `training` and `analysis` folders.

The `analysis` folder contains the code where we analyze the bias in the student and teacher models.

The `training`, `models`, and `analysis` folders currently contain `temp.txt` files in each since they don't contain any code so far. When other files are added to these folders make sure to delete them as they only serve the purpose of giving the repository structure.