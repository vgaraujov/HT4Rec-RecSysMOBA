# Hierarchical Transformers for Sequential Recommendation: Application in MOBA Games

This repository contains the implementation of HT4Rec. 

<p align="center"> 
    <img src="images/model.png" width="500">
</p>

## Requirements
We include a `requirements.txt` file to make it easy to install dependencies. Run the following command in your environment:

```bash
$ pip install -r requirements.txt
```

## Usage

First, you have to create a folder `data` and `logs` in the root. 

Next, you need to download the datasets from the following [link](https://drive.google.com/drive/folders/1dxBzg2M3gTjdsdzbeGqDChclZC_7eMzx?usp=sharing) and copy it to the `data` folder.

Finally, run the following command to train the model on MOBA dataset:

```bash
$ python main.py
```
You can also run the model with the movies dataset:

```bash
$ python main_movies.py
```
