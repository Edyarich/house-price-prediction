# House price prediction

Housing costs demand a significant investment from both consumers and developers. And when it comes to planning a budget—whether personal or corporate—the last thing anyone needs is uncertainty about one of their biggets expenses. Provided ML model helps their customers by making predictions about realty prices so renters, developers, and lenders are more confident when they sign a lease or purchase a building.

The model predicts the sale price of each property based on its location, size, nearest objects, etc. The target variable is called `price_doc` in `train.csv`.

Link to the dataset: [kaggle](https://www.kaggle.com/c/sberbank-russian-housing-market/data)


### Install dependencies
```bash
python3 -m venv env
source env/bin/activate
python3 -m pip install -r requirements.txt
python -m ipykernel install --user --name=python3 --display-name "Python (house-price-pred)"
```

The last command is used for choosing an appropriate kernel while running `notebook.ipynb`

### Run scripts
- Training
```bash
    python train.py data/train.csv model.pkl 
```
- Inference

```bash
    python inference.py
```


```bash
    python test_inference.py
```
