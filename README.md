## Deep Learning of Mixture Opacities
A PyTorch deep learning model for predicting nuclear hydrodynamic mixture opacities (Rosseland and Planck) from composition and thermodynamic properties. 

### Architecture
We strive to achieve maximum accuracy and efficiency in our design.

- This neural network has 5 inputs: (Mixture % of H, He, Al), Temperature, Density
- There are four hidden dense layers (configurable)
- The output is the Rosseland and Planck opacity
- There are additional optimizations and parallel predictions

## Setup
- Clone the repository
    - `git clone https://github.com/eblissss/mixture-opacities.git`
    - `cd mixture-opacities`
- Have Python Installed and use virtual environment
    - Run `python3 -m venv .venv` and `source venv/bin/activate` (Linux/Mac)
- Then install requirements
    - Run `pip install -r requirements.txt`


## Fetching Data
We fetch the data from [LANL TOPS Opacities](https://aphysics2.lanl.gov/)

- Install requirements (if not already done)
    - Run `pip install selenium numpy webdriver-manager`
- Edit settings in `fetch_data.py` if needed
- Run `python fetch_data.py` to execute fetching
- Check `opacity_data.csv` for results!


## Training the Model
Feel free to modify the configuration, including hyperparameters
- Model architecture configured in `model.py`

Simply run `python src/train.py` to train!
- Outputs `best.pth` as the best checkpoint with full state
- Outputs `model.pth` as inference-ready model
- Also includes TensorBoard logs in `runs/` directory

## Making Predictions
Modify `predict.py` to load your data as a Numpy array (shape (n, 5))
Simply run `python src/predict.py`
- Predictions saved to `predictions.csv` (combined inputs and predictions)