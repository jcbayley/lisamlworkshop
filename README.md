# lisamlworkshop

LISA workshop comparing ML and traditional methods for signal detection

Click the button below to open the notebook in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jcbayley/lisamlworkshop/blob/main/lisa_workshop_detection.ipynb)

## Running locally

To run the tutorial locally, first clone the repository:

```bash
git clone https://github.com/jcbayley/lisamlworkshop.git
```

Then install the dependencies. We suggest using a virtual environment (`venv`) or `conda` environment. This will keep the installation separate from the system installation. See below from instructions.

Once you have installed the dependencies and activated the environment, run the following command in the base directory of the repository

```bash
jupter notebook
```

and then navigate to the jupyter notebook: `lisa_workshop_detection.ipynb`.

### Using conda

A conda environment can be created directly from the [environment file](https://github.com/jcbayley/lisamlworkshop/blob/main/environment.yml) found in the git repository:

```bash
conda env create -f environment.yml
```

The environment will be called `t1mldetection` and can be activated by running

```bash
conda activate t1mldetection
```

### Using virtual enviroment

Follow the instructions [here](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments) to create a virtual environment and then activate the environment. Once it is activated, the dependencies can be installed using the [`requirements.txt` file](https://github.com/jcbayley/lisamlworkshop/blob/main/requirements.txt) found in the git repository:

```bash
pip install -r requirements.txt
```