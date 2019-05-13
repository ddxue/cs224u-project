import pandas as pd
import csv
import numpy as np
import nltk
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import seaborn as sb

BINARY = '2-Way'
HEXARY = '6-Way'
REGRESS = 'Regress'
MODES = [BINARY, HEXARY, REGRESS]
MODE = MODES[0]

def get_batch(batch_size=1):
	pass # return a tuple (X, y)

def evaluate(y, yhat, mode=MODE):
	assert mode in MODES
	if mode == BINARY:
		pass
	elif mode == HEXARY:
		pass
	elif mode == REGRESS:
		pass

def load_data():
	pass