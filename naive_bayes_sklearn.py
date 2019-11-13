import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

subjects = pd.read_csv('emails/dbworld_subjects_stemmed.csv')
subjects = subjects.drop(columns=['id'])

subjects_train = subjects.sample(frac=0.8, random_state=200)
subjects_test = subjects.drop(subjects_train.index)
