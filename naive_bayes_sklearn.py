import pandas as pd
from sklearn.naive_bayes import MultinomialNB
import itertools as it


def accuracy_naive_bayes(dataset_predictions, dataset_act_class):
	TP, TN, FP, FN = 0.0, 0.0, 0.0, 0.0

	for p, a in it.izip(dataset_predictions, dataset_act_class):
		if p == 0 and a == 0:
			TP = TP + 1
		elif p == 1 and a == 1:
			TN = TN + 1
		elif p == 0 and a == 1:
			FP = FP + 1
		elif p == 1 and a == 0:
			FN = FN + 1

	pre = TP / (TP + FP)
	rec = TP / (TP + FN)

	f_mes = (2 * pre * rec) / (pre + rec)

	return f_mes


subjects = pd.read_csv('emails/dbworld_subjects_stemmed.csv')
subjects = subjects.drop(columns=['id'])

subjects_train = subjects.sample(frac=0.8, random_state=50)
subjects_test = subjects.drop(subjects_train.index)

X_subjects_train = subjects_train.iloc[:, :-1].values
y_subjects_train = subjects_train.iloc[:, -1].values

X_subjects_test = subjects_test.iloc[:, :-1].values
y_subjects_test = subjects_test.iloc[:, -1].values

clf_subjects = MultinomialNB()
clf_subjects.fit(X_subjects_train, y_subjects_train)

p_subjects = clf_subjects.predict(X_subjects_test)

subjects_f_mes = accuracy_naive_bayes(p_subjects, y_subjects_test)

print subjects_f_mes

bodies = pd.read_csv('emails/dbworld_bodies_stemmed.csv')
bodies = bodies.drop(columns=['id'])

bodies_train = bodies.sample(frac=0.8, random_state=50)
bodies_test = bodies.drop(bodies_train.index)

X_bodies_train = bodies_train.iloc[:, :-1].values
y_bodies_train = bodies_train.iloc[:, -1].values

X_bodies_test = bodies_test.iloc[:, :-1].values
y_bodies_test = bodies_test.iloc[:, -1].values

clf_bodies = MultinomialNB()
clf_bodies.fit(X_bodies_train, y_bodies_train)

p_bodies = clf_bodies.predict(X_bodies_test)

bodies_f_mes = accuracy_naive_bayes(p_bodies, y_bodies_test)

print bodies_f_mes




