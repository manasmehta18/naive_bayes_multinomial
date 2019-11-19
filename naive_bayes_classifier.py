import pandas as pd
import itertools as it


def group_class(dataset):
	dataset_class_0 = dataset[dataset['CLASS'] == 0]
	dataset_class_0 = dataset_class_0.drop(columns=['CLASS'])
	dataset_class_1 = dataset[dataset['CLASS'] == 1]
	dataset_class_1 = dataset_class_1.drop(columns=['CLASS'])

	return dataset_class_0, dataset_class_1


def train_naive_bayes(dataset_train, alpha):
	dataset_class_0, dataset_class_1 = group_class(dataset_train)

	prior_class_0 = (float(len(dataset_class_0.index)) / (len(dataset_class_0.index) + len(dataset_class_1.index)))
	prior_class_1 = (float(len(dataset_class_1.index)) / (len(dataset_class_0.index) + len(dataset_class_1.index)))

	prob_dataset_0 = (dataset_class_0.sum(axis=0, skipna=True) + alpha) / (
				dataset_class_0.values.sum() + len(dataset_class_0.columns))

	prob_dataset_1 = (dataset_class_1.sum(axis=0, skipna=True) + alpha) / (
				dataset_class_1.values.sum() + len(dataset_class_1.columns))

	return prior_class_0, prior_class_1, prob_dataset_0, prob_dataset_1


def predict_naive_bayes(prior_class_0, prior_class_1, prob_dataset_0, prob_dataset_1, dataset_test):
	dataset_test = dataset_test.drop(columns=['CLASS'])

	predictions = []

	for index, row in dataset_test.iterrows():
		row_prediction_0 = prior_class_0
		row_prediction_1 = prior_class_1
		i = 0
		for item in row:
			if item != 0:
				row_prediction_0 = row_prediction_0 * prob_dataset_0.iloc[i]
				row_prediction_1 = row_prediction_1 * prob_dataset_1.iloc[i]
			i = i + 1

		if row_prediction_0 >= row_prediction_1:
			predictions.append(0)
		else:
			predictions.append(1)

	return predictions


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

prior_subjects_class_0, prior_subjects_class_1, prob_subjects_0, prob_subjects_1 = train_naive_bayes(subjects_train, 1)

subjects_predictions = predict_naive_bayes(prior_subjects_class_0, prior_subjects_class_1, prob_subjects_0, prob_subjects_1, subjects_test)

subjects_act_class = []

for item in subjects_test.iloc[:, -1]:
	subjects_act_class.append(item)

subjects_f_mes = accuracy_naive_bayes(subjects_predictions, subjects_act_class)

print subjects_f_mes

bodies = pd.read_csv('emails/dbworld_bodies_stemmed.csv')
bodies = bodies.drop(columns=['id'])

bodies_train = bodies.sample(frac=0.8, random_state=50)
bodies_test = bodies.drop(bodies_train.index)

prior_bodies_class_0, prior_bodies_class_1, prob_bodies_0, prob_bodies_1 = train_naive_bayes(bodies_train, 1)

bodies_predictions = predict_naive_bayes(prior_bodies_class_0, prior_bodies_class_1, prob_bodies_0, prob_bodies_1, bodies_test)
bodies_act_class = []

for item in bodies_test.iloc[:, -1]:
	bodies_act_class.append(item)

bodies_f_mes = accuracy_naive_bayes(bodies_predictions, bodies_act_class)

print bodies_f_mes
