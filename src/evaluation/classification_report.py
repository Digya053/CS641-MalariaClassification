import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, classification_report, auc

def get_classification_report(true_label, predictions):
	"""Plots confusion matrix and finds precision, recall, f1-score, accuracy, macro avg, weighted avg.

	Parameters
	---------
	true_label: array
		The true label of the dataset
	predictions: array
		Labels predicted by the model
	"""
	confusion_matrix = metrics.confusion_matrix(true_label, predictions)
	TP, FP, FN, TN = confusion_matrix.ravel()

	plt.figure(figsize=(5,5))
	sns.heatmap(confusion_matrix, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = plt.cm.coolwarm);
	plt.ylabel('Actual label');
	plt.xlabel('Predicted label');
	all_sample_title = 'Accuracy Score: {0:.3f}'.format(accuracy_score(true_label, predictions))
	plt.title(all_sample_title, size = 15);

	target_names = ['0','1']
	print(classification_report(true_label, predictions, target_names=target_names))
