import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score, f1_score

def plot_data(X, y, title):
    """Plots images of the dataset. We are plotting only 64 images here.
    Parameters
    ----------
    X: list
        List of images
    y: list
        List of labels
    title: string
        Title of the plot
    """
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(64):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(X[i], cmap=plt.cm.binary, interpolation='nearest')
        # label the image with the target value
        ax.text(0, 7, str(y[i]))
    fig.suptitle(title, fontsize=16, y=-0)

def plot_accuracy_error_graph(i_values, acc_train, acc_val, acc_test):
	"""Plots accuracy vs n_estimator graph and error vs n_estimator graph.
	Parameters
	---------
	i_values: list
		list of indexes
	acc_train: list
		list of training set accuracies
	acc_val: list
		list of validation set accuracies
	acc_test: list
		list of testing set accuracies
	"""

	error_train = 1 - (np.array(acc_train)/100)
	error_val = 1 - (np.array(acc_val)/100)
	error_test = 1 - (np.array(acc_test)/100)

	f, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='row', figsize=(5,8)) # gets the current figure and then the axes

	ax1.plot(i_values, error_train, "-b", label="error_train")
	ax1.plot(i_values, error_val, "-g", label="error_val")
	ax1.set_title("Error Vs n_estimator")
	ax1.set_xlabel("n_estimator")
	ax1.set_ylabel("Error")
	ax1.legend()

	ax2.plot(i_values, acc_train, "-b", label="accuracy_train")
	ax2.plot(i_values, acc_val, "-g", label="accuracy_val")
	ax2.set_title("Accuracy Vs n_estimator")
	ax2.set_xlabel("n_estimator")
	ax2.set_ylabel("Accuracy")
	ax2.legend()

	plt.show()

def plot_roc_auc_curve(true_label, predictions):
	"""
	Plots roc_auc curve
	Parameter
	---------
	true_label: list
		List of target labels
	predictions: list
		List of predictions by the model
	"""

	train_auc = roc_auc_score(true_label, predictions)
	# summarize scores
	print('Train: ROC AUC=%.3f' % (train_auc))
	# calculate roc curves
	train_fpr, train_tpr, _ = roc_curve(true_label, predictions, pos_label=1)
	# plot the roc curve for the model
	plt.plot(train_fpr, train_tpr, linestyle='--', label='Train')
	# axis labels
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	# show the legend
	plt.legend()
	# show the plot
	plt.show()

def plot_precision_recall_curve(true_label, predictions):
	"""Plots precision recall curve.
	Parameter
	---------
	true_label: list
		List of true target labels
	predictions: list
		Predicted values by the model
	"""

	test_precision, test_recall, _ = precision_recall_curve(true_label, predictions, pos_label=1)
	test_f1, test_auc = f1_score(true_label, predictions), auc(test_recall, test_precision)
	# summarize scores
	print('Test: f1=%.3f auc=%.3f' % (test_f1, test_auc))
	# plot the precision-recall curves
	plt.plot(test_recall, test_precision, marker='.', label='Test')
	# axis labels
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	# show the legend
	plt.legend()
	# show the plot
	plt.show()
