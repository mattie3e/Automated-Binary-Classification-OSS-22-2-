#PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/zizi-ctrl/Automated-Binary-Classification-OSS-22-2-.git

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sys

def load_dataset(dataset_path):
	data = pd.read_csv(dataset_path)
	
	return data

def dataset_stat(dataset_df):	
	df = dataset_df
	feature_num = df.shape[1] - 1
	class_zero_num = len(df.loc[df['target'] == 0])
	class_one_num = len(df.loc[df['target'] == 1])
	return feature_num, class_zero_num, class_one_num

def split_dataset(dataset_df, testset_size):
	data = dataset_df
	feature = data.drop(columns='target', axis=1)
	label = data['target']
	train_data, test_data, train_label, test_label = train_test_split(feature, label, test_size = testset_size)
	return train_data, test_data, train_label, test_label

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	model = DecisionTreeClassifier()
	model.fit(x_train, y_train.values.ravel())
 
	predict = model.predict(x_test)
	accuracy = accuracy_score(y_test, predict)
	precision = precision_score(y_test, predict)
	recall = recall_score(y_test, predict)
	
	return accuracy, precision, recall

def random_forest_train_test(x_train, x_test, y_train, y_test):
	model = RandomForestClassifier()
	model.fit(x_train, y_train.values.ravel())
 
	predict = model.predict(x_test)
	accuracy = accuracy_score(y_test, predict)
	precision = precision_score(y_test, predict)
	recall = recall_score(y_test, predict)
	
	return accuracy, precision, recall

def svm_train_test(x_train, x_test, y_train, y_test):
	SVM_pipe = make_pipeline(
		StandardScaler(),
		SVC()
	)	

	SVM_pipe.fit(x_train, y_train)
 
	predict = SVM_pipe.predict(x_test)
	accuracy = accuracy_score(y_test, predict)
	precision = precision_score(y_test, predict)
	recall = recall_score(y_test, predict)
	
	return accuracy, precision, recall

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)