import sklearn.model_selection
from sklearn.datasets import fetch_openml
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")

X, y = fetch_openml(data_id=40691, as_frame=True, return_X_y=True)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf = clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)
print("RF Accuracy", sklearn.metrics.accuracy_score(y_test, y_hat))

from autosklearn.classification import AutoSklearnClassifier
from imblearn.over_sampling import SMOTE

smoteX, smoteY = SMOTE(random_state=307).fit_resample(X, y)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(smoteX, smoteY, random_state=42)

automl = AutoSklearnClassifier(time_left_for_this_task=300, 
	resampling_strategy='cv', 
	resampling_strategy_arguments={"folds":10})
automl.fit(X_train, y_train)
y_hat = automl.predict(X_test)
print("AutoML Accuracy", sklearn.metrics.accuracy_score(y_test, y_hat))
