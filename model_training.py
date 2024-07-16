from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTEN
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB  , BernoulliNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score , make_scorer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scipy.stats import randint
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
from sklearn.metrics import precision_score, recall_score
from handle_class_imbalaced import *

def label_encoding(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column])
    return df

def split_x_y(data,target_column_name):
    X = data.drop(target_column_name,axis=1)
    Y = data[target_column_name]
    return X,Y


def split_data(x,y):
    X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2)
    return X_train, X_test, Y_train, Y_test

def choose_classifier(X_train,Y_train,x_test,y_test):
    classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Support Vector Machine': SVC(),
    'k-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes1': GaussianNB(),
    'Naive Bayes2': BernoulliNB(),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }

    best_score = 0
    best_clf_name = None
    best_clf = None

    # Evaluate each classifier using cross-validation
    for name, clf in classifiers.items():
        model = clf
        model.fit(X_train,Y_train)
        pred = model.predict(x_test)
        pred = accuracy_score(y_test,pred)
        
        
        if pred > best_score:
            best_score = pred
            best_clf_name = name
            best_clf = clf
    return best_clf_name

def train_logistic_regression(x_train,y_train):
    log_reg = LogisticRegression()
    param_dist = {
        'C': uniform(loc=0, scale=4), 
        'max_iter':[50,100,250,500,750,1000,1500,2000],
        'penalty': ['l1', 'l2', 'elasticnet', 'none'], 
        'solver': ['lbfgs', 'liblinear', 'saga'], 
        'l1_ratio': uniform(loc=0, scale=1) 
    }
    
    random_search = RandomizedSearchCV(log_reg, param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)

    random_search.fit(x_train, y_train)
    best_log_reg = random_search.best_estimator_
    
    model = best_log_reg
    model.fit(x_train,y_train)
    return model, best_log_reg

def train_decision_tree_classifier(x_train,y_train):
    dt = DecisionTreeClassifier()

    param_dist = {
        'max_depth': randint(1, 40),
        'min_samples_split': randint(2, 40),
        'min_samples_leaf': randint(1, 40),
        "max_features": randint(1, 15),
        'criterion': ['gini', 'entropy']
    }

    # Create a RandomizedSearchCV instance
    random_search = RandomizedSearchCV(
        estimator=dt,
        param_distributions=param_dist,
        n_iter=100,  
        cv=2,  
    )

    random_search.fit(x_train, y_train)

    best_dt = random_search.best_estimator_
    model = best_dt
    model.fit(x_train,y_train)
    return model , best_dt

def train_randomForest_classifier(x_train,y_train):
    rf = RandomForestClassifier()
    
    param = {
    'n_estimators': [100, 200, 300,400,500],
    'max_depth': [None, 10,15, 20],
    'min_samples_split': [2, 5, 10,15],
    'min_samples_leaf': [1, 2, 4,6],
    'max_features': [ 'sqrt', 'log2']
}

    grid_search = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=param, cv=2, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    model = best_model
    model.fit(x_train,y_train)
    return model,best_model

def train_GradientBoosting_classifier(x_train,y_train):
    gbc = GradientBoostingClassifier()


    param_distributions = {  
    "learning_rate": sp_randFloat(),
    "subsample"    : sp_randFloat(),
    "n_estimators" : sp_randInt(100, 1000),
    "max_depth"    : sp_randInt(4, 10)                      
    }



    # Create a RandomizedSearchCV instance
    random_search = RandomizedSearchCV(
        estimator=gbc,
        param_distributions=param_distributions, 
        cv=2, 
        n_iter=10,
        n_jobs=-1,
    )


    random_search.fit(x_train, y_train)

    best_gbc = random_search.best_estimator_
    model = best_gbc
    model.fit(x_train, y_train)
    return model, best_gbc

def train_kNearest_Neighbors(x_train,y_train):
    knn = KNeighborsClassifier()

    param_dist = {
        'n_neighbors': randint(1, 10),  
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree', 'brute'],
        'leaf_size': randint(10, 100), 
        'p': [1, 2] 
    }

    random_search = RandomizedSearchCV(
        estimator=knn,
        param_distributions=param_dist,
        n_iter=100,
        cv=2,   
        n_jobs=-1 
    )

    random_search.fit(x_train, y_train)
    best_est = random_search.best_estimator_
    model = best_est
    model.fit(x_train,y_train)
    return model,best_est

def train_GaussianNB(x_train,y_train):
    param_distributions = {
    'var_smoothing': np.logspace(0, -9, num=100)
    }

    gnb = GaussianNB()
    random_search = RandomizedSearchCV(
        estimator=gnb,
        param_distributions=param_distributions,
        n_iter=100,
        scoring='accuracy',
        cv=5,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(x_train, y_train)

    best_estimator = random_search.best_estimator_
    model = best_estimator
    model.fit(x_train,y_train)
    return model,best_estimator

def train_bernouliNB(x_train,y_train):
    param_distributions = {
    'alpha': np.logspace(-3, 1, 50),
    'fit_prior': [True, False],
    'binarize': [0.0, 0.5, 1.0, 1.5, 2.0]
    }


    bnb = BernoulliNB()

    random_search = RandomizedSearchCV(
        estimator=bnb,
        param_distributions=param_distributions,
        n_iter=100,
        scoring='accuracy',
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(x_train, y_train)

    best_estimator = random_search.best_estimator_
    model = best_estimator
    model.fit(x_train,y_train)
    return model,best_estimator

def train_XGboost_classifier(x_train,y_train):
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    param_distributions = {
        'n_estimators': [100, 200, 300, 400, 500],  
        'max_depth': [3, 4, 5, 6, 7],  
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],  
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],  
        'reg_alpha': [0, 0.1, 0.5, 1.0], 
        'reg_lambda': [0, 0.1, 0.5, 1.0], 
    }

    random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_distributions,
        n_iter=50,
        scoring=make_scorer(accuracy_score),
        cv=2,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(x_train, y_train)
    best_xgb = random_search.best_estimator_
    model = best_xgb
    model.fit(x_train,y_train)
    return model,best_xgb

def train_svc(x_train,y_train):
    model = SVC()
    model.fit(x_train,y_train)
    return model

def train_model(name,x,y,x_train,y_train,x_test,y_test):
    ratio = check_imbalance(y)
    print(ratio)
    if name == 'Logistic Regression':
        model,best = train_logistic_regression(x_train,y_train)
        pred = model.predict(x_test)
        acc = accuracy_score(y_test,pred)
        pred_train = model.predict(x_train)
        acc_tr = accuracy_score(y_train,pred_train)
        precision_rf = precision_score(y_test, pred)
        recall_rf = recall_score(y_test, pred)
        if ratio >1.5:
            X_sampled,y_sampled = apply_oversampling(x,y)
            X_train, X_test, Y_train, Y_test = split_data(X_sampled,y_sampled)
            model = best
            model.fit(X_train,Y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(Y_test,pred)
            pred_train = model.predict(X_train)
            acc_tr = accuracy_score(Y_train,pred_train)
            precision_rf = precision_score(Y_test, pred)
            recall_rf = recall_score(Y_test, pred)
            
        return model , acc, acc_tr,precision_rf,recall_rf
    
    if name == 'Decision Tree':
        model,best = train_decision_tree_classifier(x_train,y_train)
        pred = model.predict(x_test)
        acc = accuracy_score(y_test,pred)
        pred_train = model.predict(x_train)
        acc_tr = accuracy_score(y_train,pred_train)
        precision_rf = precision_score(y_test, pred)
        recall_rf = recall_score(y_test, pred)
        if ratio >1.5:
            X_sampled,y_sampled = apply_oversampling(x,y)
            X_train, X_test, Y_train, Y_test = split_data(X_sampled,y_sampled)
            model = best
            model.fit(X_train,Y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(Y_test,pred)
            pred_train = model.predict(X_train)
            acc_tr = accuracy_score(Y_train,pred_train)
            precision_rf = precision_score(Y_test, pred)
            recall_rf = recall_score(Y_test, pred)
            hell = print("this")
            
        return model , acc, acc_tr,precision_rf,recall_rf
    
    if name == 'Random Forest':
        model,best = train_randomForest_classifier(x_train,y_train)
        pred = model.predict(x_test)
        acc = accuracy_score(y_test,pred)
        pred_train = model.predict(x_train)
        acc_tr = accuracy_score(y_train,pred_train)
        precision_rf = precision_score(y_test, pred)
        recall_rf = recall_score(y_test, pred)
        if ratio >1.5:
            X_sampled,y_sampled = apply_oversampling(x,y)
            X_train, X_test, Y_train, Y_test = split_data(X_sampled,y_sampled)
            model = best
            model.fit(X_train,Y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(Y_test,pred)
            pred_train = model.predict(X_train)
            acc_tr = accuracy_score(Y_train,pred_train)
            precision_rf = precision_score(Y_test, pred)
            recall_rf = recall_score(Y_test, pred)
        return model , acc, acc_tr,precision_rf,recall_rf
    
    if name == 'Gradient Boosting':
        model ,best = train_GradientBoosting_classifier(x_train,y_train)
        pred = model.predict(x_test)
        acc = accuracy_score(y_test,pred)
        pred_train = model.predict(x_train)
        acc_tr = accuracy_score(y_train,pred_train)
        precision_rf = precision_score(y_test, pred)
        recall_rf = recall_score(y_test, pred)
        if ratio >1.5:
            X_sampled,y_sampled = apply_oversampling(x,y)
            X_train, X_test, Y_train, Y_test = split_data(X_sampled,y_sampled)
            model = best
            model.fit(X_train,Y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(Y_test,pred)
            pred_train = model.predict(X_train)
            acc_tr = accuracy_score(Y_train,pred_train)
            precision_rf = precision_score(Y_test, pred)
            recall_rf = recall_score(Y_test, pred)
        return model , acc, acc_tr,precision_rf,recall_rf
    
    if name == 'k-Nearest Neighbors':
        model , best = train_kNearest_Neighbors(x_train,y_train)
        pred = model.predict(x_test)
        acc = accuracy_score(y_test,pred)
        pred_train = model.predict(x_train)
        acc_tr = accuracy_score(y_train,pred_train)
        precision_rf = precision_score(y_test, pred)
        recall_rf = recall_score(y_test, pred)
        if ratio >1.5:
            X_sampled,y_sampled = apply_oversampling(x,y)
            X_train, X_test, Y_train, Y_test = split_data(X_sampled,y_sampled)
            model = best
            model.fit(X_train,Y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(Y_test,pred)
            pred_train = model.predict(X_train)
            acc_tr = accuracy_score(Y_train,pred_train)
            precision_rf = precision_score(Y_test, pred)
            recall_rf = recall_score(Y_test, pred)
        return model , acc, acc_tr,precision_rf,recall_rf
    if name == 'Naive Bayes1':
        model, best = train_GaussianNB(x_train,y_train)
        pred = model.predict(x_test)
        acc = accuracy_score(y_test,pred)
        pred_train = model.predict(x_train)
        acc_tr = accuracy_score(y_train,pred_train)
        precision_rf = precision_score(y_test, pred)
        recall_rf = recall_score(y_test, pred)
        if ratio >1.5:
            X_sampled,y_sampled = apply_oversampling(x,y)
            X_train, X_test, Y_train, Y_test = split_data(X_sampled,y_sampled)
            model = best
            model.fit(X_train,Y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(Y_test,pred)
            pred_train = model.predict(X_train)
            acc_tr = accuracy_score(Y_train,pred_train)
            precision_rf = precision_score(Y_test, pred)
            recall_rf = recall_score(Y_test, pred)
        return model , acc, acc_tr,precision_rf,recall_rf
    
    if name == 'Naive Bayes2':
        model, best = train_bernouliNB(x_train,y_train)
        pred = model.predict(x_test)
        acc = accuracy_score(y_test,pred)
        pred_train = model.predict(x_train)
        acc_tr = accuracy_score(y_train,pred_train)
        precision_rf = precision_score(y_test, pred)
        recall_rf = recall_score(y_test, pred)
        if ratio >1.5:
            X_sampled,y_sampled = apply_oversampling(x,y)
            X_train, X_test, Y_train, Y_test = split_data(X_sampled,y_sampled)
            model = best
            model.fit(X_train,Y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(Y_test,pred)
            pred_train = model.predict(X_train)
            acc_tr = accuracy_score(Y_train,pred_train)
            precision_rf = precision_score(Y_test, pred)
            recall_rf = recall_score(Y_test, pred)
        return model , acc, acc_tr,precision_rf,recall_rf
    
    if name == 'XGBoost':
        model, best = train_XGboost_classifier(x_train,y_train)
        pred = model.predict(x_test)
        acc = accuracy_score(y_test,pred)
        pred_train = model.predict(x_train)
        acc_tr = accuracy_score(y_train,pred_train)
        precision_rf = precision_score(y_test, pred)
        recall_rf = recall_score(y_test, pred)
        if ratio >1.5:
            X_sampled,y_sampled = apply_oversampling(x,y)
            X_train, X_test, Y_train, Y_test = split_data(X_sampled,y_sampled)
            model = best
            model.fit(X_train,Y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(Y_test,pred)
            pred_train = model.predict(X_train)
            acc_tr = accuracy_score(Y_train,pred_train)
            precision_rf = precision_score(Y_test, pred)
            recall_rf = recall_score(Y_test, pred)
        return model , acc, acc_tr,precision_rf,recall_rf
    
    if name == 'Support Vector Machine':
        model = train_svc(x_train,y_train)
        pred = model.predict(x_test)
        acc = accuracy_score(y_test,pred)
        pred_train = model.predict(x_train)
        acc_tr = accuracy_score(y_train,pred_train)
        precision_rf = precision_score(y_test, pred)
        recall_rf = recall_score(y_test, pred)
        if ratio >1.5:
            X_sampled,y_sampled = apply_oversampling(x,y)
            X_train, X_test, Y_train, Y_test = split_data(X_sampled,y_sampled)
            model = train_svc(X_train,Y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(Y_test,pred)
            pred_train = model.predict(X_train)
            acc_tr = accuracy_score(Y_train,pred_train)
            precision_rf = precision_score(Y_test, pred)
            recall_rf = recall_score(Y_test, pred)
        return model , acc, acc_tr,precision_rf,recall_rf
    
    

