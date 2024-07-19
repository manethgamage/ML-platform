from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
import xgboost as xgb
from scipy.stats import randint

def choose_regressor(X_train, y_train, X_test, y_test):
    regressors = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(),
        'Support Vector Machine': SVR(),
        'XGBoost': xgb.XGBRegressor(use_label_encoder=False, eval_metric='rmse')
    }

    best_score = float('-inf')
    best_regressor_name = None
    best_regressor = None

    # Evaluate each regressor
    for name, reg in regressors.items():
        model = reg
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = r2_score(y_test, pred)  # Use R² score for regression
        
        if score > best_score:
            best_score = score
            best_regressor_name = name
            best_regressor = reg

    return best_regressor_name

def train_with_linear_regression(x_train,y_train):
    model = LinearRegression()
    model.fit(x_train,y_train)
    return model

def train_lasso_regression(X_train, y_train):
    lasso = Lasso()
    param_dist = {
        'alpha': uniform(loc=0, scale=10),  # Regularization strength
        'max_iter': [50, 100, 250, 500, 750, 1000, 1500, 2000],
        'selection': ['random', 'cyclic']
    }
    
    random_search = RandomizedSearchCV(
        lasso, 
        param_distributions=param_dist, 
        n_iter=10000, 
        cv=2, 
        scoring='neg_mean_squared_error', 
        random_state=42, 
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    best_lasso = random_search.best_estimator_
    
    model = best_lasso
    model.fit(X_train, y_train)
    return model

def train_ridge_regression(X_train, y_train):
    ridge = Ridge()
    param_dist = {
        'alpha': uniform(loc=0, scale=10),  # Regularization strength
        'max_iter': [50, 100, 250, 500, 750, 1000, 1500, 2000],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']
    }
    
    random_search = RandomizedSearchCV(
        ridge, 
        param_distributions=param_dist, 
        n_iter=10000, 
        cv=2, 
        scoring='neg_mean_squared_error', 
        random_state=42, 
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    best_ridge = random_search.best_estimator_
    
    model = best_ridge
    model.fit(X_train, y_train)
    return model

def train_decision_tree_regressor(X_train, y_train):
    dt_regressor = DecisionTreeRegressor()
    param_dist = {
        'max_depth': [None] + list(range(1, 21)),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'max_features': ['auto', 'sqrt', 'log2', None],
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
    }

    random_search = RandomizedSearchCV(
        dt_regressor, 
        param_distributions=param_dist, 
        cv=5, 
        scoring='neg_mean_squared_error', 
        random_state=42, 
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    best_dt_regressor = random_search.best_estimator_
    
    model = best_dt_regressor
    model.fit(X_train, y_train)
    return model

def train_random_forest_regressor(X_train, y_train):
    rf_regressor = RandomForestRegressor()
    param_dist = {
        'n_estimators': randint(100, 1000,50),  # Number of trees in the forest
        'max_depth': [None] + list(range(1, 21)),  # Maximum depth of the tree
        'min_samples_split': randint(2, 20),  # Minimum number of samples required to split an internal node
        'min_samples_leaf': randint(1, 20),  # Minimum number of samples required to be at a leaf node
        'max_features': ['auto', 'sqrt', 'log2', None],  # Number of features to consider when looking for the best split
        'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
    }

    random_search = RandomizedSearchCV(
        rf_regressor, 
        param_distributions=param_dist,  
        cv=2, 
        scoring='neg_mean_squared_error', 
        random_state=42, 
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    best_rf_regressor = random_search.best_estimator_
    
    model = best_rf_regressor
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting_regressor(X_train, y_train):
    gb_regressor = GradientBoostingRegressor()
    param_dist = {
        'n_estimators': randint(50, 500,50),  # Number of boosting stages to be run
        'learning_rate': uniform(0.01, 0.3),  # Learning rate
        'max_depth': randint(1, 10),  # Maximum depth of the individual regression estimators
        'min_samples_split': randint(2, 20),  # Minimum number of samples required to split an internal node
        'min_samples_leaf': randint(1, 20),  # Minimum number of samples required to be at a leaf node
        'subsample': uniform(0.5, 0.5),  # Fraction of samples used for fitting the individual base learners
        'max_features': ['auto', 'sqrt', 'log2', None]  # Number of features to consider when looking for the best split
    }

    random_search = RandomizedSearchCV(
        gb_regressor, 
        param_distributions=param_dist, 
        n_iter=100, 
        cv=5, 
        scoring='neg_mean_squared_error', 
        random_state=42, 
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    best_gb_regressor = random_search.best_estimator_
    
    model = best_gb_regressor
    model.fit(X_train, y_train)
    return model

def train_svr(X_train, y_train):
    svr = SVR()
    param_dist = {
        'C': uniform(0.1, 10),
        'epsilon': uniform(0.01, 1)
    }

    random_search = RandomizedSearchCV(
        svr, 
        param_distributions=param_dist, 
        n_iter=100, 
        cv=2, 
        scoring='neg_mean_squared_error', 
        random_state=42, 
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    best_svr = random_search.best_estimator_
    
    model = best_svr
    model.fit(X_train, y_train)
    return model, best_svr

def train_xgb_regressor(X_train, y_train):
    xgb_reg = xgb.XGBRegressor(objective='reg:squarederror')
    
    param_dist = {
        'n_estimators': randint(50, 500,50),  # Number of boosting rounds
        'max_depth': randint(3, 10),  # Maximum tree depth
        'learning_rate': uniform(0.01, 0.3),  # Boosting learning rate
        'subsample': uniform(0.5, 0.5),  # Subsample ratio of the training instances
        'colsample_bytree': uniform(0.5, 0.5),  # Subsample ratio of columns when constructing each tree
        'gamma': uniform(0, 5),  # Minimum loss reduction required to make a further partition on a leaf node
        'reg_alpha': uniform(0, 1),  # L1 regularization term on weights
        'reg_lambda': uniform(0, 1)  # L2 regularization term on weights
    }
    
    random_search = RandomizedSearchCV(
        xgb_reg, 
        param_distributions=param_dist, 
        n_iter=100, 
        cv=2, 
        scoring='neg_mean_squared_error', 
        random_state=42, 
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    best_xgb = random_search.best_estimator_
    
    model = best_xgb
    model.fit(X_train, y_train)
    return model

def train_regressor(name, X, y, X_train, y_train, X_test, y_test):
    if name == 'Linear Regression':
        model = train_with_linear_regression(X_train, y_train)
    elif name == 'Ridge Regression':
        model = train_ridge_regression(X_train, y_train)
    elif name == 'Lasso Regression':
        model = train_lasso_regression(X_train, y_train)
    elif name == 'Decision Tree':
        model = train_decision_tree_regressor(X_train, y_train)
    elif name == 'Random Forest':
        model = train_random_forest_regressor(X_train, y_train)
    elif name == 'Gradient Boosting':
        model = train_gradient_boosting_regressor(X_train, y_train)
    elif name == 'Support Vector Machine':
        model = train_svr(X_train, y_train)
    elif name == 'XGBoost':
        model = train_xgb_regressor(X_train, y_train)
    else:
        raise ValueError(f"Unknown model name: {name}")

    pred = model.predict(X_test)
    acc = r2_score(y_test, pred)
    mse = mean_squared_error(y_test, pred)

    pred_train = model.predict(X_train)
    acc_tr = r2_score(y_train, pred_train)
    mse_tr = mean_squared_error(y_train, pred_train)

    return model, acc, acc_tr, mse, mse_tr






    