from handle_null_values import *
from handle_outliers import *
from model_trainin_regression import *
from model_training_classification import *
from train_FNN import *

data , df = read_file('combined_adult.csv') 
data['age'] = data['age'].astype('float64')

df = remove_null_values(df)
data = handle_null_values(data, df)
data = handle_outliers(data,'income')
data['income'] = data['income'].str.strip()
data['income'] = data['income'].replace({'<=50K.': '<=50K', '>50K.': '>50K'})
data,mapping = label_encoding_regression(data)
X, Y = split_x_y(data, 'income')
model, acc, acc_tr, mse, mse_tr = fnn(len(Y.unique()),X,Y)
# X_train, X_test, y_train, y_test = split_data(X, Y)
# name = choose_regressor(X_train, y_train, X_test, y_test)
# model, acc, acc_tr, mse, mse_tr = train_regressor(name, X, Y, X_train, y_train, X_test, y_test)

print(acc)
print(acc_tr)
print(mse)
print(mse_tr)
