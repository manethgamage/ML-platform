import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from handle_class_imbalaced import *
from model_training_classification import *
from tensorflow.keras.utils import to_categorical



def fnn(output_dim,x,y):
    x, y = apply_oversampling(x, y)
    scaler = StandardScaler()
    x= scaler.fit_transform(x)
    
    x_train, x_test, y_train, y_test = split_data(x, y)
    y_train = to_categorical(y_train, num_classes=output_dim)
    y_test = to_categorical(y_test, num_classes=output_dim)
    
    model = Sequential([
        Dense(512, input_dim=x_train.shape[1], activation='relu'),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(32, activation='relu'),
        Dropout(0.1),
        
        Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(16, activation='relu'),
        Dropout(0.1),  
        Dense(output_dim, activation='softmax') 
])
    
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
    y_pred_train = model.predict(x_train).argmax(axis=1)
    y_pred_test = model.predict(x_test).argmax(axis=1)
    y_train_true = y_train.argmax(axis=1)
    y_test_true = y_test.argmax(axis=1)
    
    train_accuracy = accuracy_score(y_train_true, y_pred_train)
    test_accuracy = accuracy_score(y_test_true, y_pred_test)
    precision = precision_score(y_test_true, y_pred_test, average='weighted')
    recall = recall_score(y_test_true, y_pred_test, average='weighted')
    
    return model, scaler, test_accuracy, train_accuracy, precision, recall