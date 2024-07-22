from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import os
from handle_null_values import *
from handle_class_imbalaced import *
from handle_outliers import *
from model_training_classification import *
from train_FNN import *
import joblib
import pickle
import json
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import yagmail

app = Flask(__name__)
CORS(app)



# Global variables
data_set = None
file_name = None
df_set = None
selected_column = None
selected_algorithm = None
# user_email = None

# Function to check allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to read column names from a file-like object
def read_column_names(file):
    try:
        df = pd.read_csv(file)
        return df.columns.tolist()
    except Exception as e:
        print(f"Error reading columns: {e}")
        return []

@app.route('/upload', methods=['POST'])
def upload_file():
    global data_set, df_set, user_email, file_name

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    user_email = request.form.get('email')

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File extension not allowed. Please upload a .csv file.'}), 400
    file.name = file_name
    data, df = read_file(file)
    data_set = data.copy()
    df_set = df.copy()
    columns = columns_name(data)

    if not columns:
        return jsonify({'error': 'Failed to read columns from file.'}), 500

    return jsonify({'message': 'File uploaded successfully', 'columns': columns}), 200

@app.route('/remove-columns', methods=['POST'])
def remove_columns():
    global data_set, df_set
    data = request.get_json()
    columns_to_remove = data.get('columnsToRemove', [])

    if 'None' in columns_to_remove:
        return jsonify({'message': 'No columns removed', 'columns': columns_name(data_set)}), 200

    data_set.drop(columns=columns_to_remove, inplace=True)
    df_set.drop(columns=columns_to_remove, inplace=True)
    columns = columns_name(data_set)

    return jsonify({'message': 'Columns removed successfully', 'columns': columns}), 200

@app.route('/selected-columns', methods=['POST'])
def select_column():
    global selected_column
    data = request.get_json()
    selected_column = data.get('selectedColumn')

    if not selected_column:
        return jsonify({'error': 'No column selected'}), 400

    print(f"Selected column: {selected_column}")

    return jsonify({'message': 'Column selected successfully'}), 200

def train_model_with_algorithm(data, df, algorithm, column):
    data['age'] = data['age'].astype('float64')
    df = remove_null_values(df)
    data = handle_null_values(data, df)
    data = handle_outliers(data, column)
    data['income'] = data['income'].str.strip()
    data['income'] = data['income'].replace({'<=50K.': '<=50K', '>50K.': '>50K'})
    data, mapping = label_encoding(data, column)
    X, Y = split_x_y(data, column)
    X_train, X_test, y_train, y_test = split_data(X, Y)

    if algorithm == 'Train With Best Algorithm':
        name = choose_classifier(X_train, y_train, X_test, y_test)
        model, acc, tr_acc, precision_rf, recall_rf = train_model(name, X, Y, X_train, y_train, X_test, y_test)
    elif algorithm == 'Neural Network':
        model,scaler, acc, tr_acc, precision_rf, recall_rf = fnn(len(Y.unique()),X,Y)
        
        
    else:
        model, acc, tr_acc, precision_rf, recall_rf = train_model(algorithm, X, Y, X_train, y_train, X_test, y_test)

    if algorithm == 'Neural Network':
        model_filename = 'trained_model.h5'
        model.save(model_filename)
        send_email_fnn(model_filename,scaler,mapping,algorithm)
    else:   
        model_filename = 'trained_model.pkl'
        joblib.dump(model, model_filename)
        send_email( model_filename, mapping,algorithm)

    return model_filename, acc, tr_acc, precision_rf, recall_rf

def send_email( model_filename, class_mappings,algorithm):
    yag = yagmail.SMTP('manethgamage654@gmail.com', 'xhwe owef faow gzhp')
    
    subject = "Model Training Completed"
    body = f"""
    Hello,

    Your model training is complete with {algorithm} algorithm. The trained model has been saved as {model_filename}.

    Class mappings for encoded columns are as follows:
    {class_mappings}

    Best regards,
    Your Team
    """
    yag.send(to='denidugamage3@gmail.com', subject=subject, contents=body)

def send_email_fnn(model_filename, scaler, class_mappings, algorithm):
    yag = yagmail.SMTP('manethgamage654@gmail.com', 'xhwe owef faow gzhp')
    
    subject = "Model Training Completed"
    body = f"""
    Hello,

    Your model training is complete with the {algorithm} algorithm. The trained model has been saved as {model_filename}.

    Class mappings for encoded columns are as follows:
    {class_mappings}

    Best regards,
    Your Team
    """
    
    # Serialize the scaler and class mappings
    scaler_bytes = pickle.dumps(scaler)
    
    # Attach the serialized scaler and class mappings
    attachments = [
        (f"{model_filename}", 'application/octet-stream', model_filename),
        ('scaler.pkl', 'application/octet-stream', scaler_bytes)
    ]
    
    yag.send(to='denidugamage3@gmail.com', subject=subject, contents=body, attachments=attachments)


@app.route('/selected-algorithm', methods=['POST'])
def select_algorithm():
    global selected_algorithm
    data = request.get_json()
    selected_algorithm = data.get('selectedAlgorithm')

    if not selected_algorithm:
        return jsonify({'error': 'No algorithm selected'}), 400

    print(f"Selected algorithm: {selected_algorithm}")

    model_filename, acc, tr_acc, precision_rf, recall_rf = train_model_with_algorithm(data_set, df_set, selected_algorithm, selected_column)

    return jsonify({
        'message': 'Algorithm selected and model trained successfully',
        'model_filename': model_filename,
        'testAccuracy': acc,
        'trainingAccuracy': tr_acc,
        'precision': precision_rf,
        'recall': recall_rf
    }), 200

@app.route('/download-model', methods=['GET'])
def download_model():
    model_filename = request.args.get('model_filename')
    if model_filename and os.path.exists(model_filename):
        return send_file(model_filename, as_attachment=True)
    else:
        return jsonify({'error': 'Model file not found'}), 404

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
