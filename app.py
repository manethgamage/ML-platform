import builtins
import tempfile
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
import io
import yagmail

app = Flask(__name__)
CORS(app)

# Global variables
user_email = None
data_set = None
file_name = None
df_set = None
selected_column = None
selected_algorithm = None

# Function to check allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}

@app.route('/submit-email', methods=['POST'])
def submit_email():
    global user_email
    data = request.get_json()
    email = data.get('email')
    user_email = email
    print(user_email)
    

    if not email:
        return jsonify({'error': 'Email is required'}), 400

    # Handle email here (e.g., save to database, send confirmation email, etc.)
    print(f"Received email: {email}")

    return jsonify({'message': 'Email received successfully'}), 200


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
    global data_set, df_set, file_name

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    # print(file.filename)
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File extension not allowed. Please upload a .csv file.'}), 400
    
    # print(file.name)

    file_name = file.filename
    print(file_name)
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
    global file_name, user_email
    
    df = remove_null_values(df)
    data = handle_null_values(data, df)
    data = handle_outliers(data, column)
    data, mapping = label_encoding(data, column)
    X, Y = split_x_y(data, column)
    X_train, X_test, y_train, y_test = split_data(X, Y)

    model_content = None
    file_extension = 'pkl'

    if algorithm == 'Train With Best Algorithm':
        name = choose_classifier(X_train, y_train, X_test, y_test)
        model, acc, tr_acc, precision_rf, recall_rf = train_model(name, X, Y, X_train, y_train, X_test, y_test)
    elif algorithm == 'Neural Network':
        model, scaler, acc, tr_acc, precision_rf, recall_rf = fnn(len(Y.unique()), X, Y)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as temp_file:
            model.save(temp_file.name, save_format='h5')
            with builtins.open(temp_file.name, 'rb') as file:
                model_content = file.read()
            send_email_fnn(user_email,model_content,file_name, scaler, mapping, algorithm)
        file_extension = 'h5'
    else:
        model, acc, tr_acc, precision_rf, recall_rf = train_model(algorithm, X, Y, X_train, y_train, X_test, y_test)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
            joblib.dump(model, temp_file.name)
            with builtins.open(temp_file.name, 'rb') as file:
                model_content = file.read()
            send_email(user_email,model_content,file_name, mapping, algorithm)

    return model_content, acc, tr_acc, precision_rf, recall_rf, file_extension

def send_email(user_mail,model_content,file_name, class_mappings, algorithm):
    yag = yagmail.SMTP('manethgamage654@gmail.com', 'xhwe owef faow gzhp')
    
    subject = f"Model Training Completed {file_name}"
    body = f"""
    Hello,

    Your model training is complete with {algorithm} algorithm.

    Class mappings for encoded columns are as follows:
    {class_mappings}

    Best regards,
    Your Team
    """
    
    model_file = io.BytesIO(model_content)
    model_file.name = 'trained_model.pkl'
    
    yag.send(to=user_mail, subject=subject, contents=body, attachments=[model_file])

def send_email_fnn(user_mail,model_content,file_name, scaler, class_mappings, algorithm):
    yag = yagmail.SMTP('manethgamage654@gmail.com', 'xhwe owef faow gzhp')
    
    subject = f"Model Training Completed {file_name}"
    body = f"""
    Hello,

    Your model training is complete with the {algorithm} algorithm.

    Class mappings for encoded columns are as follows:
    {class_mappings}

    Best regards,
    Your Team
    """
    
    # Serialize the scaler
    scaler_bytes = pickle.dumps(scaler)
    scaler_file = io.BytesIO(scaler_bytes)
    scaler_file.name = 'scaler.pkl'
    
    model_file = io.BytesIO(model_content)
    model_file.name = 'trained_model.h5'
    
    attachments = [model_file, scaler_file]
    
    yag.send(to=user_mail, subject=subject, contents=body, attachments=attachments)

@app.route('/selected-algorithm', methods=['POST'])
def select_algorithm():
    global selected_algorithm, file_name
    data = request.get_json()
    selected_algorithm = data.get('selectedAlgorithm')

    if not selected_algorithm:
        return jsonify({'error': 'No algorithm selected'}), 400

    print(f"Selected algorithm: {selected_algorithm}")

    model_content, acc, tr_acc, precision_rf, recall_rf, file_extension = train_model_with_algorithm(data_set, df_set, selected_algorithm, selected_column)

    return jsonify({
        'message': 'Algorithm selected and model trained successfully',
        'model_content': model_content.decode('latin1'),  # Encode the bytes for JSON transmission
        'testAccuracy': acc,
        'trainingAccuracy': tr_acc,
        'precision': precision_rf,
        'recall': recall_rf,
        'file_extension': file_extension
    }), 200

@app.route('/download-model', methods=['POST'])
def download_model():
    model_content = request.json.get('model_content')
    file_extension = request.json.get('file_extension')
    model_content_bytes = model_content.encode('latin1')
    model_filename = f'trained_model.{file_extension}'

    return send_file(io.BytesIO(model_content_bytes), as_attachment=True, download_name=model_filename)

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
