import React, { useState, useRef } from 'react';
import { Container, Row, Col, Button, Form, Table } from 'react-bootstrap';
import './FileUploader.css';

interface FileUploaderProps {
  onFileChange: (file: File) => void;
}

const FileUploader: React.FC<FileUploaderProps> = ({ onFileChange }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [selectedColumnsToRemove, setSelectedColumnsToRemove] = useState<string[]>([]);
  const [selectedColumn, setSelectedColumn] = useState<string | null>(null);
  const [algorithm, setAlgorithm] = useState<string | null>(null);
  const [showAlgorithmSelection, setShowAlgorithmSelection] = useState<boolean>(false);
  const [tableData, setTableData] = useState<any[]>([]);
  const [showRemoveColumns, setShowRemoveColumns] = useState<boolean>(false);
  const [showSelectTargetColumn, setShowSelectTargetColumn] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const fileList = event.target.files;
    if (fileList && fileList.length > 0) {
      const selectedFile = fileList[0];
      onFileChange(selectedFile);
      setSelectedFile(selectedFile); // Store the selected file
      setSelectedFileName(selectedFile.name); // Set the selected file name
    }
  };

  const handleFormSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!selectedFile) {
      alert('Please choose a file.');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const result = await response.json();
        setColumns(result.columns); // Set the received column names
        setTableData(prevData => [...prevData, { fileName: selectedFileName, status: 'Uploaded', testAccuracy: '', trainingAccuracy: '', precision: '', recall: '', model: '' }]);
        setShowRemoveColumns(true); // Show the "Remove Columns" section
      } else {
        const errorResult = await response.json();
        alert(`Failed to upload file. Error: ${errorResult.error}`);
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('An error occurred while uploading the file.');
    }
  };

  const handleColumnRemoveChange = (column: string) => {
    setSelectedColumnsToRemove(prevState =>
      prevState.includes(column) ? prevState.filter(col => col !== column) : [...prevState, column]
    );
  };

  const handleRemoveSubmit = async () => {
    if (selectedColumnsToRemove.length === 0) {
      alert('Please select columns to remove.');
      return;
    }

    try {
      const response = await fetch('http://localhost:5000/remove-columns', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ columnsToRemove: selectedColumnsToRemove }),
      });

      if (response.ok) {
        const result = await response.json();
        setColumns(result.columns); // Update the columns after removal
        setSelectedColumnsToRemove([]); // Clear the selected columns to remove
        setShowRemoveColumns(false); // Hide the "Remove Columns" section
        setShowSelectTargetColumn(true); // Show the "Select Target Column" section
      } else {
        const errorResult = await response.json();
        alert(`Failed to remove columns. Error: ${errorResult.error}`);
      }
    } catch (error) {
      console.error('Error removing columns:', error);
      alert('An error occurred while removing the columns.');
    }
  };

  const handleRadioChange = (column: string) => {
    setSelectedColumn(column);
  };

  const handleSelectSubmit = async () => {
    if (!selectedColumn) {
      alert('Please select a column.');
      return;
    }

    try {
      const response = await fetch('http://localhost:5000/selected-columns', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ selectedColumn }),
      });

      if (response.ok) {
        setShowAlgorithmSelection(true);
        setSelectedColumn(null); // Clear selected column
        setColumns([]); // Clear columns list (if needed)
        setShowSelectTargetColumn(false); // Hide the "Select Target Column" section
      } else {
        const errorResult = await response.json();
        alert(`Failed to select column. Error: ${errorResult.error}`);
      }
    } catch (error) {
      console.error('Error selecting column:', error);
      alert('An error occurred while selecting the column.');
    }
  };

  const handleAlgorithmChange = (algorithm: string) => {
    setAlgorithm(algorithm);
  };

  const handleTrainSubmit = async () => {
    if (!algorithm) {
      alert('Please select an algorithm.');
      return;
    }

    setTableData(prevData => prevData.map(row => row.fileName === selectedFileName ? { ...row, status: 'Training' } : row));

    try {
      const response = await fetch('http://localhost:5000/selected-algorithm', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ selectedAlgorithm: algorithm }),
      });

      if (response.ok) {
        const result = await response.json();
        const downloadUrl = `http://localhost:5000/download-model?model_filename=${result.model_filename}`;

        setTableData(prevData => prevData.map(row =>
          row.fileName === selectedFileName
            ? { ...row, status: 'Completed', testAccuracy: result.testAccuracy, trainingAccuracy: result.trainingAccuracy, precision: result.precision, recall: result.recall, model: downloadUrl }
            : row
        ));
        resetForm(); // Reset the form after training and model download
      } else {
        const errorResult = await response.json();
        alert(`Failed to train model. Error: ${errorResult.error}`);
      }
    } catch (error) {
      console.error('Error training model:', error);
      alert('An error occurred while training the model.');
    }
  };

  const resetForm = () => {
    setSelectedFile(null);
    setSelectedFileName(null);
    setColumns([]);
    setSelectedColumnsToRemove([]);
    setSelectedColumn(null);
    setAlgorithm(null);
    setShowAlgorithmSelection(false);
    setShowRemoveColumns(false);
    setShowSelectTargetColumn(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = ''; // Clear file input
    }
  };

  return (
    <Container className="file-uploader-container">
      <Row className="justify-content-center">
        <Col>
          <Form className="file-uploader" onSubmit={handleFormSubmit}>
            <div className="d-flex align-items-center justify-content-between">
              <div className="file-chooser">
                <Form.Label htmlFor="fileUpload" className="file-uploader-label">Choose File:</Form.Label>
                <Form.Control 
                  id="fileUpload" 
                  type="file" 
                  onChange={handleFileChange} 
                  className="file-uploader-input" 
                  ref={fileInputRef}
                />
              </div>
              
            </div>
            <Button variant="primary" type="submit" className="mt-3 file-uploader-button">
              Upload
            </Button>
          </Form>
          {showRemoveColumns && (
            <div className="columns-list mt-3">
              <h5>Select Unwanted Columns to Remove:</h5>
              <Form>
                {columns.map((col, index) => (
                  <Form.Check 
                    key={index} 
                    type="checkbox" 
                    id={`remove-column-${index}`} 
                    label={col}
                    onChange={() => handleColumnRemoveChange(col)} 
                    checked={selectedColumnsToRemove.includes(col)}
                  />
                ))}
                <Form.Check
                  key="none"
                  type="checkbox"
                  id="remove-column-none"
                  label="None"
                  onChange={() => handleColumnRemoveChange("None")}
                  checked={selectedColumnsToRemove.includes("None")}
                />
              </Form>
              <Button 
                variant="primary" 
                className="mt-3" 
                onClick={handleRemoveSubmit}
                disabled={selectedColumnsToRemove.length === 0}
              >
                Remove
              </Button>
            </div>
          )}
          {showSelectTargetColumn && (
            <div className="columns-list mt-3">
              <h5>Select Target Column:</h5>
              <Form>
                {columns.map((col, index) => (
                  <Form.Check 
                    key={index} 
                    type="radio" 
                    id={`column-${index}`} 
                    name="columnRadio"
                    label={col}
                    onChange={() => handleRadioChange(col)} 
                    checked={selectedColumn === col}
                  />
                ))}
              </Form>
              <Button 
                variant="primary" 
                className="mt-3" 
                onClick={handleSelectSubmit}
                disabled={!selectedColumn}
              >
                Select
              </Button>
            </div>
          )}
          {showAlgorithmSelection && (
            <div className="algorithm-list mt-3">
              <h5>Select an Algorithm:</h5>
              <Form>
                {['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'Support Vector Machine', 'k-Nearest Neighbors', 'Naive Bayes1', 'Naive Bayes2', 'Naive Bayes3', 'XGBoost', 'Train With Best Algorithm'].map((alg, index) => (
                  <Form.Check 
                    key={index} 
                    type="radio" 
                    id={`algorithm-${index}`} 
                    name="algorithmRadio"
                    label={alg}
                    onChange={() => handleAlgorithmChange(alg)} 
                    checked={algorithm === alg}
                  />
                ))}
              </Form>
              <Button 
                variant="primary" 
                className="mt-3" 
                onClick={handleTrainSubmit}
                disabled={!algorithm}
              >
                Train
              </Button>
            </div>
          )}
        </Col>
      </Row>
      <Row className="justify-content-center">
        <Col>
          <Table striped bordered hover className="mt-3">
            <thead>
              <tr>
                <th>File Name</th>
                <th>Status</th>
                <th>Test Accuracy</th>
                <th>Training Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>Model</th>
              </tr>
            </thead>
            <tbody>
              {tableData.map((data, index) => (
                <tr key={index}>
                  <td>{data.fileName}</td>
                  <td>{data.status}</td>
                  <td>{data.testAccuracy}</td>
                  <td>{data.trainingAccuracy}</td>
                  <td>{data.precision}</td>
                  <td>{data.recall}</td>
                  <td>
                    {data.model ? (
                      <a href={data.model} download>Download</a>
                    ) : (
                      'N/A'
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </Table>
        </Col>
      </Row>
    </Container>
  );
};

export default FileUploader;
