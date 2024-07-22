import React, { useState } from 'react';
import FileUploader from './FileUploader'; // Adjust the path as per your file structure
import { Container } from 'react-bootstrap';
import './App.css';

const App: React.FC = () => {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  // Handle file change callback
  const handleFileChange = (file: File) => {
    setUploadedFile(file);
    // You can perform additional actions here with the uploaded file
  };

  return (
    <Container className="App">
      <h1 className="text-center">Model Trainer</h1>
      <FileUploader onFileChange={handleFileChange} />
    </Container>
  );
};

export default App;
