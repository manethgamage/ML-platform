import React, { useState } from 'react';
import FileUploader from './FileUploader'; // Adjust the path as per your file structure
import FrontPage from './FrontPage'; // Ensure this path is correct
import './App.css';

const App: React.FC = () => {
  const [showFileUploader, setShowFileUploader] = useState(false);

  const handleStartClick = () => {
    setShowFileUploader(true);
  };

  return (
    <>
      {showFileUploader ? (
        <div className="app-container">
          <h1 className="text-center">Model Trainer</h1>
          <FileUploader onFileChange={(file) => console.log(file)} />
        </div>
      ) : (
        <FrontPage onStartClick={handleStartClick} />
      )}
    </>
  );
};

export default App;
