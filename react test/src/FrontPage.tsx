import React, { useState } from 'react';
import './FrontPage.css';

interface FrontPageProps {
  onStartClick: () => void;
}

const FrontPage: React.FC<FrontPageProps> = ({ onStartClick }) => {
  const [email, setEmail] = useState<string>('');
  const [errorMessage, setErrorMessage] = useState<string>('');

  const handleEmailChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setEmail(e.target.value);
  };

  const handleStartClick = async () => {
    if (!email) {
      setErrorMessage('Please enter a valid email address.');
    } else {
      setErrorMessage('');

      // Send the email address to the Flask backend
      try {
        await fetch('http://localhost:5000/submit-email', { // Adjust URL as needed
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ email }),
        });

        // Trigger the onStartClick callback
        onStartClick();
      } catch (error) {
        console.error('Error submitting email:', error);
        setErrorMessage('Failed to submit email. Please try again.');
      }
    }
  };

  return (
    <div className="front-page">
      <div className="overlay">
        <div className="content">
          <h1>ModelForge</h1>
          <p className="main-tagline">Unleash the Power of Machine Learning. Innovate. Achieve More.</p>
          <p className="sub-tagline">Free to use. Easy to try. Just upload your data and let ModelForge do the rest.</p>
          <div className="email-signup">
            <input
              type="email"
              placeholder="Enter your email"
              className="email-input"
              value={email}
              onChange={handleEmailChange}
            />
            <button className="start-button" onClick={handleStartClick}>Start Now</button>
          </div>
          {errorMessage && <p className="error-message">{errorMessage}</p>}
        </div>
      </div>
    </div>
  );
};

export default FrontPage;
