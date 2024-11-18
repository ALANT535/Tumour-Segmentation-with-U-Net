import React, { useState } from 'react';

const ImageUpload = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreview(URL.createObjectURL(file));
      setError(null);
      setPrediction(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedImage) {
      setError('Please select an image first');
      return;
    }

    const formData = new FormData();
    formData.append('image', selectedImage);

    try {
      const response = await fetch('http://localhost:8000/api/predict/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const data = await response.json();
      setPrediction(data.prediction);
    } catch (err) {
      setError('Failed to get prediction. Please try again.');
      console.error('Error:', err);
    }
  };

  return (
    <div className="image-upload">
      <h2>Image Classification</h2>
      <form onSubmit={handleSubmit} className="upload-form">
        <div className="file-upload">
          <input
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            className="hidden"
            id="image-input"
          />
          <label htmlFor="image-input" className="upload-label">
            Choose Image
          </label>
        </div>

        {preview && (
          <div className="image-preview-container">
            <img src={preview} alt="Preview" className="image-preview" />
          </div>
        )}

        {error && <div style={{ color: 'var(--error-color)' }}>{error}</div>}
        
        {prediction && (
          <div>
            <h3>Prediction Result:</h3>
            <p>{prediction}</p>
          </div>
        )}

        <button 
          type="submit" 
          className="submit-button"
          disabled={!selectedImage}
        >
          Get Prediction
        </button>
      </form>
    </div>
  );
};

export default ImageUpload;
