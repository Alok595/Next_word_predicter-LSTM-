import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [text, setText] = useState("");
  const [prediction, setPrediction] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handlePredict = async () => {
    if (!text.trim()) {
      setPrediction("");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await axios.post("http://127.0.0.1:8000/predict", {
        text: text,
      });

      setPrediction(response.data.prediction);
    } catch (err) {
      setError("Failed to connect to API");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>Next Word Prediction</h1>

      <textarea
        rows="4"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter some text..."
      />

      <button onClick={handlePredict}>Predict Next Word</button>

      {loading && <p>Predicting...</p>}

      {error && <p className="error">{error}</p>}

      {prediction && (
        <div className="result">
          <h2>Predicted Word:</h2>
          <span>{prediction}</span>
        </div>
      )}
    </div>
  );
}

export default App;
