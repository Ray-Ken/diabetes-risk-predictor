// frontend/src/components/Footer.jsx
import React from 'react';

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-content">
        <p className="footer-main">
          © 2026 Diabetes Risk Predictor - MSc Applied Data Science Project
        </p>
        <p className="footer-stats">
          XGBoost Model | 97.19% Accuracy | MCC: 0.81 | Trained on 100,520 samples
        </p>
        <p className="footer-disclaimer">
          ⚠️ This tool is for informational purposes only. Always consult a healthcare professional.
        </p>
      </div>
    </footer>
  );
}
