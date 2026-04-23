// frontend/src/components/Header.jsx
import React from 'react';

export default function Header() {
  return (
    <header className="header">
      <div className="header-container">
        <div className="header-left">
          <svg className="header-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
              d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
          </svg>
          <div>
            <h1 className="header-title">Diabetes Risk Predictor</h1>
            <p className="header-subtitle">AI-Powered Type 2 Diabetes Risk Assessment</p>
          </div>
        </div>
        <div className="header-badge">
          <svg className="header-badge-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
              d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
          <span>XGBoost 97.19%</span>
        </div>
      </div>
    </header>
  );
}
