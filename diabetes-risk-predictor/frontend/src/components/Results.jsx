// frontend/src/components/Results.jsx
import React, { useState } from 'react';
import { submitFeedback, downloadPDF } from '../services/api';

export default function Results({ results, loading }) {
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [downloadingPDF, setDownloadingPDF] = useState(false);

  const handleFeedback = async (feedbackType) => {
    try {
      await submitFeedback({
        prediction_id: Date.now(),
        feedback: feedbackType
      });
      setFeedbackSubmitted(true);
    } catch (error) {
      console.error('Feedback error:', error);
    }
  };

  const handleDownloadPDF = async () => {
    if (!results?.inputData) return;
    
    setDownloadingPDF(true);
    try {
      const blob = await downloadPDF(results.inputData);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `diabetes_risk_report_${Date.now()}.pdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('PDF download error:', error);
      alert('Failed to generate PDF. Please try again.');
    } finally {
      setDownloadingPDF(false);
    }
  };

  if (loading) {
    return (
      <div className="results-loading">
        <div className="loading-spinner"></div>
        <p>Analyzing your health data with AI...</p>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="results-empty">
        <div>
          <svg className="results-empty-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
              d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
          </svg>
          <h3 style={{ fontSize: '1.25rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
            Your Results Will Appear Here
          </h3>
          <p>Fill out the form on the left to get your personalized diabetes risk assessment.</p>
        </div>
      </div>
    );
  }

  const getRiskClass = () => {
    if (results.risk_category === 'Low Risk') return 'low';
    if (results.risk_category === 'Medium Risk') return 'medium';
    return 'high';
  };

  const getPriorityClass = (priority) => {
    return priority.toLowerCase();
  };

  return (
    <div>
      {/* Risk Score */}
      <div className={`risk-box ${getRiskClass()}`}>
        <div className="risk-header">
          <h3 className="risk-title">Risk Assessment</h3>
          <svg className="risk-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            {results.risk_category === 'High Risk' && (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            )}
            {results.risk_category === 'Medium Risk' && (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            )}
            {results.risk_category === 'Low Risk' && (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            )}
          </svg>
        </div>
        <div className="risk-score">{results.risk_percentage}</div>
        <div className="risk-category">{results.risk_category}</div>
        <div className="risk-bar">
          <div 
            className={`risk-bar-fill ${getRiskClass()}`}
            style={{ width: `${results.risk_score * 100}%` }}
          ></div>
        </div>
      </div>

      {/* Download PDF Button */}
      <button onClick={handleDownloadPDF} className="btn btn-download" disabled={downloadingPDF}>
        {downloadingPDF ? (
          <>
            <svg className="btn-icon spinner" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Generating PDF...
          </>
        ) : (
          <>
            <svg className="btn-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Download Full Report (PDF)
          </>
        )}
      </button>

      {/* Top Risk Factors */}
      {results.top_risk_factors && results.top_risk_factors.length > 0 && (
        <div className="section" style={{ marginTop: '1.5rem' }}>
          <div className="section-header">
            <svg className="section-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
            <h3 className="section-title">Top Risk Factors</h3>
          </div>
          {results.top_risk_factors.map((factor, idx) => (
            <div key={idx} className="factor-item">
              <div>
                <span className="factor-name">{factor.feature}</span>
                <span className={`factor-impact ${factor.impact === 'increases' ? 'increase' : 'decrease'}`}>
                  {factor.impact === 'increases' ? '↑' : '↓'} {factor.impact} risk
                </span>
              </div>
              <span className="factor-value">Value: {factor.value.toFixed(1)}</span>
            </div>
          ))}
        </div>
      )}

      {/* Recommendations */}
      <div className="section">
        <div className="section-header">
          <svg className="section-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
              d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
          <h3 className="section-title">Personalized Recommendations</h3>
        </div>
        {results.recommendations.map((rec, idx) => (
          <div key={idx} className="recommendation">
            <div className="rec-header">
              <span className="rec-category">{rec.category}</span>
              <span className={`rec-badge ${getPriorityClass(rec.priority)}`}>
                {rec.priority}
              </span>
            </div>
            <p className="rec-message">{rec.message}</p>
            <p className="rec-action">→ {rec.action}</p>
          </div>
        ))}
      </div>

      {/* Feedback */}
      {!feedbackSubmitted ? (
        <div className="feedback-box">
          <p className="feedback-title">Was this assessment helpful?</p>
          <div className="feedback-buttons">
            <button onClick={() => handleFeedback('helpful')} className="feedback-btn positive">
              👍 Yes, helpful
            </button>
            <button onClick={() => handleFeedback('not_helpful')} className="feedback-btn negative">
              👎 Not helpful
            </button>
          </div>
        </div>
      ) : (
        <div className="feedback-success">
          ✓ Thank you! Your feedback helps improve our AI.
        </div>
      )}
    </div>
  );
}
