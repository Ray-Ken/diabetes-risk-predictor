// frontend/src/components/InputForm.jsx
import React, { useState } from 'react';
import { predictDiabetes } from '../services/api';

export default function InputForm({ setResults, loading, setLoading }) {
  const [formData, setFormData] = useState({
    age: '',
    gender: '',
    bmi: '',
    HbA1c_level: '',
    blood_glucose_level: '',
    smoking_history: '',
    hypertension: '',
    heart_disease: ''
  });
  
  const [errors, setErrors] = useState({});
  const [apiError, setApiError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    // Clear error when user types
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  const validate = () => {
    const newErrors = {};
    
    if (!formData.age) newErrors.age = 'Age is required';
    else if (formData.age < 1 || formData.age > 120) newErrors.age = 'Age must be between 1 and 120';
    
    if (!formData.gender) newErrors.gender = 'Gender is required';
    
    if (!formData.bmi) newErrors.bmi = 'BMI is required';
    else if (formData.bmi < 10 || formData.bmi > 70) newErrors.bmi = 'BMI must be between 10 and 70';
    
    if (!formData.HbA1c_level) newErrors.HbA1c_level = 'HbA1c level is required';
    else if (formData.HbA1c_level < 3 || formData.HbA1c_level > 15) newErrors.HbA1c_level = 'HbA1c must be between 3 and 15';
    
    if (!formData.blood_glucose_level) newErrors.blood_glucose_level = 'Blood glucose is required';
    else if (formData.blood_glucose_level < 50 || formData.blood_glucose_level > 400) newErrors.blood_glucose_level = 'Glucose must be between 50 and 400';
    
    if (!formData.smoking_history) newErrors.smoking_history = 'Smoking history is required';
    if (formData.hypertension === '') newErrors.hypertension = 'This field is required';
    if (formData.heart_disease === '') newErrors.heart_disease = 'This field is required';
    
    return newErrors;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    const validationErrors = validate();
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      return;
    }

    setLoading(true);
    setApiError(null);

    try {
      const formattedData = {
        age: parseFloat(formData.age),
        bmi: parseFloat(formData.bmi),
        HbA1c_level: parseFloat(formData.HbA1c_level),
        blood_glucose_level: parseFloat(formData.blood_glucose_level),
        gender: formData.gender,
        smoking_history: formData.smoking_history,
        hypertension: parseInt(formData.hypertension),
        heart_disease: parseInt(formData.heart_disease)
      };

      const result = await predictDiabetes(formattedData);
      result.inputData = formattedData; // Store for PDF generation
      setResults(result);
    } catch (err) {
      console.error('Prediction error:', err);
      setApiError(err.response?.data?.detail || 'Prediction failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2 className="form-title">Enter Your Health Information</h2>

      {apiError && (
        <div className="alert alert-error">
          <svg className="alert-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
              d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>{apiError}</span>
        </div>
      )}

      <form onSubmit={handleSubmit}>
        {/* Age */}
        <div className="form-group">
          <label className="form-label">Age (years) *</label>
          <input
            type="number"
            name="age"
            value={formData.age}
            onChange={handleChange}
            className="form-input"
            placeholder="e.g., 45"
          />
          {errors.age && <p className="form-error">{errors.age}</p>}
        </div>

        {/* Gender */}
        <div className="form-group">
          <label className="form-label">Gender *</label>
          <select
            name="gender"
            value={formData.gender}
            onChange={handleChange}
            className="form-select"
          >
            <option value="">Select gender</option>
            <option value="Male">Male</option>
            <option value="Female">Female</option>
            <option value="Other">Other</option>
          </select>
          {errors.gender && <p className="form-error">{errors.gender}</p>}
        </div>

        {/* BMI */}
        <div className="form-group">
          <label className="form-label">BMI (Body Mass Index) *</label>
          <input
            type="number"
            step="0.1"
            name="bmi"
            value={formData.bmi}
            onChange={handleChange}
            className="form-input"
            placeholder="e.g., 28.5"
          />
          <p className="form-hint">Weight (kg) ÷ Height² (m²)</p>
          {errors.bmi && <p className="form-error">{errors.bmi}</p>}
        </div>

        {/* HbA1c */}
        <div className="form-group">
          <label className="form-label">HbA1c Level (%) *</label>
          <input
            type="number"
            step="0.1"
            name="HbA1c_level"
            value={formData.HbA1c_level}
            onChange={handleChange}
            className="form-input"
            placeholder="e.g., 5.7"
          />
          <p className="form-hint">Normal: &lt;5.7%, Prediabetes: 5.7-6.4%, Diabetes: ≥6.5%</p>
          {errors.HbA1c_level && <p className="form-error">{errors.HbA1c_level}</p>}
        </div>

        {/* Blood Glucose */}
        <div className="form-group">
          <label className="form-label">Fasting Blood Glucose (mg/dL) *</label>
          <input
            type="number"
            name="blood_glucose_level"
            value={formData.blood_glucose_level}
            onChange={handleChange}
            className="form-input"
            placeholder="e.g., 110"
          />
          <p className="form-hint">Normal: &lt;100, Prediabetes: 100-125, Diabetes: ≥126</p>
          {errors.blood_glucose_level && <p className="form-error">{errors.blood_glucose_level}</p>}
        </div>

        {/* Smoking History */}
        <div className="form-group">
          <label className="form-label">Smoking History *</label>
          <select
            name="smoking_history"
            value={formData.smoking_history}
            onChange={handleChange}
            className="form-select"
          >
            <option value="">Select smoking status</option>
            <option value="never">Never smoked</option>
            <option value="former">Former smoker</option>
            <option value="current">Current smoker</option>
            <option value="No Info">Prefer not to say</option>
          </select>
          {errors.smoking_history && <p className="form-error">{errors.smoking_history}</p>}
        </div>

        {/* Hypertension */}
        <div className="form-group">
          <label className="form-label">Do you have Hypertension (High Blood Pressure)? *</label>
          <div className="radio-group">
            <label className="radio-label">
              <input
                type="radio"
                name="hypertension"
                value="0"
                checked={formData.hypertension === '0'}
                onChange={handleChange}
              />
              No
            </label>
            <label className="radio-label">
              <input
                type="radio"
                name="hypertension"
                value="1"
                checked={formData.hypertension === '1'}
                onChange={handleChange}
              />
              Yes
            </label>
          </div>
          {errors.hypertension && <p className="form-error">{errors.hypertension}</p>}
        </div>

        {/* Heart Disease */}
        <div className="form-group">
          <label className="form-label">Do you have Heart Disease? *</label>
          <div className="radio-group">
            <label className="radio-label">
              <input
                type="radio"
                name="heart_disease"
                value="0"
                checked={formData.heart_disease === '0'}
                onChange={handleChange}
              />
              No
            </label>
            <label className="radio-label">
              <input
                type="radio"
                name="heart_disease"
                value="1"
                checked={formData.heart_disease === '1'}
                onChange={handleChange}
              />
              Yes
            </label>
          </div>
          {errors.heart_disease && <p className="form-error">{errors.heart_disease}</p>}
        </div>

        {/* Submit Button */}
        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading ? (
            <>
              <svg className="btn-icon spinner" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Analyzing...
            </>
          ) : (
            'Assess Diabetes Risk'
          )}
        </button>
      </form>
    </div>
  );
}
