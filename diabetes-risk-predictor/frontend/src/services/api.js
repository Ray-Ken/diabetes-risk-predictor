// frontend/src/services/api.js
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const predictDiabetes = async (inputData) => {
  const response = await api.post('/predict', inputData);
  return response.data;
};

export const submitFeedback = async (feedbackData) => {
  const response = await api.post('/feedback', feedbackData);
  return response.data;
};

export const getStats = async () => {
  const response = await api.get('/stats');
  return response.data;
};

export const downloadPDF = async (inputData) => {
  const response = await api.post('/generate-report', inputData, {
    responseType: 'blob'
  });
  return response.data;
};

export default api;
