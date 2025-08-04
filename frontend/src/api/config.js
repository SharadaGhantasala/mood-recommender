// API Configuration

export const API_CONFIG = {
  BASE_URL: process.env.REACT_APP_API_BASE || 'http://localhost:8000',
  TIMEOUT: 10000, // 10 seconds
  HEADERS: {
    'Content-Type': 'application/json',
  },
};

export const ENDPOINTS = {
  // Authentication
  AUTH: {
    REGISTER: '/auth/register',
    LOGIN: '/auth/login',
  },
  
  // Preferences
  PREFERENCES: {
    SET: '/preferences/set',
    GET: '/preferences/get',
  },
  
  // Recommendations
  RECOMMENDATIONS: {
    GET: '/recommend',
  },
  
  // Feedback
  FEEDBACK: {
    SUBMIT: '/feedback',
  },
  
  // Analytics
  ANALYTICS: {
    BANDITS: '/analytics/contextual-bandits',
    AB_TESTING: '/analytics/ab-testing',
  },
};