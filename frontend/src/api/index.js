// API Module - Main Export

// Export all API services
export { preferencesAPI } from './preferences';
export { recommendationsAPI } from './recommendations';
export { feedbackAPI } from './feedback';
export { analyticsAPI } from './analytics';

// Export API client and config
export { apiClient, default as APIClient } from './client';
export { API_CONFIG, ENDPOINTS } from './config';

// Export types (for JSDoc)
export * from './types';

// Legacy API object for backward compatibility
import { preferencesAPI } from './preferences';
import { recommendationsAPI } from './recommendations';
import { feedbackAPI } from './feedback';
import { analyticsAPI } from './analytics';

/**
 * Legacy API object for backward compatibility
 * @deprecated Use individual API services instead
 */
export const API = {
  // Auth methods (placeholder - implement as needed)
  register: async (email) => {
    console.warn('API.register is not implemented yet');
    return { message: 'Registration not implemented' };
  },

  // Preferences
  setPrefs: (user_id, prefs) => preferencesAPI.setPreferences(prefs),
  
  // Recommendations
  recommend: (req) => recommendationsAPI.getRecommendations(req),
  
  // Feedback
  feedback: (fb) => feedbackAPI.submitFeedback(fb),
  
  // Analytics
  getBandits: () => analyticsAPI.getBanditsAnalytics(),
  getAB: () => analyticsAPI.getABTestingAnalytics(),
};