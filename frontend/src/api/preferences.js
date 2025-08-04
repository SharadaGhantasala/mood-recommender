// Preferences API

import { apiClient } from './client';
import { ENDPOINTS } from './config';

/**
 * Preferences API service
 */
export const preferencesAPI = {
  /**
   * Set user preferences
   * @param {UserPreferences} preferences - User music preferences
   * @returns {Promise<PreferencesResponse>} - Success response
   */
  async setPreferences(preferences) {
    try {
      const response = await apiClient.post(ENDPOINTS.PREFERENCES.SET, preferences);
      return response;
    } catch (error) {
      console.error('Failed to set preferences:', error);
      throw new Error(`Failed to save preferences: ${error.message}`);
    }
  },

  /**
   * Get user preferences
   * @param {string} userId - User identifier
   * @returns {Promise<UserPreferences>} - User preferences
   */
  async getPreferences(userId) {
    try {
      const response = await apiClient.get(ENDPOINTS.PREFERENCES.GET, { user_id: userId });
      return response; // Return full response, let the caller extract what they need
    } catch (error) {
      console.error('Failed to get preferences:', error);
      throw new Error(`Failed to load preferences: ${error.message}`);
    }
  },

  /**
   * Validate preferences object
   * @param {Object} preferences - Preferences to validate
   * @returns {boolean} - Is valid
   */
  validatePreferences(preferences) {
    const required = ['energy', 'danceability', 'valence', 'tempo'];
    
    for (const field of required) {
      if (!(field in preferences)) {
        return false;
      }
      
      const value = preferences[field];
      if (typeof value !== 'number' || value < 0 || value > 1) {
        return false;
      }
    }
    
    return true;
  },
};