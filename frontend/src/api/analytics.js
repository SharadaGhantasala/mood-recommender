// Analytics API

import { apiClient } from './client';
import { ENDPOINTS } from './config';

/**
 * Analytics API service
 */
export const analyticsAPI = {
  /**
   * Get contextual bandits analytics
   * @returns {Promise<Object>} - Bandits analytics data
   */
  async getBanditsAnalytics() {
    try {
      const response = await apiClient.get(ENDPOINTS.ANALYTICS.BANDITS);
      return response;
    } catch (error) {
      console.error('Failed to get bandits analytics:', error);
      throw new Error(`Failed to load bandits analytics: ${error.message}`);
    }
  },

  /**
   * Get A/B testing analytics
   * @returns {Promise<Object>} - A/B testing analytics data
   */
  async getABTestingAnalytics() {
    try {
      const response = await apiClient.get(ENDPOINTS.ANALYTICS.AB_TESTING);
      return response;
    } catch (error) {
      console.error('Failed to get A/B testing analytics:', error);
      throw new Error(`Failed to load A/B testing analytics: ${error.message}`);
    }
  },

  /**
   * Get all analytics data
   * @returns {Promise<Object>} - Combined analytics data
   */
  async getAllAnalytics() {
    try {
      const [bandits, abTesting] = await Promise.all([
        this.getBanditsAnalytics(),
        this.getABTestingAnalytics(),
      ]);

      return {
        bandits,
        abTesting,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      console.error('Failed to get analytics data:', error);
      throw new Error(`Failed to load analytics: ${error.message}`);
    }
  },
};