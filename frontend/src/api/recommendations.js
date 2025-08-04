// Recommendations API

import { apiClient } from './client';
import { ENDPOINTS } from './config';

/**
 * Recommendations API service
 */
export const recommendationsAPI = {
  /**
   * Get music recommendations
   * @param {RecommendationRequest} request - Recommendation parameters
   * @returns {Promise<RecommendationResponse>} - Recommendations response
   */
  async getRecommendations(request = {}) {
    try {
      const params = {};
      
      // Add alpha parameter if provided
      if (request.alpha !== undefined) {
        params.alpha = request.alpha;
      }
      
      // Add seed tracks for cold start if provided
      if (request.seed_tracks) {
        params.seed_tracks = Array.isArray(request.seed_tracks) 
          ? request.seed_tracks.join(',')
          : request.seed_tracks;
      }

      const response = await apiClient.get(ENDPOINTS.RECOMMENDATIONS.GET, params);
      return response;
    } catch (error) {
      console.error('Failed to get recommendations:', error);
      throw new Error(`Failed to load recommendations: ${error.message}`);
    }
  },

  /**
   * Get cold start recommendations for new users
   * @param {string[]} seedTracks - Array of seed track IDs
   * @param {number} alpha - Diversity factor (0-1)
   * @returns {Promise<RecommendationResponse>} - Recommendations response
   */
  async getColdStartRecommendations(seedTracks, alpha = 0.7) {
    return this.getRecommendations({
      seed_tracks: seedTracks,
      alpha: alpha,
    });
  },

  /**
   * Get hybrid recommendations for returning users
   * @param {number} alpha - CF vs content blending factor (0-1)
   * @returns {Promise<RecommendationResponse>} - Recommendations response
   */
  async getHybridRecommendations(alpha = 0.7) {
    return this.getRecommendations({
      alpha: alpha,
    });
  },

  /**
   * Validate recommendation request
   * @param {RecommendationRequest} request - Request to validate
   * @returns {boolean} - Is valid
   */
  validateRequest(request) {
    if (request.alpha !== undefined) {
      if (typeof request.alpha !== 'number' || request.alpha < 0 || request.alpha > 1) {
        return false;
      }
    }
    
    if (request.seed_tracks !== undefined) {
      if (!Array.isArray(request.seed_tracks) && typeof request.seed_tracks !== 'string') {
        return false;
      }
    }
    
    return true;
  },
};