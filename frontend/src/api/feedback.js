// Feedback API

import { apiClient } from './client';
import { ENDPOINTS } from './config';

/**
 * Feedback API service
 */
export const feedbackAPI = {
  /**
   * Submit user feedback on recommendations
   * @param {Object} feedback - Feedback data
   * @param {string} feedback.trackId - Track ID
   * @param {number} feedback.rating - Rating (1-5)
   * @param {string} [feedback.userId] - User ID
   * @param {string} [feedback.comment] - Optional comment
   * @returns {Promise<Object>} - Feedback response
   */
  async submitFeedback(feedback) {
    try {
      const response = await apiClient.post(ENDPOINTS.FEEDBACK.SUBMIT, feedback);
      return response;
    } catch (error) {
      console.error('Failed to submit feedback:', error);
      throw new Error(`Failed to submit feedback: ${error.message}`);
    }
  },

  /**
   * Submit track like/dislike
   * @param {string} trackId - Track ID
   * @param {boolean} liked - Whether user liked the track
   * @param {string} [userId] - User ID
   * @returns {Promise<Object>} - Feedback response
   */
  async submitLike(trackId, liked, userId = null) {
    const feedback = {
      trackId,
      rating: liked ? 5 : 1,
      userId,
      type: 'like',
    };
    
    return this.submitFeedback(feedback);
  },

  /**
   * Submit detailed rating
   * @param {string} trackId - Track ID
   * @param {number} rating - Rating 1-5
   * @param {string} [userId] - User ID
   * @param {string} [comment] - Optional comment
   * @returns {Promise<Object>} - Feedback response
   */
  async submitRating(trackId, rating, userId = null, comment = null) {
    const feedback = {
      trackId,
      rating,
      userId,
      comment,
      type: 'rating',
    };
    
    return this.submitFeedback(feedback);
  },

  /**
   * Validate feedback object
   * @param {Object} feedback - Feedback to validate
   * @returns {boolean} - Is valid
   */
  validateFeedback(feedback) {
    if (!feedback.trackId || typeof feedback.trackId !== 'string') {
      return false;
    }
    
    if (!feedback.rating || typeof feedback.rating !== 'number') {
      return false;
    }
    
    if (feedback.rating < 1 || feedback.rating > 5) {
      return false;
    }
    
    return true;
  },
};