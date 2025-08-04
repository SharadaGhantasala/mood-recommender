// API Types and Interfaces

/**
 * @typedef {Object} UserPreferences
 * @property {number} energy - Energy level (0-1)
 * @property {number} danceability - Danceability level (0-1)
 * @property {number} valence - Valence level (0-1)
 * @property {number} tempo - Tempo preference (0-1 normalized)
 */

/**
 * @typedef {Object} TrackRecommendation
 * @property {string} track_id - Unique track identifier
 * @property {number} score - Recommendation score
 */

/**
 * @typedef {Object} RecommendationResponse
 * @property {string} mode - Recommendation mode ('cold_start' or 'hybrid')
 * @property {TrackRecommendation[]} recommendations - Array of track recommendations
 */

/**
 * @typedef {Object} PreferencesResponse
 * @property {string} message - Success message
 * @property {UserPreferences} preferences - Saved preferences
 */

/**
 * @typedef {Object} RecommendationRequest
 * @property {number} [alpha] - Blending parameter (0-1)
 * @property {string} [seed_tracks] - Comma-separated track IDs for cold start
 */

/**
 * @typedef {Object} APIError
 * @property {string} message - Error message
 * @property {number} [status] - HTTP status code
 */

export {};