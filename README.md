# 🎵 Neural Music Recommendation System

**Production-grade music recommendation engine powered by Enhanced Neural Collaborative Filtering with A/B testing, contextual bandits, and real-time learning capabilities.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org)
[![React](https://img.shields.io/badge/React-19.1+-blue.svg)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![AWS](https://img.shields.io/badge/AWS-ECS-orange.svg)](https://aws.amazon.com/ecs/)

## Overview

This system demonstrates **advanced ML engineering concepts** used in production recommendation systems at companies like Spotify, Netflix, and YouTube. Built from scratch with cutting-edge techniques including Thompson sampling contextual bandits, statistical A/B testing, and real-time neural network updates.

### Key Innovations

- **Enhanced Neural Collaborative Filtering**: Hybrid Matrix Factorization + Multi-Layer Perceptron architecture (266K parameters)
- **Thompson Sampling Contextual Bandits**: Automated exploration vs exploitation optimization
- **Production A/B Testing Framework**: Statistical significance testing with automated strategy selection
- **Real-time Learning**: Instant model updates from user feedback with background processing
- **Cold Start Handling**: Sophisticated new user onboarding with content-based fallbacks

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React UI      │    │   FastAPI        │    │  Enhanced       │
│   - Chakra UI   │◄──►│   - CORS         │◄──►│  Neural CF      │
│   - Analytics   │    │   - Background   │    │  - A/B Testing  │
│   - Real-time   │    │     Tasks        │    │  - Bandits      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User          │    │   Docker         │    │  TensorFlow     │
│   Experience    │    │   Container      │    │  Model          │
│   - Preferences │    │   - AWS ECS      │    │  - 266K params  │
│   - Feedback    │    │   - Load Balancer│    │  - Real-time    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Machine Learning Pipeline

### Neural Collaborative Filtering
- **Matrix Factorization Branch**: Captures linear user-item interactions
- **Multi-Layer Perceptron Branch**: Models complex non-linear patterns  
- **Fusion Layer**: Combines both approaches for superior performance
- **Advanced Regularization**: L2 regularization, dropout, batch normalization

### A/B Testing Framework
```python
# Automatic strategy assignment with statistical significance
strategies = [ExperimentStrategy.NEURAL_CF, 
              ExperimentStrategy.CONTENT_BASED, 
              ExperimentStrategy.HYBRID_ENSEMBLE]
              
user_strategy = ab_test_manager.assign_user_to_strategy(
    experiment_id="recommendation_strategy",
    user_id=user_id
)
```

### Contextual Bandits
- **Thompson Sampling**: Bayesian approach to exploration vs exploitation
- **20-Dimensional Context**: Time, mood, device, listening history, session data
- **Real-time Optimization**: Automatically selects best recommendation strategy per user

## Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Model Parameters** | 266,401 | Hybrid MF + MLP architecture |
| **Training MAE** | 0.086 | Mean Absolute Error on validation set |
| **Real-time Latency** | <100ms | API response time for recommendations |
| **A/B Test Coverage** | 100% | All users automatically enrolled |
| **Bandit Arms** | 5 | Active recommendation strategies |

## Technology Stack

### Backend
- **Python 3.10+** - Core language
- **TensorFlow/Keras** - Deep learning framework  
- **FastAPI** - High-performance async API
- **Pandas/NumPy** - Data processing and numerical computing
- **Scikit-learn** - ML utilities and preprocessing

### Frontend  
- **React 19.1** - Modern UI framework
- **Chakra UI** - Component library
- **React Router** - Client-side routing
- **Recharts** - Data visualization

### Infrastructure
- **Docker** - Containerization
- **AWS ECS** - Container orchestration
- **GitHub Actions** - CI/CD pipeline

## Quick Start

### Prerequisites
```bash
# Backend
Python 3.10+
pip install -r requirements.txt

# Frontend  
Node.js 18+
npm install
```

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/neural-music-recommender.git
   cd neural-music-recommender
   ```

2. **Backend Setup**
   ```bash
   cd backend
   pip install -r requirements.txt
   
   # Train the enhanced model
   python neural_cf_ensemble.py
   
   # Start the API server
   python api.py
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   npm start
   ```

4. **Access the application**
   - Backend API: `http://localhost:8000`
   - Frontend UI: `http://localhost:3000`
   - API Documentation: `http://localhost:8000/docs`

## 📈 Usage Examples

### User Registration & Preferences
```javascript
// Register new user
const user = await API.register("user@example.com");

// Set music preferences  
await API.setPrefs(user.user_id, {
  genres: ["pop", "rock", "electronic"],
  energy: 0.7,
  mood: "energetic"
});
```

### Get Personalized Recommendations
```javascript
// Get recommendations (A/B testing happens automatically)
const recommendations = await API.recommend({
  user_id: user.user_id,
  n_recommendations: 10,
  mood: "happy",
  device_type: "desktop"
});
```

### Real-time Feedback & Learning
```javascript
// Submit rating (triggers real-time model updates)
await API.feedback({
  user_id: user.user_id,
  track_id: "track_123",
  rating: 5,
  context: { mood: "party", device_type: "mobile" }
});
```

## 🔬 Advanced Features

### A/B Testing Analytics
```bash
curl http://localhost:8000/analytics/ab-testing
```
Returns statistical significance testing, strategy performance, and automated recommendations.

### Contextual Bandit Performance  
```bash
curl http://localhost:8000/analytics/contextual-bandits
```
Shows Thompson sampling results, arm performance, and exploration vs exploitation balance.

### Real-time System Metrics
```bash
curl http://localhost:8000/analytics/system-metrics
```
Comprehensive production monitoring including model performance, user engagement, and ML pipeline status.

## Key Technical Achievements

### 1. Production ML Engineering
- **Real-time model updates** from user feedback
- **Automated A/B testing** with statistical significance
- **Cold start problem** solved with content-based fallbacks
- **Scalable architecture** ready for millions of users

### 2. Advanced ML Algorithms
- **Contextual multi-armed bandits** for strategy optimization
- **Thompson sampling** for optimal exploration vs exploitation  
- **Hybrid neural architecture** combining MF and deep learning
- **Statistical significance testing** for experiment validation

### 3. Full-Stack Implementation
- **Production API** with comprehensive error handling
- **Modern React UI** with real-time updates
- **Docker containerization** for consistent deployment
- **AWS ECS deployment** with load balancing

## API Documentation

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check and system status |
| `/auth/register` | POST | User registration |
| `/preferences/set` | POST | Set user music preferences |
| `/recommend` | POST | Get personalized recommendations |
| `/feedback` | POST | Submit song ratings |
| `/analytics/ab-testing` | GET | A/B test results |
| `/analytics/contextual-bandits` | GET | Bandit performance |

### Example Response
```json
{
  "track_id": "spotify_123",
  "track_name": "Blinding Lights",
  "artist": "The Weeknd", 
  "prediction_score": 4.32,
  "confidence": 0.87,
  "strategy": "neural_cf",
  "explanation": "Recommended via neural_cf strategy using deep learning"
}
```

## Testing & Validation

### Model Performance
- **Cross-validation**: 5-fold CV with stratified sampling
- **A/B testing**: Live experiments with statistical significance
- **Cold start evaluation**: Content-based fallback performance
- **Real-time learning**: Feedback incorporation speed

### System Testing
- **Load testing**: 1000+ concurrent users
- **API testing**: Comprehensive endpoint coverage
- **Integration testing**: End-to-end user flows
- **Performance monitoring**: Sub-100ms response times

## 🚀 Deployment

### Docker Deployment
```bash
# Build and run with Docker
docker build -t neural-music-recommender .
docker run -p 8000:8000 neural-music-recommender
```

### AWS ECS Deployment
```bash
# Deploy to AWS ECS (configured in docker-compose.yml)
docker-compose up --build
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)  
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Spotify Web API** for music metadata
- **TensorFlow team** for the ML framework
- **FastAPI community** for the excellent async framework
- **Neural Collaborative Filtering** research by He et al.

---

## Contact

**Sharada Ghantasala** - [sghantasala9@gatech.edu]

Project Link: [https://github.com/yourusername/neural-music-recommender](https://github.com/yourusername/neural-music-recommender)
