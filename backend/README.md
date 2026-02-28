# Misa Backend API

Backend server for Misa - AI/ML platform for ecological data and migration tracking.

## 🚀 Quick Start

```bash
# Install dependencies
npm install

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run development server
npm run dev

# Run production server
npm start
```

Server runs on `http://localhost:5000`

## 📋 Prerequisites

- Node.js 16 or higher
- npm or yarn
- Google Gemini API key

## 🛠️ Installation

### 1. Install Dependencies

```bash
npm install
```

### 2. Environment Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# Server Configuration
PORT=5000
NODE_ENV=development

# API Keys
GEMINI_API_KEY=your_gemini_api_key_here

# External APIs (Optional)
NOAA_API_KEY=your_noaa_api_key
MOVEBANK_USERNAME=your_movebank_username
MOVEBANK_PASSWORD=your_movebank_password

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
```

### 3. Run the Server

**Development Mode (with auto-reload):**
```bash
npm run dev
```

**Production Mode:**
```bash
npm start
```

## 📁 Project Structure

```
backend/
├── src/
│   ├── routes/              # API route definitions
│   │   ├── migration.routes.js
│   │   ├── prediction.routes.js
│   │   ├── scenario.routes.js
│   │   └── data.routes.js
│   │
│   ├── controllers/         # Request handlers
│   │   ├── migration.controller.js
│   │   ├── prediction.controller.js
│   │   ├── scenario.controller.js
│   │   └── data.controller.js
│   │
│   ├── services/            # Business logic
│   │   ├── migration.service.js
│   │   ├── prediction.service.js
│   │   ├── scenario.service.js
│   │   └── data.service.js
│   │
│   ├── utils/               # Helper functions
│   │   └── helpers.js
│   │
│   ├── config/              # Configuration
│   │   └── config.js
│   │
│   └── server.js            # Express server entry point
│
├── .env.example             # Environment template
├── .gitignore
├── package.json
└── README.md
```

## 🔌 API Endpoints

### Health Check

```http
GET /health
```

Returns server status and timestamp.

### Migration Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/migration/species` | Get all tracked species |
| GET | `/api/migration/species/:id` | Get migration data for species |
| GET | `/api/migration/points` | Get migration points with filters |
| GET | `/api/migration/risk-score/:id` | Get risk assessment |

### Prediction Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predictions/route` | Predict migration route |
| POST | `/api/predictions/attribution` | Get SHAP feature attribution |
| POST | `/api/predictions/explain` | Get AI explanation |

### Scenario Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/scenarios/simulate` | Run scenario simulation |
| GET | `/api/scenarios/templates` | Get predefined scenarios |
| POST | `/api/scenarios/compare` | Compare multiple scenarios |

### Environmental Data Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/data/environmental` | Get environmental overlays |
| GET | `/api/data/climate` | Get climate data |
| GET | `/api/data/forest-cover` | Get forest cover data |
| GET | `/api/data/events` | Get real-time events |

## 📖 API Documentation

For detailed API documentation, see [API.md](../docs/API.md)

## 🧪 Testing

```bash
# Run tests (when implemented)
npm test

# Run with coverage
npm run test:coverage
```

## 🔒 Security

### Implemented Security Features

- **Helmet.js**: Security headers
- **CORS**: Cross-origin resource sharing
- **Environment Variables**: Sensitive data protection
- **Input Validation**: Request parameter validation

### Security Best Practices

1. Never commit `.env` file
2. Rotate API keys regularly
3. Use HTTPS in production
4. Implement rate limiting
5. Add authentication for sensitive endpoints

## 🚀 Deployment

### Heroku

```bash
# Login to Heroku
heroku login

# Create app
heroku create misa-backend

# Set environment variables
heroku config:set GEMINI_API_KEY=your_key

# Deploy
git push heroku main
```

### Docker

```bash
# Build image
docker build -t misa-backend .

# Run container
docker run -p 5000:5000 --env-file .env misa-backend
```

### AWS Elastic Beanstalk

```bash
# Initialize
eb init -p node.js-16 misa-backend

# Create environment
eb create misa-backend-prod

# Deploy
eb deploy
```

See [DEPLOYMENT.md](../docs/DEPLOYMENT.md) for detailed deployment instructions.

## 🔧 Configuration

### Server Configuration

Edit `src/config/config.js`:

```javascript
module.exports = {
  server: {
    port: process.env.PORT || 5000,
    env: process.env.NODE_ENV || 'development'
  },
  // ... other configuration
};
```

### CORS Configuration

Update allowed origins in `.env`:

```env
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173,https://your-domain.com
```

## 📊 Monitoring

### Logging

The server uses Morgan for HTTP request logging:

```
GET /api/migration/species 200 45.123 ms - 1234
POST /api/predictions/route 200 1234.567 ms - 5678
```

### Health Monitoring

Check server health:

```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "ok",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

## 🐛 Troubleshooting

### Common Issues

**Port Already in Use:**
```bash
# Find process using port 5000
lsof -i :5000

# Kill process
kill -9 <PID>
```

**Module Not Found:**
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

**CORS Errors:**
- Check `ALLOWED_ORIGINS` in `.env`
- Verify frontend URL is included

**Gemini API Errors:**
- Verify API key is correct
- Check API quota limits
- Ensure internet connection

## 📝 Development

### Adding New Endpoints

1. **Create Route** (`src/routes/feature.routes.js`):
```javascript
const express = require('express');
const router = express.Router();
const controller = require('../controllers/feature.controller');

router.get('/endpoint', controller.handler);

module.exports = router;
```

2. **Create Controller** (`src/controllers/feature.controller.js`):
```javascript
const service = require('../services/feature.service');

exports.handler = async (req, res, next) => {
  try {
    const data = await service.getData();
    res.json({ success: true, data });
  } catch (error) {
    next(error);
  }
};
```

3. **Create Service** (`src/services/feature.service.js`):
```javascript
exports.getData = async () => {
  // Business logic here
  return data;
};
```

4. **Mount Route** (`src/server.js`):
```javascript
const featureRoutes = require('./routes/feature.routes');
app.use('/api/feature', featureRoutes);
```

### Code Style

- Use `async/await` for asynchronous operations
- Follow RESTful conventions
- Use meaningful variable names
- Add comments for complex logic
- Handle errors properly

## 🔄 Updates

### Updating Dependencies

```bash
# Check for outdated packages
npm outdated

# Update all dependencies
npm update

# Update specific package
npm update package-name
```

### Version Management

Follow semantic versioning (SemVer):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

## 📚 Resources

- [Express.js Documentation](https://expressjs.com/)
- [Google Gemini API](https://ai.google.dev/)
- [Node.js Best Practices](https://github.com/goldbergyoni/nodebestpractices)
- [REST API Design](https://restfulapi.net/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write/update tests
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](../LICENSE) file for details.

## 📧 Support

For issues or questions:
- Open a GitHub issue
- Check existing documentation
- Review API documentation

## 🎯 Roadmap

### Phase 1 (Current)
- [x] Basic API endpoints
- [x] Gemini AI integration
- [x] Mock data services
- [x] CORS and security

### Phase 2 (Planned)
- [ ] Database integration (PostgreSQL/MongoDB)
- [ ] Authentication & authorization
- [ ] Rate limiting
- [ ] Caching layer (Redis)
- [ ] WebSocket support

### Phase 3 (Future)
- [ ] Real-time data integration
- [ ] Advanced ML models
- [ ] Microservices architecture
- [ ] GraphQL API
- [ ] Comprehensive testing

---

**Built with ❤️ for biodiversity conservation**
