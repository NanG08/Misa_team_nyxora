# System Architecture

## Overview

Misa is a full-stack web application designed to visualize and analyze animal migration patterns using AI/ML technologies. The system follows a client-server architecture with clear separation of concerns.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  React 19 + TypeScript + Vite                        │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐    │  │
│  │  │ Components │  │  Services  │  │   State    │    │  │
│  │  │  - Map     │  │  - API     │  │  - Hooks   │    │  │
│  │  │  - Charts  │  │  - Gemini  │  │  - Context │    │  │
│  │  └────────────┘  └────────────┘  └────────────┘    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP/REST
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                         Backend API                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Node.js + Express                                   │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐    │  │
│  │  │   Routes   │  │Controllers │  │  Services  │    │  │
│  │  │  - REST    │  │  - Logic   │  │  - ML      │    │  │
│  │  │  - CORS    │  │  - Valid.  │  │  - AI      │    │  │
│  │  └────────────┘  └────────────┘  └────────────┘    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ API Calls
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    External Services                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Gemini AI    │  │    NOAA      │  │  Movebank    │     │
│  │ - Narratives │  │  - Climate   │  │  - Tracking  │     │
│  │ - Explain    │  │  - Weather   │  │  - GPS Data  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Component Architecture

### Frontend Layer

#### 1. Presentation Layer
- **Components**: React functional components with TypeScript
- **Styling**: Tailwind CSS 4 with custom design system
- **Animation**: Framer Motion for smooth transitions
- **Visualization**: D3.js for interactive maps

#### 2. Service Layer
- **API Client**: Axios-based HTTP client with interceptors
- **Gemini Service**: Direct integration with Google Gemini API
- **State Management**: React hooks (useState, useEffect)

#### 3. Data Flow
```
User Interaction → Component → Service → API → Backend
                                    ↓
                              Local State Update
                                    ↓
                              Component Re-render
```

### Backend Layer

#### 1. API Layer (Routes)
- RESTful endpoint definitions
- Request validation
- Route grouping by feature

#### 2. Controller Layer
- Request/response handling
- Input validation
- Error handling
- Response formatting

#### 3. Service Layer
- Business logic implementation
- External API integration
- Data transformation
- ML model simulation

#### 4. Utility Layer
- Helper functions
- Error classes
- Configuration management

## Data Flow

### Migration Data Flow

```
1. User selects species
   ↓
2. Frontend calls GET /api/migration/points
   ↓
3. Backend fetches from Movebank (or mock data)
   ↓
4. Backend processes and formats data
   ↓
5. Frontend receives JSON response
   ↓
6. D3.js renders points on map
   ↓
7. User interacts with map (hover, click)
   ↓
8. Tooltip displays point details
```

### Prediction Flow

```
1. User clicks "AI Route Prediction"
   ↓
2. Frontend calls POST /api/predictions/attribution
   ↓
3. Backend calculates SHAP values (simulated)
   ↓
4. Frontend calls POST /api/predictions/explain
   ↓
5. Backend calls Gemini API for explanation
   ↓
6. Gemini generates natural language explanation
   ↓
7. Backend returns structured response
   ↓
8. Frontend displays prediction with XAI
```

### Scenario Simulation Flow

```
1. User selects scenario (e.g., +2°C warming)
   ↓
2. Frontend calls POST /api/scenarios/simulate
   ↓
3. Backend applies scenario parameters
   ↓
4. Backend calculates impact metrics
   ↓
5. Backend calls Gemini for narrative
   ↓
6. Backend returns simulation results
   ↓
7. Frontend displays impact visualization
```

## Technology Stack

### Frontend Technologies

| Technology | Purpose | Version |
|------------|---------|---------|
| React | UI Framework | 19 |
| TypeScript | Type Safety | 5.x |
| Vite | Build Tool | 5.x |
| Tailwind CSS | Styling | 4.x |
| D3.js | Data Visualization | 7.x |
| Framer Motion | Animations | 11.x |
| Axios | HTTP Client | 1.x |

### Backend Technologies

| Technology | Purpose | Version |
|------------|---------|---------|
| Node.js | Runtime | 16+ |
| Express | Web Framework | 4.x |
| Gemini API | AI/ML | 1.5 Flash |
| Helmet | Security | 7.x |
| CORS | Cross-Origin | 2.x |
| dotenv | Config | 16.x |

## Security Architecture

### Frontend Security
- Environment variable protection
- XSS prevention via React
- HTTPS enforcement (production)
- Content Security Policy

### Backend Security
- Helmet.js security headers
- CORS configuration
- Input validation
- Rate limiting (planned)
- API key authentication (planned)

### API Security
- Gemini API key stored server-side
- No sensitive data in client
- HTTPS for all external calls

## Scalability Considerations

### Current Architecture
- Stateless backend (horizontal scaling ready)
- Client-side rendering
- Mock data for development

### Future Enhancements

#### Phase 1: Database Integration
```
Backend → PostgreSQL/MongoDB
  ↓
Cache Layer (Redis)
  ↓
API Response
```

#### Phase 2: Microservices
```
API Gateway
  ├── Migration Service
  ├── Prediction Service
  ├── Scenario Service
  └── Data Service
```

#### Phase 3: Real-time Updates
```
WebSocket Server
  ↓
Real-time Migration Updates
  ↓
Client Auto-refresh
```

## Performance Optimization

### Frontend
- Code splitting with Vite
- Lazy loading components
- D3.js canvas rendering for large datasets
- Debounced API calls
- Memoized calculations

### Backend
- Response compression (gzip)
- Caching headers
- Async/await for I/O operations
- Connection pooling (future)

## Error Handling

### Frontend Error Handling
```typescript
try {
  const data = await api.getMigrationData();
  setData(data);
} catch (error) {
  console.error('Failed to load:', error);
  // Fallback to mock data
  setData(MOCK_DATA);
}
```

### Backend Error Handling
```javascript
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(err.status || 500).json({
    error: {
      message: err.message,
      status: err.status || 500
    }
  });
});
```

## Monitoring & Logging

### Current Implementation
- Console logging (development)
- Morgan HTTP request logging
- Error stack traces

### Planned Enhancements
- Winston structured logging
- Application Performance Monitoring (APM)
- Error tracking (Sentry)
- Analytics (Google Analytics)

## Deployment Architecture

### Development
```
localhost:5173 (Frontend)
localhost:5000 (Backend)
```

### Production
```
CDN (Frontend Static Assets)
  ↓
Load Balancer
  ↓
Backend Instances (Auto-scaling)
  ↓
Database Cluster
```

## API Design Principles

1. **RESTful**: Standard HTTP methods (GET, POST, PUT, DELETE)
2. **Consistent**: Uniform response format
3. **Versioned**: API version in URL path
4. **Documented**: OpenAPI/Swagger (planned)
5. **Secure**: Authentication and authorization
6. **Performant**: Caching and optimization

## Testing Strategy

### Frontend Testing (Planned)
- Unit tests: Vitest
- Component tests: React Testing Library
- E2E tests: Playwright

### Backend Testing (Planned)
- Unit tests: Jest
- Integration tests: Supertest
- API tests: Postman/Newman

## Development Workflow

```
1. Feature Branch
   ↓
2. Local Development
   ↓
3. Testing
   ↓
4. Code Review
   ↓
5. Merge to Main
   ↓
6. CI/CD Pipeline
   ↓
7. Deployment
```

## Future Architecture Enhancements

1. **GraphQL API**: Replace REST with GraphQL
2. **Server-Side Rendering**: Next.js migration
3. **Edge Computing**: Deploy ML models to edge
4. **Digital Twin**: Real-time biosphere simulation
5. **Mobile Apps**: React Native applications
6. **Offline Support**: Progressive Web App (PWA)
