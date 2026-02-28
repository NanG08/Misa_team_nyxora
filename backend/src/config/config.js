module.exports = {
  // Server configuration
  server: {
    port: process.env.PORT || 5000,
    env: process.env.NODE_ENV || 'development'
  },

  // API configuration
  api: {
    version: 'v1',
    prefix: '/api'
  },

  // External API endpoints
  externalAPIs: {
    noaa: {
      baseURL: 'https://www.ncdc.noaa.gov/cdo-web/api/v2',
      timeout: 10000
    },
    movebank: {
      baseURL: 'https://www.movebank.org/movebank/service',
      timeout: 15000
    },
    globalForestWatch: {
      baseURL: 'https://data-api.globalforestwatch.org',
      timeout: 10000
    }
  },

  // AI/ML configuration
  ai: {
    gemini: {
      model: 'gemini-1.5-flash',
      temperature: 0.7,
      maxTokens: 2048
    }
  },

  // Species configuration
  species: {
    tracked: [
      'monarch',
      'arctic-tern',
      'gray-whale',
      'caribou',
      'bar-tailed-godwit'
    ]
  },

  // Risk scoring thresholds
  riskThresholds: {
    low: 30,
    moderate: 50,
    high: 70,
    critical: 85
  },

  // Scenario templates
  scenarios: {
    climate: ['warming-2c', 'warming-4c'],
    habitat: ['habitat-loss-25', 'habitat-loss-50'],
    conservation: ['conservation-success']
  },

  // Cache configuration
  cache: {
    ttl: 3600, // 1 hour in seconds
    checkPeriod: 600 // 10 minutes
  }
};
