const axios = require('axios');

// Mock data for development - replace with real Movebank API calls
const MOCK_SPECIES = [
  { id: 'monarch', name: 'Monarch Butterfly', scientificName: 'Danaus plexippus', status: 'endangered' },
  { id: 'arctic-tern', name: 'Arctic Tern', scientificName: 'Sterna paradisaea', status: 'least-concern' },
  { id: 'gray-whale', name: 'Gray Whale', scientificName: 'Eschrichtius robustus', status: 'vulnerable' }
];

exports.fetchMigrationData = async (speciesId, options = {}) => {
  // In production, fetch from Movebank API
  // const response = await axios.get(`https://www.movebank.org/movebank/service/direct-read`, {
  //   params: { entity_type: 'event', study_id: speciesId }
  // });
  
  return {
    speciesId,
    points: generateMockMigrationPoints(speciesId, options),
    metadata: { source: 'movebank', lastUpdated: new Date().toISOString() }
  };
};

exports.getMigrationPoints = async (speciesId, startTime, endTime) => {
  const points = generateMockMigrationPoints(speciesId, { startTime, endTime });
  return points.filter(p => {
    const time = new Date(p.timestamp).getTime();
    return (!startTime || time >= new Date(startTime).getTime()) &&
           (!endTime || time <= new Date(endTime).getTime());
  });
};

exports.calculateRiskScore = async (speciesId) => {
  // Calculate risk based on habitat fragmentation, climate stressors, etc.
  const habitatFragmentation = Math.random() * 40 + 10; // 10-50
  const climateStress = Math.random() * 30 + 10; // 10-40
  const humanImpact = Math.random() * 20 + 5; // 5-25
  
  const riskScore = Math.min(100, habitatFragmentation + climateStress + humanImpact);
  
  return {
    score: Math.round(riskScore),
    factors: {
      habitatFragmentation: Math.round(habitatFragmentation),
      climateStress: Math.round(climateStress),
      humanImpact: Math.round(humanImpact)
    },
    severity: riskScore > 70 ? 'critical' : riskScore > 50 ? 'high' : riskScore > 30 ? 'moderate' : 'low',
    timestamp: new Date().toISOString()
  };
};

exports.getAllTrackedSpecies = async () => {
  return MOCK_SPECIES;
};

// Helper function to generate mock migration points
function generateMockMigrationPoints(speciesId, options = {}) {
  const points = [];
  const numPoints = 50;
  const baseDate = new Date('2024-01-01');
  
  for (let i = 0; i < numPoints; i++) {
    const lat = 40 + Math.sin(i / 5) * 20 + (Math.random() - 0.5) * 2;
    const lon = -100 + Math.cos(i / 5) * 30 + (Math.random() - 0.5) * 2;
    const timestamp = new Date(baseDate.getTime() + i * 24 * 60 * 60 * 1000);
    
    points.push({
      id: `${speciesId}-${i}`,
      lat,
      lon,
      timestamp: timestamp.toISOString(),
      status: ['active', 'resting', 'breeding'][Math.floor(Math.random() * 3)],
      altitude: Math.random() * 1000,
      speed: Math.random() * 50
    });
  }
  
  return points;
}
