import { MigrationPoint, EnvironmentalFactor, RiskScore } from './types';

export const MOCK_MIGRATION_DATA: MigrationPoint[] = [
  { lat: 34.0522, lng: -118.2437, timestamp: '2024-03-01', species: 'Monarch Butterfly', status: 'active' },
  { lat: 30.2672, lng: -97.7431, timestamp: '2024-03-05', species: 'Monarch Butterfly', status: 'active' },
  { lat: 25.7617, lng: -80.1918, timestamp: '2024-03-10', species: 'Monarch Butterfly', status: 'resting' },
  { lat: 19.4326, lng: -99.1332, timestamp: '2024-03-15', species: 'Monarch Butterfly', status: 'breeding' },
  
  { lat: 64.8378, lng: -147.7164, timestamp: '2024-05-01', species: 'Arctic Tern', status: 'active' },
  { lat: 45.5231, lng: -122.6765, timestamp: '2024-05-15', species: 'Arctic Tern', status: 'active' },
  { lat: 0.0, lng: -80.0, timestamp: '2024-06-10', species: 'Arctic Tern', status: 'active' },
  { lat: -33.8688, lng: 151.2093, timestamp: '2024-07-20', species: 'Arctic Tern', status: 'resting' },

  { lat: -34.6037, lng: -58.3816, timestamp: '2024-08-01', species: 'Blue Whale', status: 'active' },
  { lat: -20.0, lng: -40.0, timestamp: '2024-08-15', species: 'Blue Whale', status: 'active' },
  { lat: 0.0, lng: -20.0, timestamp: '2024-09-01', species: 'Blue Whale', status: 'active' },
];

export const MOCK_ENVIRONMENTAL_FACTORS: EnvironmentalFactor[] = [
  {
    type: 'temperature',
    severity: 0.8,
    location: { lat: 25.0, lng: -90.0 },
    description: 'Unusual heatwave disrupting traditional feeding grounds.',
    impactOnMigration: 'Causes early departure and metabolic stress.'
  },
  {
    type: 'deforestation',
    severity: 0.9,
    location: { lat: -3.0, lng: -60.0 },
    description: 'Rapid loss of canopy cover in the Amazon basin.',
    impactOnMigration: 'Loss of critical resting stopovers.'
  },
  {
    type: 'wildfire',
    severity: 0.75,
    location: { lat: 45.0, lng: -120.0 },
    description: 'Intense seasonal wildfires in the Pacific Northwest.',
    impactOnMigration: 'Smoke plumes force significant route deviations.'
  }
];

export const MOCK_RISK_SCORES: Record<string, RiskScore> = {
  'Monarch Butterfly': { species: 'Monarch Butterfly', score: 82, trend: 'declining', primaryThreat: 'Habitat Loss' },
  'Arctic Tern': { species: 'Arctic Tern', score: 45, trend: 'stable', primaryThreat: 'Climate Change' },
  'Blue Whale': { species: 'Blue Whale', score: 68, trend: 'declining', primaryThreat: 'Ocean Warming' },
  'African Elephant': { species: 'African Elephant', score: 75, trend: 'declining', primaryThreat: 'Fragmentation' },
};
