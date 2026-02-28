export interface MigrationPoint {
  lat: number;
  lng: number;
  timestamp: string;
  species: string;
  status: 'active' | 'resting' | 'breeding';
}

export interface EnvironmentalFactor {
  type: 'temperature' | 'deforestation' | 'drought' | 'habitat_loss' | 'wildfire';
  severity: number; // 0 to 1
  location: { lat: number; lng: number };
  description: string;
  impactOnMigration: string;
}

export interface MigrationStory {
  id: string;
  title: string;
  content: string;
  species: string;
  date: string;
  impactLevel: 'low' | 'medium' | 'high';
  tags: string[];
}

export interface PredictionResult {
  species: string;
  predictedRoute: { lat: number; lng: number }[];
  confidence: number;
  reasoning: string;
  featureAttribution: { factor: string; importance: number }[];
}

export interface RiskScore {
  species: string;
  score: number; // 0-100
  trend: 'improving' | 'declining' | 'stable';
  primaryThreat: string;
}

export interface ScenarioResult {
  scenario: string;
  impactSummary: string;
  projectedRangeShift: number; // in km
  extinctionRiskIncrease: number; // percentage
}
