import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Globe, 
  Wind, 
  Thermometer, 
  Trees, 
  AlertTriangle, 
  ChevronRight, 
  Info,
  Activity,
  Map as MapIcon,
  BookOpen,
  TrendingUp,
  RefreshCw
} from 'lucide-react';
import { MigrationMap } from './components/MigrationMap';
import { geminiService } from './services/geminiService';
import { migrationAPI, predictionAPI, scenarioAPI } from './services/api';
import { MOCK_MIGRATION_DATA, MOCK_ENVIRONMENTAL_FACTORS, MOCK_RISK_SCORES } from './constants';
import { MigrationStory, PredictionResult, ScenarioResult } from './types';
import { cn } from './lib/utils';
import ReactMarkdown from 'react-markdown';

export default function App() {
  const [selectedSpecies, setSelectedSpecies] = useState<string>('monarch');
  const [stories, setStories] = useState<MigrationStory[]>([]);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [scenario, setScenario] = useState<ScenarioResult | null>(null);
  const [isLoadingStories, setIsLoadingStories] = useState(false);
  const [isLoadingPrediction, setIsLoadingPrediction] = useState(false);
  const [isLoadingScenario, setIsLoadingScenario] = useState(false);
  const [activeTab, setActiveTab] = useState<'map' | 'stories' | 'predictions' | 'scenarios'>('map');
  const [currentTime, setCurrentTime] = useState<string>('2024-09-01');
  const [overlay, setOverlay] = useState<'none' | 'temperature' | 'forest' | 'drought'>('none');
  const [migrationPoints, setMigrationPoints] = useState<any[]>(MOCK_MIGRATION_DATA);
  const [riskScore, setRiskScore] = useState<any>(null);

  useEffect(() => {
    loadStories();
    loadMigrationData();
    loadRiskScore();
  }, [selectedSpecies]);

  const loadStories = async () => {
    setIsLoadingStories(true);
    try {
      const data = await geminiService.generateMigrationStories(selectedSpecies);
      setStories(data);
    } catch (error) {
      console.error(error);
    } finally {
      setIsLoadingStories(false);
    }
  };

  const loadMigrationData = async () => {
    try {
      const points = await migrationAPI.getMigrationPoints(selectedSpecies);
      setMigrationPoints(points);
    } catch (error) {
      console.error('Failed to load migration data:', error);
      setMigrationPoints(MOCK_MIGRATION_DATA);
    }
  };

  const loadRiskScore = async () => {
    try {
      const score = await migrationAPI.getRiskScore(selectedSpecies);
      setRiskScore(score);
    } catch (error) {
      console.error('Failed to load risk score:', error);
      setRiskScore(MOCK_RISK_SCORES[selectedSpecies]);
    }
  };

  const handlePredict = async () => {
    setIsLoadingPrediction(true);
    try {
      const attribution = await predictionAPI.getFeatureAttribution(selectedSpecies, {});
      const explanation = await predictionAPI.explainPrediction(selectedSpecies, {}, attribution.features);
      
      const result: PredictionResult = {
        confidence: 0.85,
        reasoning: explanation.explanation,
        featureAttribution: attribution.features.map((f: any) => ({
          factor: f.name,
          importance: f.importance
        }))
      };
      
      setPrediction(result);
      setActiveTab('predictions');
    } catch (error) {
      console.error('Prediction failed:', error);
      const factors = MOCK_ENVIRONMENTAL_FACTORS.map(f => f.description);
      const result = await geminiService.predictMigrationShift(selectedSpecies, factors);
      setPrediction(result);
      setActiveTab('predictions');
    } finally {
      setIsLoadingPrediction(false);
    }
  };

  const handleSimulate = async (scenarioId: string) => {
    setIsLoadingScenario(true);
    try {
      const result = await scenarioAPI.simulate(selectedSpecies, scenarioId);
      
      const scenarioResult: ScenarioResult = {
        scenario: result.scenarioId,
        impactSummary: result.narrative,
        extinctionRiskIncrease: result.results.riskChange,
        projectedRangeShift: result.results.rangeShift.north
      };
      
      setScenario(scenarioResult);
      setActiveTab('scenarios');
    } catch (error) {
      console.error('Simulation failed:', error);
      const result = await geminiService.simulateScenario(selectedSpecies, scenarioId);
      setScenario(result);
      setActiveTab('scenarios');
    } finally {
      setIsLoadingScenario(false);
    }
  };



  const getFactorExplanation = (factor: string, importance: number, species: string): string => {
    const explanations: Record<string, string> = {
      'Temperature': `Rising temperatures (${(importance * 100).toFixed(0)}% influence) are causing ${species} to alter their migration timing and routes. Warmer conditions affect food availability, breeding cycles, and energy expenditure during flight. This species is particularly sensitive to thermal stress, which can lead to earlier departures from breeding grounds and shifts in stopover locations to cooler regions.`,
      'Habitat Loss': `Habitat fragmentation contributes ${(importance * 100).toFixed(0)}% to route changes. Critical stopover sites and breeding grounds are being converted to agricultural or urban land, forcing ${species} to find alternative resting areas. This increases migration distance and energy costs, potentially reducing survival rates, especially for juveniles making their first journey.`,
      'Food Availability': `Changes in food sources account for ${(importance * 100).toFixed(0)}% of the predicted shift. Climate-driven phenological mismatches mean that peak food availability no longer aligns with traditional migration schedules. ${species} must now adjust routes to track shifting resource distributions, particularly affecting nectar sources, prey populations, or vegetation patterns critical to their survival.`,
      'Wind Patterns': `Altered wind patterns influence ${(importance * 100).toFixed(0)}% of route decisions. Changing atmospheric circulation affects tailwind availability and storm frequency along traditional corridors. ${species} relies on favorable winds to reduce energy expenditure during long-distance flights, and disruptions force them to seek alternative pathways with more predictable wind assistance.`,
      'Precipitation': `Rainfall pattern changes contribute ${(importance * 100).toFixed(0)}% to migration shifts. Drought conditions in traditional habitats reduce water availability and vegetation cover, while excessive rainfall can flood nesting sites. ${species} is adapting by shifting to regions with more stable precipitation regimes, though this may expose them to new predators and competition.`
    };
    return explanations[factor] || `${factor} contributes ${(importance * 100).toFixed(0)}% to the predicted route shift for ${species}. This environmental variable plays a significant role in determining optimal migration pathways and timing.`;
  };

  return (
    <div className="min-h-screen flex flex-col data-grid">
      {/* Header */}
      <header className="border-b border-line bg-white/50 backdrop-blur-xl sticky top-0 z-50 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-ink text-bg flex items-center justify-center rounded-lg shadow-lg">
            <Globe className="w-6 h-6" />
          </div>
          <div>
            <h1>Misa</h1>
            <p className="text-[10px] font-mono opacity-50 uppercase tracking-[0.2em]">Environmental Intelligence Platform</p>
          </div>
        </div>

        <nav className="hidden md:flex items-center gap-1 bg-ink/5 p-1 rounded-full border border-line">
          {(['map', 'stories', 'predictions', 'scenarios'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={cn(
                "px-4 py-1.5 rounded-full text-xs font-medium transition-all uppercase tracking-wider",
                activeTab === tab ? "bg-white text-ink shadow-sm" : "text-ink/40 hover:text-ink/60"
              )}
            >
              {tab}
            </button>
          ))}
        </nav>

        <div className="flex items-center gap-4">
          <div className="hidden sm:flex flex-col items-end">
            <span className="text-[10px] font-mono opacity-50 uppercase">System Status</span>
            <span className="text-[10px] font-mono text-emerald-600 uppercase flex items-center gap-1">
              <Activity className="w-3 h-3" /> Operational
            </span>
          </div>
        </div>
      </header>

      <main className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-0">
        {/* Left Sidebar: Controls & Species Selection */}
        <aside className="lg:col-span-3 border-r border-line bg-white/30 p-6 flex flex-col gap-6 overflow-y-auto max-h-[calc(100vh-80px)]">
          <section>
            <h2 className="text-[11px] font-serif italic uppercase opacity-50 mb-4 tracking-widest">Select Species</h2>
            <div className="space-y-2">
              {[
                { id: 'monarch', name: 'Monarch Butterfly' },
                { id: 'arctic-tern', name: 'Arctic Tern' },
                { id: 'gray-whale', name: 'Gray Whale' }
              ].map((species) => (
                <button
                  key={species.id}
                  onClick={() => setSelectedSpecies(species.id)}
                  className={cn(
                    "w-full text-left px-4 py-3 rounded-xl border transition-all flex items-center justify-between group",
                    selectedSpecies === species.id 
                      ? "bg-ink text-bg border-ink shadow-xl" 
                      : "bg-white border-line hover:border-ink/20"
                  )}
                >
                  <span className="text-sm font-medium">{species.name}</span>
                  <ChevronRight className={cn("w-4 h-4 transition-transform", selectedSpecies === species.id ? "translate-x-1" : "opacity-0 group-hover:opacity-100")} />
                </button>
              ))}
            </div>
          </section>

          {/* Risk Score Card */}
          {riskScore && (
            <section className="p-5 bg-ink text-bg rounded-2xl shadow-xl">
              <div className="flex justify-between items-start mb-4">
                <h3 className="text-[10px] font-mono uppercase opacity-50 tracking-widest">Migration Risk Score</h3>
                <span className={cn(
                  "text-[10px] font-mono px-2 py-0.5 rounded-full uppercase",
                  riskScore.trend === 'declining' ? "bg-red-500/20 text-red-400" : "bg-emerald-500/20 text-emerald-400"
                )}>
                  {riskScore.trend}
                </span>
              </div>
              <div className="flex items-baseline gap-2 mb-2">
                <span className="text-5xl font-serif italic">{riskScore.score}</span>
                <span className="text-xs opacity-40">/ 100</span>
              </div>
              <p className="text-xs opacity-60">Primary Threat: <span className="text-bg opacity-100 font-medium">{riskScore.primaryThreat}</span></p>
            </section>
          )}

          <section>
            <h2 className="text-[11px] font-serif italic uppercase opacity-50 mb-4 tracking-widest">Environmental Layers</h2>
            <div className="grid grid-cols-2 gap-2">
              {(['none', 'temperature', 'forest', 'drought'] as const).map((l) => (
                <button
                  key={l}
                  onClick={() => setOverlay(l)}
                  className={cn(
                    "px-3 py-2 rounded-lg border text-[10px] font-mono uppercase tracking-wider transition-all",
                    overlay === l ? "bg-ink text-bg border-ink" : "bg-white border-line hover:bg-ink/5"
                  )}
                >
                  {l}
                </button>
              ))}
            </div>
          </section>

          <section>
            <h2 className="text-[11px] font-serif italic uppercase opacity-50 mb-4 tracking-widest">Scenario Simulations</h2>
            <div className="space-y-2">
              {[
                { label: '+2°C Global Warming', id: 'warming-2c' },
                { label: '50% Habitat Loss', id: 'habitat-loss-50' },
                { label: 'Severe Drought', id: 'drought-severe' }
              ].map((s, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSimulate(s.id)}
                  className="w-full text-left p-3 rounded-xl bg-white border border-line hover:border-ink/20 transition-all group"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-bold">{s.label}</span>
                    <RefreshCw className="w-3 h-3 opacity-0 group-hover:opacity-40 transition-opacity" />
                  </div>
                  <p className="text-[10px] text-ink/50 leading-tight">Simulate long-term impact on species.</p>
                </button>
              ))}
            </div>
          </section>

          <button 
            onClick={handlePredict}
            disabled={isLoadingPrediction}
            className="mt-4 w-full bg-ink text-bg py-4 rounded-xl font-bold uppercase tracking-widest text-xs flex items-center justify-center gap-2 hover:scale-[1.02] active:scale-[0.98] transition-all disabled:opacity-50"
          >
            {isLoadingPrediction ? <RefreshCw className="w-4 h-4 animate-spin" /> : <TrendingUp className="w-4 h-4" />}
            AI Route Prediction
          </button>
        </aside>

        {/* Main Content Area */}
        <div className="lg:col-span-9 flex flex-col">
          <AnimatePresence mode="wait">
            {activeTab === 'map' && (
              <motion.div 
                key="map"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="flex-1 p-6 flex flex-col"
              >
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h2 className="text-3xl font-serif italic font-semibold">{selectedSpecies} Migration</h2>
                    <p className="text-sm text-ink/50">Real-time telemetry and historical route analysis</p>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="flex flex-col items-end">
                      <span className="text-[10px] font-mono uppercase opacity-40">Timeline</span>
                      <input 
                        type="range" 
                        min="2024-03-01" 
                        max="2024-09-01" 
                        step="1"
                        value={currentTime}
                        onChange={(e) => setCurrentTime(e.target.value)}
                        className="w-48 h-1 bg-line rounded-lg appearance-none cursor-pointer accent-ink"
                      />
                      <span className="text-[10px] font-mono mt-1">{currentTime}</span>
                    </div>
                  </div>
                </div>
                <div className="flex-1 min-h-[500px]">
                  <MigrationMap 
                    points={migrationPoints} 
                    selectedSpecies={selectedSpecies} 
                    currentTime={currentTime}
                    showEnvironmentalOverlay={overlay}
                  />
                </div>
              </motion.div>
            )}

            {activeTab === 'stories' && (
              <motion.div 
                key="stories"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="flex-1 p-8 max-w-4xl mx-auto"
              >
                <div className="mb-12 text-center">
                  <h2 className="text-4xl font-serif italic font-bold mb-4">The {selectedSpecies} Chronicles</h2>
                  <p className="text-ink/60">AI-curated narratives exploring the human impact on migration</p>
                </div>

                {isLoadingStories ? (
                  <div className="flex flex-col items-center justify-center py-20 gap-4">
                    <RefreshCw className="w-8 h-8 animate-spin opacity-20" />
                    <p className="text-xs font-mono uppercase tracking-widest opacity-40">Synthesizing narratives...</p>
                  </div>
                ) : (
                  <div className="space-y-12">
                    {stories.map((story) => (
                      <article key={story.id} className="group">
                        <div className="flex items-center gap-4 mb-4">
                          <span className="text-[10px] font-mono uppercase tracking-widest opacity-40">{story.date}</span>
                          <div className="h-px flex-1 bg-line" />
                          <div className="flex gap-2">
                            {story.tags.map(tag => (
                              <span key={tag} className="text-[10px] font-mono uppercase px-2 py-0.5 rounded-full bg-ink/5 text-ink/60 border border-line">
                                {tag}
                              </span>
                            ))}
                          </div>
                          <span className={cn(
                            "text-[10px] font-mono uppercase px-2 py-0.5 rounded border",
                            story.impactLevel === 'high' ? "border-red-200 text-red-600 bg-red-50" : "border-line text-ink/50"
                          )}>
                            Impact: {story.impactLevel}
                          </span>
                        </div>
                        <h3 className="text-2xl font-serif font-bold mb-4 group-hover:text-ink/70 transition-colors">{story.title}</h3>
                        <div className="prose prose-sm max-w-none text-ink/80 leading-relaxed font-serif text-lg">
                          <ReactMarkdown>{story.content}</ReactMarkdown>
                        </div>
                      </article>
                    ))}
                  </div>
                )}
              </motion.div>
            )}

            {activeTab === 'predictions' && (
              <motion.div 
                key="predictions"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 1.05 }}
                className="flex-1 p-8"
              >
                <div className="max-w-4xl mx-auto">
                  <div className="bg-ink text-bg p-12 rounded-[2rem] shadow-2xl relative overflow-hidden">
                    <div className="absolute top-0 right-0 p-8 opacity-10">
                      <TrendingUp className="w-64 h-64" />
                    </div>
                    
                    <div className="relative z-10">
                      <div className="flex items-center gap-3 mb-8">
                        <div className="w-12 h-12 bg-white/10 rounded-full flex items-center justify-center backdrop-blur-md border border-white/20">
                          <Activity className="w-6 h-6 text-emerald-400" />
                        </div>
                        <div>
                          <h2 className="text-[11px] font-mono uppercase tracking-[0.3em] opacity-60">Predictive Analysis Engine</h2>
                          <p className="text-2xl font-serif italic">Route Shift Projection</p>
                        </div>
                      </div>

                      {prediction ? (
                        <div className="space-y-8">
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                            <div className="space-y-4">
                              <h3 className="text-sm font-mono uppercase opacity-50 border-b border-white/10 pb-2">Confidence Score</h3>
                              <div className="flex items-end gap-2">
                                <span className="text-6xl font-serif italic">{(prediction.confidence * 100).toFixed(0)}%</span>
                                <span className="text-xs font-mono mb-2 opacity-40">Probability</span>
                              </div>
                            </div>
                            <div className="space-y-4">
                              <h3 className="text-sm font-mono uppercase opacity-50 border-b border-white/10 pb-2">Explainable AI (XAI)</h3>
                              <div className="space-y-2">
                                {prediction.featureAttribution.map((attr, idx) => (
                                  <div key={idx} className="space-y-1">
                                    <div className="flex justify-between text-[10px] font-mono uppercase opacity-60">
                                      <span>{attr.factor}</span>
                                      <span>{(attr.importance * 100).toFixed(0)}%</span>
                                    </div>
                                    <div className="h-1 bg-white/10 rounded-full overflow-hidden">
                                      <motion.div 
                                        initial={{ width: 0 }}
                                        animate={{ width: `${attr.importance * 100}%` }}
                                        className="h-full bg-emerald-400"
                                      />
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          </div>

                          <div className="space-y-4">
                            <h3 className="text-sm font-mono uppercase opacity-50 border-b border-white/10 pb-2">AI Reasoning</h3>
                            <p className="text-xl font-serif leading-relaxed italic opacity-90">
                              "{prediction.reasoning}"
                            </p>
                          </div>

                          <div className="space-y-6 mt-8 pt-8 border-t border-white/10">
                            <h3 className="text-sm font-mono uppercase opacity-50">In-Depth Factor Analysis</h3>
                            <div className="space-y-6">
                              {prediction.featureAttribution.map((attr, idx) => (
                                <div key={idx} className="bg-white/5 p-6 rounded-2xl border border-white/10">
                                  <div className="flex items-center justify-between mb-4">
                                    <h4 className="text-lg font-bold">{attr.factor}</h4>
                                    <span className="text-2xl font-serif italic text-emerald-400">{(attr.importance * 100).toFixed(0)}%</span>
                                  </div>
                                  <p className="text-sm leading-relaxed opacity-80">
                                    {getFactorExplanation(attr.factor, attr.importance, selectedSpecies)}
                                  </p>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="py-20 text-center">
                          <p className="text-xl font-serif italic opacity-50">Run the prediction engine from the sidebar to see future route shifts.</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {activeTab === 'scenarios' && (
              <motion.div 
                key="scenarios"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="flex-1 p-8 max-w-4xl mx-auto"
              >
                <div className="mb-12">
                  <h2 className="text-4xl font-serif italic font-bold mb-2">Scenario Simulation</h2>
                  <p className="text-ink/60">Long-term ecological impact modeling for {selectedSpecies}</p>
                </div>

                {isLoadingScenario ? (
                  <div className="flex flex-col items-center justify-center py-20 gap-4">
                    <RefreshCw className="w-8 h-8 animate-spin opacity-20" />
                    <p className="text-xs font-mono uppercase tracking-widest opacity-40">Running ecological models...</p>
                  </div>
                ) : scenario ? (
                  <div className="space-y-8">
                    <div className="p-8 bg-white border border-line rounded-3xl shadow-sm">
                      <h3 className="text-[10px] font-mono uppercase opacity-40 mb-4 tracking-widest">Scenario: {scenario.scenario}</h3>
                      <p className="text-2xl font-serif italic leading-relaxed mb-8">
                        {scenario.impactSummary}
                      </p>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="p-6 bg-red-50 rounded-2xl border border-red-100">
                          <h4 className="text-[10px] font-mono uppercase text-red-600 mb-2">Extinction Risk Increase</h4>
                          <span className="text-4xl font-serif italic text-red-700">+{scenario.extinctionRiskIncrease}%</span>
                        </div>
                        <div className="p-6 bg-orange-50 rounded-2xl border border-orange-100">
                          <h4 className="text-[10px] font-mono uppercase text-orange-600 mb-2">Projected Range Shift</h4>
                          <span className="text-4xl font-serif italic text-orange-700">{scenario.projectedRangeShift}km</span>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-4 p-6 bg-ink text-bg rounded-2xl">
                      <AlertTriangle className="w-8 h-8 text-orange-400 shrink-0" />
                      <div>
                        <h4 className="text-xs font-bold uppercase mb-1">Critical Intervention Required</h4>
                        <p className="text-xs opacity-60">This simulation suggests that current conservation efforts may be insufficient under this scenario. Habitat restoration must be prioritized in the northern corridors.</p>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="py-20 text-center border-2 border-dashed border-line rounded-3xl">
                    <Info className="w-12 h-12 mx-auto mb-4 opacity-20" />
                    <p className="text-xl font-serif italic opacity-50">Select a scenario from the sidebar to begin simulation.</p>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>

      {/* Footer / Status Bar */}
      <footer className="border-t border-line bg-white px-6 py-2 flex items-center justify-between text-[10px] font-mono uppercase tracking-wider opacity-50">
        <div className="flex gap-6">
          <span>Lat: 34.0522</span>
          <span>Lng: -118.2437</span>
          <span>Alt: 1,240m</span>
        </div>
        <div className="flex gap-6">
          <span>Source: NOAA / Global Forest Watch</span>
          <span>© 2024 Misa</span>
        </div>
      </footer>
    </div>
  );
}
