import { GoogleGenAI, Type } from "@google/genai";
import { MigrationStory, PredictionResult, ScenarioResult } from "../types";

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || '' });

export const geminiService = {
  async generateMigrationStories(species: string): Promise<MigrationStory[]> {
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: `Generate 3 short, narrative-driven migration stories for the ${species}. 
      Each story should focus on a specific environmental challenge (climate change, deforestation, etc.).
      Include tags for event types (e.g., #heatwave, #wildfire).
      Format the output as a JSON array of objects with id, title, content, species, date, impactLevel (low, medium, high), and tags.`,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              id: { type: Type.STRING },
              title: { type: Type.STRING },
              content: { type: Type.STRING },
              species: { type: Type.STRING },
              date: { type: Type.STRING },
              impactLevel: { type: Type.STRING, enum: ["low", "medium", "high"] },
              tags: { type: Type.ARRAY, items: { type: Type.STRING } },
            },
            required: ["id", "title", "content", "species", "date", "impactLevel", "tags"],
          },
        },
      },
    });

    try {
      return JSON.parse(response.text || "[]");
    } catch (e) {
      console.error("Failed to parse stories", e);
      return [];
    }
  },

  async predictMigrationShift(species: string, factors: string[]): Promise<PredictionResult> {
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: `Predict the migration shift for ${species} given these factors: ${factors.join(", ")}.
      Provide a predicted route (3-4 lat/lng points), a confidence score (0-1), a reasoning explanation, 
      and a feature attribution list showing the importance of each factor (0-1).
      Format as JSON.`,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            species: { type: Type.STRING },
            predictedRoute: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  lat: { type: Type.NUMBER },
                  lng: { type: Type.NUMBER },
                },
                required: ["lat", "lng"],
              },
            },
            confidence: { type: Type.NUMBER },
            reasoning: { type: Type.STRING },
            featureAttribution: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  factor: { type: Type.STRING },
                  importance: { type: Type.NUMBER },
                },
                required: ["factor", "importance"],
              },
            },
          },
          required: ["species", "predictedRoute", "confidence", "reasoning", "featureAttribution"],
        },
      },
    });

    try {
      return JSON.parse(response.text || "{}");
    } catch (e) {
      console.error("Failed to parse prediction", e);
      throw e;
    }
  },

  async simulateScenario(species: string, scenario: string): Promise<ScenarioResult> {
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: `Simulate the impact of the following scenario on ${species}: "${scenario}".
      Provide a summary of the impact, projected range shift in km, and extinction risk increase percentage.
      Format as JSON.`,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            scenario: { type: Type.STRING },
            impactSummary: { type: Type.STRING },
            projectedRangeShift: { type: Type.NUMBER },
            extinctionRiskIncrease: { type: Type.NUMBER },
          },
          required: ["scenario", "impactSummary", "projectedRangeShift", "extinctionRiskIncrease"],
        },
      },
    });

    try {
      return JSON.parse(response.text || "{}");
    } catch (e) {
      console.error("Failed to parse scenario", e);
      throw e;
    }
  }
};
