const predictionService = require('../services/prediction.service');

exports.predictRoute = async (req, res, next) => {
  try {
    const { speciesId, currentLocation, environmentalFactors } = req.body;
    
    const prediction = await predictionService.predictMigrationRoute(
      speciesId,
      currentLocation,
      environmentalFactors
    );
    
    res.json({ success: true, data: prediction });
  } catch (error) {
    next(error);
  }
};

exports.getFeatureAttribution = async (req, res, next) => {
  try {
    const { speciesId, predictionData } = req.body;
    
    const attribution = await predictionService.calculateFeatureAttribution(
      speciesId,
      predictionData
    );
    
    res.json({ success: true, data: attribution });
  } catch (error) {
    next(error);
  }
};

exports.explainPrediction = async (req, res, next) => {
  try {
    const { speciesId, prediction, factors } = req.body;
    
    const explanation = await predictionService.generateAIExplanation(
      speciesId,
      prediction,
      factors
    );
    
    res.json({ success: true, data: explanation });
  } catch (error) {
    next(error);
  }
};
