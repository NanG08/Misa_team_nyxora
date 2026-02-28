const migrationService = require('../services/migration.service');

exports.getMigrationData = async (req, res, next) => {
  try {
    const { speciesId } = req.params;
    const { startDate, endDate } = req.query;
    
    const data = await migrationService.fetchMigrationData(speciesId, { startDate, endDate });
    res.json({ success: true, data });
  } catch (error) {
    next(error);
  }
};

exports.getMigrationPoints = async (req, res, next) => {
  try {
    const { speciesId, startTime, endTime } = req.query;
    
    const points = await migrationService.getMigrationPoints(speciesId, startTime, endTime);
    res.json({ success: true, data: points });
  } catch (error) {
    next(error);
  }
};

exports.getRiskScore = async (req, res, next) => {
  try {
    const { speciesId } = req.params;
    
    const riskScore = await migrationService.calculateRiskScore(speciesId);
    res.json({ success: true, data: riskScore });
  } catch (error) {
    next(error);
  }
};

exports.getAllSpecies = async (req, res, next) => {
  try {
    const species = await migrationService.getAllTrackedSpecies();
    res.json({ success: true, data: species });
  } catch (error) {
    next(error);
  }
};
