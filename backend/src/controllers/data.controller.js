const dataService = require('../services/data.service');

exports.getEnvironmentalData = async (req, res, next) => {
  try {
    const { lat, lon, radius, layers } = req.query;
    
    const data = await dataService.fetchEnvironmentalOverlays(
      parseFloat(lat),
      parseFloat(lon),
      parseFloat(radius),
      layers?.split(',')
    );
    
    res.json({ success: true, data });
  } catch (error) {
    next(error);
  }
};

exports.getClimateData = async (req, res, next) => {
  try {
    const { lat, lon, startDate, endDate } = req.query;
    
    const data = await dataService.fetchClimateData(lat, lon, startDate, endDate);
    res.json({ success: true, data });
  } catch (error) {
    next(error);
  }
};

exports.getForestCoverData = async (req, res, next) => {
  try {
    const { bounds } = req.query;
    
    const data = await dataService.fetchForestCoverData(JSON.parse(bounds));
    res.json({ success: true, data });
  } catch (error) {
    next(error);
  }
};

exports.getRealTimeEvents = async (req, res, next) => {
  try {
    const { eventTypes, bounds } = req.query;
    
    const events = await dataService.fetchRealTimeEvents(
      eventTypes?.split(','),
      bounds ? JSON.parse(bounds) : null
    );
    
    res.json({ success: true, data: events });
  } catch (error) {
    next(error);
  }
};
