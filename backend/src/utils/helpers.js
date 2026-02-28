// Error handling utilities
class AppError extends Error {
  constructor(message, statusCode) {
    super(message);
    this.statusCode = statusCode;
    this.status = `${statusCode}`.startsWith('4') ? 'fail' : 'error';
    this.isOperational = true;

    Error.captureStackTrace(this, this.constructor);
  }
}

// Async error wrapper
const catchAsync = (fn) => {
  return (req, res, next) => {
    fn(req, res, next).catch(next);
  };
};

// Validation helpers
const validateCoordinates = (lat, lon) => {
  if (lat < -90 || lat > 90) {
    throw new AppError('Invalid latitude. Must be between -90 and 90', 400);
  }
  if (lon < -180 || lon > 180) {
    throw new AppError('Invalid longitude. Must be between -180 and 180', 400);
  }
  return true;
};

const validateDateRange = (startDate, endDate) => {
  const start = new Date(startDate);
  const end = new Date(endDate);
  
  if (isNaN(start.getTime()) || isNaN(end.getTime())) {
    throw new AppError('Invalid date format', 400);
  }
  
  if (start > end) {
    throw new AppError('Start date must be before end date', 400);
  }
  
  return true;
};

// Response helpers
const sendSuccess = (res, data, statusCode = 200) => {
  res.status(statusCode).json({
    success: true,
    data
  });
};

const sendError = (res, message, statusCode = 500) => {
  res.status(statusCode).json({
    success: false,
    error: {
      message,
      status: statusCode
    }
  });
};

module.exports = {
  AppError,
  catchAsync,
  validateCoordinates,
  validateDateRange,
  sendSuccess,
  sendError
};
