// Validate and clamp training configuration

const DEFAULTS = {
  RANDOM_SEED: 42,
  TEST_SIZE: 0.2,
  THREADS: 4,
  DT_ENABLE: true,
  DT_MAX_DEPTH: 0,
  DT_MIN_SAMPLES_LEAF: 1,
  RF_ENABLE: true,
  RF_TREES: 250,
  RF_MAX_DEPTH: 0,
  RF_MIN_SAMPLES_LEAF: 1,
  LGBM_ENABLE: true,
  LGBM_N_ESTIMATORS: 1200,
  LGBM_REG_ALPHA: 0.0,
  LGBM_REG_LAMBDA: 0.0,
  MLP_ENABLE: false,
  MLP_HIDDEN: 48,
  MLP_EPOCHS: 150,
  MLP_PATIENCE: 25,
  SVR_ENABLE: true,
  SVR_C: 10.0,
  SVR_EPSILON: 0.1,
  RISK_HIGH_MAX: 3.30,
  RISK_MED_MAX: 3.50,
  GRADE_POINTS: {
    "A+": 4.0,
    "A": 3.75,
    "A-": 3.5,
    "B+": 3.25,
    "B": 3.0,
    "B-": 2.75,
    "C+": 2.5,
    "C": 2.25,
    "D": 2.0,
    "F": 0.0
  }
};

const BOUNDS = {
  TEST_SIZE: [0.1, 0.3],
  THREADS: [1, 16],
  DT_MAX_DEPTH: [1, 50],
  DT_MIN_SAMPLES_LEAF: [1, 50],
  RF_TREES: [50, 1000],
  RF_MAX_DEPTH: [1, 50],
  RF_MIN_SAMPLES_LEAF: [1, 50],
  LGBM_N_ESTIMATORS: [200, 4000],
  LGBM_REG_ALPHA: [0, 10],
  LGBM_REG_LAMBDA: [0, 10],
  MLP_HIDDEN: [16, 256],
  MLP_EPOCHS: [50, 600],
  MLP_PATIENCE: [10, 100],
  SVR_C: [0.1, 100],
  SVR_EPSILON: [0.001, 1]
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function parseBoolean(value, fallback) {
  if (value === undefined) return fallback;
  if (typeof value === 'string') return value.toLowerCase() === 'true';
  return Boolean(value);
}

export function validateConfig(inputConfig) {
  const config = { ...DEFAULTS };

  // Apply user values with clamping
  if (inputConfig.RANDOM_SEED !== undefined) {
    config.RANDOM_SEED = Math.floor(inputConfig.RANDOM_SEED);
  }

  if (inputConfig.TEST_SIZE !== undefined) {
    config.TEST_SIZE = clamp(inputConfig.TEST_SIZE, ...BOUNDS.TEST_SIZE);
  }

  if (inputConfig.THREADS !== undefined) {
    config.THREADS = Math.floor(clamp(inputConfig.THREADS, ...BOUNDS.THREADS));
  }

  if (inputConfig.DT_ENABLE !== undefined) {
    config.DT_ENABLE = parseBoolean(inputConfig.DT_ENABLE, config.DT_ENABLE);
  }

  if (inputConfig.DT_MAX_DEPTH !== undefined) {
    const value = Math.floor(inputConfig.DT_MAX_DEPTH);
    config.DT_MAX_DEPTH = value <= 0 ? 0 : Math.floor(clamp(value, ...BOUNDS.DT_MAX_DEPTH));
  }

  if (inputConfig.DT_MIN_SAMPLES_LEAF !== undefined) {
    config.DT_MIN_SAMPLES_LEAF = Math.floor(clamp(inputConfig.DT_MIN_SAMPLES_LEAF, ...BOUNDS.DT_MIN_SAMPLES_LEAF));
  }

  if (inputConfig.RF_ENABLE !== undefined) {
    config.RF_ENABLE = parseBoolean(inputConfig.RF_ENABLE, config.RF_ENABLE);
  }

  if (inputConfig.RF_TREES !== undefined) {
    config.RF_TREES = Math.floor(clamp(inputConfig.RF_TREES, ...BOUNDS.RF_TREES));
  }

  if (inputConfig.RF_MAX_DEPTH !== undefined) {
    const value = Math.floor(inputConfig.RF_MAX_DEPTH);
    config.RF_MAX_DEPTH = value <= 0 ? 0 : Math.floor(clamp(value, ...BOUNDS.RF_MAX_DEPTH));
  }

  if (inputConfig.RF_MIN_SAMPLES_LEAF !== undefined) {
    config.RF_MIN_SAMPLES_LEAF = Math.floor(clamp(inputConfig.RF_MIN_SAMPLES_LEAF, ...BOUNDS.RF_MIN_SAMPLES_LEAF));
  }

  if (inputConfig.LGBM_ENABLE !== undefined) {
    config.LGBM_ENABLE = parseBoolean(inputConfig.LGBM_ENABLE, config.LGBM_ENABLE);
  }

  if (inputConfig.LGBM_N_ESTIMATORS !== undefined) {
    config.LGBM_N_ESTIMATORS = Math.floor(clamp(inputConfig.LGBM_N_ESTIMATORS, ...BOUNDS.LGBM_N_ESTIMATORS));
  }

  if (inputConfig.LGBM_REG_ALPHA !== undefined) {
    config.LGBM_REG_ALPHA = clamp(inputConfig.LGBM_REG_ALPHA, ...BOUNDS.LGBM_REG_ALPHA);
  }

  if (inputConfig.LGBM_REG_LAMBDA !== undefined) {
    config.LGBM_REG_LAMBDA = clamp(inputConfig.LGBM_REG_LAMBDA, ...BOUNDS.LGBM_REG_LAMBDA);
  }

  if (inputConfig.MLP_ENABLE !== undefined) {
    config.MLP_ENABLE = parseBoolean(inputConfig.MLP_ENABLE, config.MLP_ENABLE);
  }

  if (inputConfig.MLP_HIDDEN !== undefined) {
    config.MLP_HIDDEN = Math.floor(clamp(inputConfig.MLP_HIDDEN, ...BOUNDS.MLP_HIDDEN));
  }

  if (inputConfig.MLP_EPOCHS !== undefined) {
    config.MLP_EPOCHS = Math.floor(clamp(inputConfig.MLP_EPOCHS, ...BOUNDS.MLP_EPOCHS));
  }

  if (inputConfig.MLP_PATIENCE !== undefined) {
    config.MLP_PATIENCE = Math.floor(clamp(inputConfig.MLP_PATIENCE, ...BOUNDS.MLP_PATIENCE));
  }

  if (inputConfig.SVR_ENABLE !== undefined) {
    config.SVR_ENABLE = parseBoolean(inputConfig.SVR_ENABLE, config.SVR_ENABLE);
  }

  if (inputConfig.SVR_C !== undefined) {
    config.SVR_C = clamp(inputConfig.SVR_C, ...BOUNDS.SVR_C);
  }

  if (inputConfig.SVR_EPSILON !== undefined) {
    config.SVR_EPSILON = clamp(inputConfig.SVR_EPSILON, ...BOUNDS.SVR_EPSILON);
  }

  if (inputConfig.RISK_HIGH_MAX !== undefined) {
    config.RISK_HIGH_MAX = parseFloat(inputConfig.RISK_HIGH_MAX);
  }

  if (inputConfig.RISK_MED_MAX !== undefined) {
    config.RISK_MED_MAX = parseFloat(inputConfig.RISK_MED_MAX);
  }

  // Validate RISK thresholds
  if (config.RISK_HIGH_MAX > config.RISK_MED_MAX) {
    throw new Error('RISK_HIGH_MAX must be <= RISK_MED_MAX');
  }

  // Validate GRADE_POINTS
  if (inputConfig.GRADE_POINTS) {
    const gradePoints = {};
    for (const [key, value] of Object.entries(inputConfig.GRADE_POINTS)) {
      const numValue = parseFloat(value);
      if (isNaN(numValue) || numValue < 0 || numValue > 10) {
        throw new Error(`Invalid grade point value for ${key}: must be between 0 and 10`);
      }
      gradePoints[key] = numValue;
    }
    config.GRADE_POINTS = gradePoints;
  }

  return config;
}
