-- Layered schemas. Data only moves forward: raw -> core -> features -> scoring -> monitoring.
--   raw        : source extract exactly as delivered (text, no constraints)
--   core       : validated model with PK / FK / CHECK constraints
--   features   : point-in-time feature tables          (Phase 5)
--   scoring    : model predictions from API and batch   (Phase 10)
--   monitoring : drift and performance metrics          (Phase 12)
--   sim        : simulator ground truth; never read by features or models

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS scoring;
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE SCHEMA IF NOT EXISTS sim;
