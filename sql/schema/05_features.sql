-- Feature layer support objects. The feature tables themselves are rebuilt by
-- `fraud features build` (sql/features/*.sql).

CREATE SCHEMA IF NOT EXISTS features;

CREATE TABLE IF NOT EXISTS features.builds (
    build_id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    feature_set_version TEXT        NOT NULL,
    dataset_hash        TEXT,
    dq_run_id           BIGINT,
    rows                BIGINT,
    seconds             NUMERIC,
    built_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Great-circle distance in km. Inputs here are country centroids, so the result has
-- COUNTRY precision: two payments in the same country are 0 km apart, whatever the city.
CREATE OR REPLACE FUNCTION features.haversine_km(
    lat1 DOUBLE PRECISION, lon1 DOUBLE PRECISION, lat2 DOUBLE PRECISION, lon2 DOUBLE PRECISION
) RETURNS DOUBLE PRECISION
LANGUAGE sql IMMUTABLE STRICT AS $$
    SELECT 2 * 6371 * asin(sqrt(
        power(sin(radians(lat2 - lat1) / 2), 2)
        + cos(radians(lat1)) * cos(radians(lat2)) * power(sin(radians(lon2 - lon1) / 2), 2)
    ))
$$;
