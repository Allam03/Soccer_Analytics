-- =============================================================================
-- Soccer Analytics ML System -- PostgreSQL Schema
-- =============================================================================

-- -----------------------------------------------------------------------------
-- TEAMS
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS teams (
    team_id             SERIAL PRIMARY KEY,
    team_name           TEXT        NOT NULL,
    country             TEXT,
    statsbomb_team_id   INT         UNIQUE NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_teams_statsbomb ON teams (statsbomb_team_id);


-- -----------------------------------------------------------------------------
-- PLAYERS
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS players (
    player_id               SERIAL PRIMARY KEY,
    statsbomb_player_id     INT     UNIQUE,
    transfermarkt_player_id TEXT,
    player_name             TEXT    NOT NULL,
    norm_name               TEXT,
    nationality             TEXT,
    position                TEXT,
    date_of_birth           DATE
);

CREATE INDEX IF NOT EXISTS idx_players_statsbomb      ON players (statsbomb_player_id);
CREATE INDEX IF NOT EXISTS idx_players_norm_name      ON players (norm_name);
CREATE INDEX IF NOT EXISTS idx_players_transfermarkt  ON players (transfermarkt_player_id);


-- -----------------------------------------------------------------------------
-- MATCHES
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS matches (
    match_id            SERIAL PRIMARY KEY,
    statsbomb_match_id  INT         UNIQUE NOT NULL,
    match_date          DATE,
    home_team_id        INT         REFERENCES teams (team_id),
    away_team_id        INT         REFERENCES teams (team_id),
    home_score          INT,
    away_score          INT,
    competition         TEXT,
    season              TEXT,
    stadium_name        TEXT,
    stadium_lat         FLOAT,
    stadium_lng         FLOAT
);

CREATE INDEX IF NOT EXISTS idx_matches_date        ON matches (match_date);
CREATE INDEX IF NOT EXISTS idx_matches_competition ON matches (competition, season);
CREATE INDEX IF NOT EXISTS idx_matches_home_team   ON matches (home_team_id);
CREATE INDEX IF NOT EXISTS idx_matches_away_team   ON matches (away_team_id);


-- -----------------------------------------------------------------------------
-- WEATHER  (one row per match)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS weather (
    weather_id          SERIAL PRIMARY KEY,
    match_id            INT     NOT NULL REFERENCES matches (match_id) ON DELETE CASCADE,
    temperature_c       FLOAT,
    humidity_pct        FLOAT,
    wind_speed_kmh      FLOAT,
    precipitation_mm    FLOAT,
    weather_condition   TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_weather_match ON weather (match_id);


-- -----------------------------------------------------------------------------
-- INJURIES  (from Transfermarkt)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS injuries (
    injury_id       SERIAL PRIMARY KEY,
    player_id       INT     NOT NULL REFERENCES players (player_id) ON DELETE CASCADE,
    injury_type     TEXT,
    injury_date     DATE,
    return_date     DATE,
    matches_missed  INT,
    season          TEXT
);

CREATE INDEX IF NOT EXISTS idx_injuries_player ON injuries (player_id);
CREATE INDEX IF NOT EXISTS idx_injuries_dates  ON injuries (injury_date, return_date);


-- -----------------------------------------------------------------------------
-- PLAYER MATCH STATS  (one row per player x match)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS player_match_stats (
    stat_id                 SERIAL PRIMARY KEY,
    player_id               INT     NOT NULL REFERENCES players  (player_id),
    match_id                INT     NOT NULL REFERENCES matches  (match_id),
    team_id                 INT             REFERENCES teams     (team_id),
    weather_id              INT             REFERENCES weather   (weather_id),

    -- match context
    result                  TEXT,           -- 'win' | 'draw' | 'loss'

    -- attacking
    goals                   INT     DEFAULT 0,
    assists                 INT     DEFAULT 0,
    shots                   INT     DEFAULT 0,
    xg                      FLOAT   DEFAULT 0,
    xa                      FLOAT   DEFAULT 0,
    key_passes              INT     DEFAULT 0,

    -- passing
    passes_attempted        INT     DEFAULT 0,
    passes_completed        INT     DEFAULT 0,
    pass_accuracy           FLOAT   DEFAULT 0,
    progressive_passes      INT     DEFAULT 0,

    -- carrying
    carry_distance          FLOAT   DEFAULT 0,
    progressive_carries     INT     DEFAULT 0,

    -- dribbling
    dribbles_completed      INT     DEFAULT 0,

    -- defending
    tackles                 INT     DEFAULT 0,
    interceptions           INT     DEFAULT 0,
    clearances              INT     DEFAULT 0,
    pressures               INT     DEFAULT 0,

    -- discipline
    yellow_cards            INT     DEFAULT 0,
    red_cards               INT     DEFAULT 0,

    -- minutes / workload
    minutes_played          INT     DEFAULT 0,
    sub_minute              INT,            -- NULL = not subbed off

    -- injury risk features (pre-computed rolling window)
    days_since_last_injury  INT,            -- NULL = no prior injury on record
    matches_last_30_days    INT     DEFAULT 0,
    minutes_last_30_days    INT     DEFAULT 0,

    -- ML target label (Model 3)
    is_injured_next_30d     BOOLEAN DEFAULT FALSE,

    UNIQUE (player_id, match_id)
);

CREATE INDEX IF NOT EXISTS idx_pms_player   ON player_match_stats (player_id);
CREATE INDEX IF NOT EXISTS idx_pms_match    ON player_match_stats (match_id);
CREATE INDEX IF NOT EXISTS idx_pms_team     ON player_match_stats (team_id);
CREATE INDEX IF NOT EXISTS idx_pms_weather  ON player_match_stats (weather_id);


-- -----------------------------------------------------------------------------
-- PASS NETWORK EDGES  (per match, aggregated passer -> receiver counts)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pass_network_edges (
    edge_id         SERIAL PRIMARY KEY,
    match_id        INT     NOT NULL REFERENCES matches  (match_id) ON DELETE CASCADE,
    team_id         INT             REFERENCES teams     (team_id),
    passer_id       INT             REFERENCES players   (player_id),
    receiver_id     INT             REFERENCES players   (player_id),
    pass_count      INT     DEFAULT 1,
    avg_x_start     FLOAT,
    avg_y_start     FLOAT,
    avg_x_end       FLOAT,
    avg_y_end       FLOAT
);

CREATE INDEX IF NOT EXISTS idx_pne_match ON pass_network_edges (match_id);
CREATE INDEX IF NOT EXISTS idx_pne_team  ON pass_network_edges (match_id, team_id);


-- -----------------------------------------------------------------------------
-- VIEW: player age at time of match
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_player_match_age AS
SELECT
    pms.stat_id,
    pms.player_id,
    pms.match_id,
    m.match_date,
    p.date_of_birth,
    EXTRACT(YEAR FROM AGE(m.match_date, p.date_of_birth))::INT AS age_at_match
FROM player_match_stats pms
JOIN matches m ON m.match_id  = pms.match_id
JOIN players p ON p.player_id = pms.player_id
WHERE p.date_of_birth IS NOT NULL;