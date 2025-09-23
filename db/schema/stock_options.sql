-- =========================
-- Core types & enums
-- =========================
CREATE TYPE option_type AS ENUM ('CALL', 'PUT');
CREATE TYPE exercise_style AS ENUM ('AMERICAN', 'EUROPEAN');
CREATE TYPE security_type AS ENUM ('EQUITY', 'ETF', 'INDEX');

-- =========================
-- Reference data
-- =========================
CREATE TABLE exchanges (
  exchange_id      SERIAL PRIMARY KEY,
  code             TEXT UNIQUE NOT NULL,         -- e.g., CBOE = 'CBOE', ISE, NYSE, NASDAQ
  mic              TEXT UNIQUE,                  -- market identifier code if you track it
  name             TEXT NOT NULL
);

CREATE TABLE securities (
  security_id      SERIAL PRIMARY KEY,
  symbol           TEXT NOT NULL,                -- e.g., 'AAPL', 'SPX'
  name             TEXT,
  type             security_type NOT NULL,
  exchange_id      INT REFERENCES exchanges(exchange_id) ON UPDATE CASCADE,
  UNIQUE (symbol, type)
);

-- Option “root” (OCC root) ties contracts to an underlying
CREATE TABLE option_roots (
  root_id          SERIAL PRIMARY KEY,
  root_symbol      TEXT NOT NULL,                -- OCC root, e.g., 'AAPL', 'SPX', 'SPY'
  underlying_id    INT NOT NULL REFERENCES securities(security_id) ON UPDATE CASCADE,
  default_multiplier INT NOT NULL DEFAULT 100,   -- usually 100 shares
  deliverable_text TEXT,                         -- for non-standard deliverables
  is_index         BOOLEAN GENERATED ALWAYS AS (
                    (SELECT s.type = 'INDEX'::security_type FROM securities s WHERE s.security_id = underlying_id)
                  ) STORED,
  UNIQUE (root_symbol, underlying_id)
);

-- Contract expirations per root (helps de-dupe expiries across many strikes)
CREATE TABLE option_expirations (
  expiration_id    SERIAL PRIMARY KEY,
  root_id          INT NOT NULL REFERENCES option_roots(root_id) ON UPDATE CASCADE ON DELETE CASCADE,
  expiration_date  DATE NOT NULL,
  settlement_date  DATE,                         -- cash vs. physical, e.g., SPX uses AM/PM settlement rules
  UNIQUE (root_id, expiration_date)
);

-- =========================
-- Contracts
-- =========================
CREATE TABLE option_contracts (
  contract_id      BIGSERIAL PRIMARY KEY,
  root_id          INT NOT NULL REFERENCES option_roots(root_id) ON UPDATE CASCADE ON DELETE CASCADE,
  expiration_id    INT NOT NULL REFERENCES option_expirations(expiration_id) ON UPDATE CASCADE ON DELETE CASCADE,
  strike           NUMERIC(12,4) NOT NULL,
  type             option_type NOT NULL,
  exercise_style   exercise_style NOT NULL DEFAULT 'AMERICAN',
  multiplier       INT NOT NULL,                 -- usually 100; can differ for adjusted/non-standard
  is_standard      BOOLEAN NOT NULL DEFAULT TRUE,
  listing_date     DATE,                         -- when it first appeared
  occ_symbol       TEXT UNIQUE,                  -- Full OCC (e.g., AAPL  250117C00190000)
  contract_symbol  TEXT,                         -- broker/ticker format if you track it
  CHECK (strike > 0),

  -- Uniqueness across the canonical contract identity:
  UNIQUE (root_id, expiration_id, type, strike)
);

-- Helpful: keep this consistent with roots’ default if you don’t ingest per-contract multiplier
ALTER TABLE option_contracts
  ALTER COLUMN multiplier SET DEFAULT 100;

-- =========================
-- Underlying prices (intraday ticks or snapshots)
-- Consider RANGE partitioning by ts if high volume.
-- =========================
CREATE TABLE underlying_prices (
  security_id      INT NOT NULL REFERENCES securities(security_id) ON UPDATE CASCADE ON DELETE CASCADE,
  ts               TIMESTAMPTZ NOT NULL,
  bid              NUMERIC(18,6),
  ask              NUMERIC(18,6),
  last             NUMERIC(18,6),
  volume           BIGINT,
  PRIMARY KEY (security_id, ts)
);
CREATE INDEX underlying_prices_ts_desc_idx ON underlying_prices (security_id, ts DESC);

-- =========================
-- Quotes & greeks (intraday)
-- Consider monthly RANGE partitioning on ts for scale.
-- =========================
CREATE TABLE option_quotes (
  contract_id      BIGINT NOT NULL REFERENCES option_contracts(contract_id) ON UPDATE CASCADE ON DELETE CASCADE,
  ts               TIMESTAMPTZ NOT NULL,
  bid              NUMERIC(18,6),
  ask              NUMERIC(18,6),
  last             NUMERIC(18,6),
  mark             NUMERIC(18,6),
  bid_size         BIGINT,
  ask_size         BIGINT,
  volume           BIGINT,
  open_interest    BIGINT,                       -- sometimes only reliable at EOD; still useful intraday
  iv               NUMERIC(12,8),                -- implied vol (as decimal, e.g., 0.2456)
  delta            NUMERIC(12,8),
  gamma            NUMERIC(12,8),
  theta            NUMERIC(12,8),
  vega             NUMERIC(12,8),
  rho              NUMERIC(12,8),
  PRIMARY KEY (contract_id, ts)
);
CREATE INDEX option_quotes_ts_desc_idx ON option_quotes (contract_id, ts DESC);
CREATE INDEX option_quotes_iv_idx ON option_quotes (contract_id, iv);

-- Latest quote per contract (fast DISTINCT ON)
CREATE VIEW option_latest_quotes AS
SELECT DISTINCT ON (q.contract_id)
  q.contract_id, q.ts, q.bid, q.ask, q.last, q.mark, q.volume, q.open_interest,
  q.iv, q.delta, q.gamma, q.theta, q.vega, q.rho
FROM option_quotes q
ORDER BY q.contract_id, q.ts DESC;

-- =========================
-- End-of-day metrics (stable OI, OHLC, greeks if you vendor them)
-- =========================
CREATE TABLE option_eod (
  contract_id      BIGINT NOT NULL REFERENCES option_contracts(contract_id) ON UPDATE CASCADE ON DELETE CASCADE,
  trading_date     DATE NOT NULL,
  open             NUMERIC(18,6),
  high             NUMERIC(18,6),
  low              NUMERIC(18,6),
  close            NUMERIC(18,6),
  volume           BIGINT,
  open_interest    BIGINT,
  iv_close         NUMERIC(12,8),
  delta_close      NUMERIC(12,8),
  gamma_close      NUMERIC(12,8),
  theta_close      NUMERIC(12,8),
  vega_close       NUMERIC(12,8),
  rho_close        NUMERIC(12,8),
  PRIMARY KEY (contract_id, trading_date)
);
CREATE INDEX option_eod_open_interest_idx ON option_eod (trading_date, open_interest DESC);

-- Underlying EOD (optional, mirrors above)
CREATE TABLE underlying_eod (
  security_id      INT NOT NULL REFERENCES securities(security_id) ON UPDATE CASCADE ON DELETE CASCADE,
  trading_date     DATE NOT NULL,
  open             NUMERIC(18,6),
  high             NUMERIC(18,6),
  low              NUMERIC(18,6),
  close            NUMERIC(18,6),
  volume           BIGINT,
  PRIMARY KEY (security_id, trading_date)
);

-- =========================
-- Corporate actions (splits, special dividends) for adjustment logic
-- =========================
CREATE TABLE corporate_actions (
  action_id        BIGSERIAL PRIMARY KEY,
  security_id      INT NOT NULL REFERENCES securities(security_id) ON UPDATE CASCADE ON DELETE CASCADE,
  ex_date          DATE NOT NULL,
  action_type      TEXT NOT NULL,                -- 'SPLIT', 'DIVIDEND', 'SPINOFF', etc.
  ratio_n          NUMERIC(12,6),                -- e.g., split 2:1 -> 2
  ratio_d          NUMERIC(12,6),                -- denominator -> 1
  cash_amount      NUMERIC(18,6),                -- dividends
  notes            TEXT
);
CREATE INDEX corporate_actions_exdate_idx ON corporate_actions (security_id, ex_date);

-- =========================
-- Convenience chain view (contracts joined with identifiers)
-- =========================
CREATE VIEW option_chain AS
SELECT
  c.contract_id,
  r.root_symbol,
  s.symbol AS underlying_symbol,
  e.expiration_date,
  c.type,
  c.exercise_style,
  c.strike,
  c.multiplier,
  c.is_standard,
  c.occ_symbol,
  c.contract_symbol
FROM option_contracts c
JOIN option_roots r       ON r.root_id = c.root_id
JOIN securities s         ON s.security_id = r.underlying_id
JOIN option_expirations e ON e.expiration_id = c.expiration_id;

-- =========================
-- Helpful indexes for chain lookups
-- =========================
CREATE INDEX option_contracts_lookup_idx
  ON option_contracts (root_id, expiration_id, type, strike);

CREATE INDEX option_chain_expiry_idx
  ON option_expirations (root_id, expiration_date);

-- =========================
-- Comments (self-documentation)
-- =========================
COMMENT ON TABLE option_contracts IS 'Canonical OCC option contracts keyed by (root, expiry, type, strike).';
COMMENT ON COLUMN option_contracts.occ_symbol IS 'Full OCC 21-char sym if available (root, yymmdd, C/P, strike*1000).';
COMMENT ON COLUMN option_quotes.iv IS 'Implied volatility in decimal (0.25 = 25%).';
