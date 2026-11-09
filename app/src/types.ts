// Mirrors backend/src/predictor/report.py (SCHEMA_VERSION = 1).
// If the backend JSON format changes, update this file too.

export type Reason = {
  text: string;
  impact: 'positive' | 'negative';
  kind?: 'news';
};

export type NewsInfo = {
  label: 'positive' | 'negative' | 'neutral';
  score: number;
  importance: 'low' | 'medium' | 'high';
  event: string;
  summary: string; // written by Claude
  items: { title: string; source: string; url: string | null; published: string; sentiment: string }[];
};

export type Pick = {
  rank: number;
  ticker: string;
  name?: string | null; // company name, e.g. "Apple"
  score: number; // 0-100 percentile rank vs the other stocks today
  raw_score?: number | null; // weighted signal sum, roughly -3..+3
  prev_close: number | null;
  premarket_price: number | null;
  premarket_gap_pct: number | null;
  momentum_5d_pct: number | null;
  trend_vs_ma50_pct: number | null;
  rsi_14: number | null;
  rel_volume_1d: number | null;
  summary?: string;   // 1-2 sentence explanation
  why_short?: string; // very short version for list rows
  reasons: Reason[];
  news?: NewsInfo | null;
  sector?: string | null;
  size?: string | null; // Large cap / Mid cap / Small cap
  history?: { start: string; end: string; closes: number[] } | null; // last 90 days
  levels?: Record<string, number | null>;
  atr?: number | null;
  gap_atr?: number | null;
  premarket_rvol?: number | null;
  rel_strength_pct?: number | null;
};

export type Report = {
  schema_version: number;
  generated_at: string;
  trading_date: string;
  universe_size: number;
  has_premarket_data: boolean;
  // Optional so older files without it still load.
  run_window?: 'premarket' | 'market_hours' | 'outside_premarket' | 'weekend';
  model: string;
  disclaimer: string;
  picks: Pick[];
};
