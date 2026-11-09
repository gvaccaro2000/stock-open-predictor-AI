import type { Report } from './types';

// The backend publishes this file every trading morning via GitHub Pages.
export const REPORT_URL =
  'https://gvaccaro2000.github.io/stock-open-predictor-AI/data/today.json';

export async function fetchReport(): Promise<Report> {
  // Cache-buster so we always get the latest morning file.
  const res = await fetch(`${REPORT_URL}?t=${Date.now()}`);
  if (!res.ok) throw new Error(`Server returned ${res.status}`);
  const data = (await res.json()) as Report;
  if (data.schema_version !== 1) {
    throw new Error('This app version is out of date. Please update it.');
  }
  return data;
}

export const fmtPct = (x: number | null, digits = 1) =>
  x == null ? '—' : `${x > 0 ? '+' : ''}${x.toFixed(digits)}%`;

export const fmtPrice = (x: number | null) => (x == null ? '—' : `$${x.toFixed(2)}`);
