"""Stock Open Predictor — daily pipeline.

Pipeline stages (one module each):
    universe  -> which tickers to analyse
    data      -> download daily + premarket prices
    features  -> turn raw prices into signals per ticker
    scoring   -> combine signals into one score + human-readable reasons
    report    -> write the JSON file the iPhone app reads
    run       -> command-line entry point that runs all stages
"""

__version__ = "0.1.0"
