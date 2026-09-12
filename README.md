# Football Prediction Engine

A modular football match prediction, analysis, validation, and decision system.

## Project Principles

- Data-driven and evidence-based analysis
- Statistical models as the foundation
- Controlled Machine Learning
- Walk-Forward and Out-of-Sample validation
- League-aware intelligence
- Integrity and match-risk screening
- Dynamic pre-match updates
- Multi-market analysis
- Decision ranking instead of probability-only prediction
- Explicit NO BET decision when evidence is insufficient

## Development Approach

The system is developed in controlled phases.

Phase 1:
- Goals
- Historical data
- Statistical prediction
- Backtesting
- Calibration
- Decision Engine foundation

Future phases:
- BTTS
- 1X2
- Over/Under
- Team Goals
- Corners
- Cards
- Shots
- Additional markets

## Architecture

Data → Features → Models → Predictions → Market Analysis
→ Integrity → League Intelligence → Decision Engine → Final Selection

## Methodology (Phase 1)

**Model type:** Point-in-time statistical model — league-baseline
Poisson with attack/defense strength blending (not classical ML yet).
Includes:
- Dixon-Coles correction for low-score correlation (goals only)
- Bayesian shrinkage toward league-neutral for teams with little history
- Recency-weighted historical averaging (exponential decay, 180-day half-life)
- Per-league calibrated `strength_weight` (see `LEAGUE_STRENGTH_WEIGHTS`)

**Cross-check signal:** an independent ELO rating model is used to
compute `model_agreement` for the Decision Engine.

**Validation:** Walk-forward, strictly point-in-time (no future-match
leakage), train/test split for calibration evaluation.

**Calibration:** Empirical monotonic binning (PAVA), evaluated with
Brier score and Expected Calibration Error (ECE).

**Markets covered so far:** Goals (1X2, BTTS, Over/Under), Shots
(match-total Over/Under, expected shots per team — experimental,
Poisson approximation; shot counts are typically over-dispersed,
Negative Binomial is a candidate future upgrade).

**Known limitations (tracked for future rounds):**
- Decision thresholds (0.78 / 0.68 in `DecisionEngine`) are not yet
  calibrated against historical decision outcomes.
- `strength_weight` is currently shared between goals and shots;
  a shots-specific calibration is a future improvement.
- Corners and cards markets are not yet implemented (data fields
  already exist in `HistoricalMatch`).
