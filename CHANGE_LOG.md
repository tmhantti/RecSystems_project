# Change Log

This file tracks the project's evolution, major updates, and methodological decisions made during the development. It provides a clear, human-readable timeline for contributors and evaluators.

## 25.10.2026
- Added long-tail distribution plots for users and items.
- Implemented data filtering based on long-tail analysis: filtered out users with fewer than 25 ratings and movies with fewer than 5 ratings.
- Added updated dataset statistics calculation after filtering.
- Refactored calculation logic into a dedicated reusable function (`print_dataset_statistics`).