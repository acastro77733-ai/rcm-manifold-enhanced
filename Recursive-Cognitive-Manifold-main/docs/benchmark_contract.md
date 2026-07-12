# Benchmark contract

The benchmark contract records the metrics that must remain reproducible for the baseline release.

## Required metrics

- recall quality
- task retention
- recovery rate
- collapse frequency
- edge count
- region count
- runtime
- peak memory

## Output contract

Benchmark reports must be written as JSON with:

- metadata.seed
- metadata.generated_at only when include_timestamp is enabled
- benchmarks.<metric_name> payloads

## Regression policy

The CI pipeline compares current benchmark results against stored baselines. A regression is flagged if the tracked metric exceeds its allowed tolerance.
