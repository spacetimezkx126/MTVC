# Reproducibility

Use the pinned environment, matching dataset contents, token vocabularies and the checkpoint-specific model recipe. Evaluate the selected best checkpoint directly; retraining from the same seed is a separate experiment.

Model initialization uses a fixed construction order, including frozen companion modules. Their construction preserves the parameter-registration and optimizer order used during training. This behavior is unconditional and has no public compatibility switch.

A full training resume restores saved model, optimizer and sampling state. A weights-only resume starts a new optimization trajectory. Checkpoint inference reports actual ACC/MCC separately from stored references and from any explicit acceptance tolerances.

For PEN and StockNet, sampling can remain active during evaluation. Independent replay requires the original evaluation state, the original batch configuration and correct mapping from the training GPU stream to the inference GPU. Older PEN archives without that state cannot guarantee their historical logged metrics. New saves retain the evaluation state; existing checkpoint files are never silently rewritten.

For portfolio results, preserve the dated prediction CSVs and the selected-seed manifest. Calendar intersection occurs before compounding, the initial wealth point is excluded from the annualization day count, and Sharpe ratio uses sample standard deviation as specified in the portfolio documentation.
