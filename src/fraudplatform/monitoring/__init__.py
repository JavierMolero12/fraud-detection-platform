"""Monitoring of the deployed model: input and prediction drift, delayed performance, alerts.

Everything here uses only what would be known on the monitoring date. The simulator's ground
truth is read by monitoring.evaluation alone, to judge the alerts afterwards.
"""
