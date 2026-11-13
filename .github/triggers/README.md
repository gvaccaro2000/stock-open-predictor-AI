Backup triggers. GitHub's own schedule doesn't always start workflows, so a scheduled
task writes the current time into these files on weekdays:

- predict:  08:05 New York time (14:05 in Madrid) -> runs Daily predictions
- evaluate: 16:35 New York time (22:35 in Madrid) -> runs Track record

Pushing a change to either file starts the matching workflow.
- longterm: weekly (also Sundays via GitHub schedule) -> runs Long-term ideas
