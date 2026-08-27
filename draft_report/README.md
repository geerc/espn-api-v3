# Draft Report

Generates commentary-free post-draft rankings for Sleeper fantasy football leagues. Teams are ranked by their highest-projected legal starting lineup using `ffanalytics` season projections. The report includes league-ranked positional radar charts plus each team's biggest draft reach and value.

## Setup

```shell
python3 -m venv venv
venv/bin/pip install -r requirements.txt -r requirements-dev.txt
Rscript -e 'if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes", repos = "https://cloud.r-project.org"); remotes::install_github("FantasyFootballAnalytics/ffanalytics", upgrade = "never")'
```

## Generate a report

```shell
venv/bin/python -m draft_report.sleeper_draft_report 1388595531374157824
```

Before the real draft is complete, use the included seeded dummy snake draft:

```shell
venv/bin/python -m draft_report.sleeper_draft_report 1388595531374157824 \
  --dummy-draft draft_report/tests/fixtures/dummy_sleeper_draft.json
```

By default, reports and radar images are written to `draft_report/reports/<season>/` inside this repository. Use `--output PATH` to choose another destination. Use `--projections PATH` to reuse an existing `ffanalytics` CSV.

## Optional AI commentary

Reports remain commentary-free by default. To add a concise statistical assessment for each team, create a local `.env` file in the repository root:

```shell
cp .env.example .env
```

Open `.env`, replace the placeholder with your key, and save it. Then enable commentary when generating the report:

```shell
venv/bin/sleeper-draft-report 1388595531374157824 --ai-commentary
```

The command automatically loads `.env` from the repository root. The file is excluded by `.gitignore`, so it will not be committed. You can optionally set `OPENAI_MODEL` in the same file or use `--ai-model MODEL` to override the default `gpt-5-mini` model.
