args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 3) {
  stop("Usage: ffanalytics_projections.R <season> <scoring.json> <output.csv>")
}
if (!requireNamespace("ffanalytics", quietly = TRUE)) {
  stop("The ffanalytics R package is required. See README.md for installation instructions.")
}

season <- as.integer(args[[1]])
scoring_input <- jsonlite::fromJSON(args[[2]], simplifyVector = TRUE)
output <- args[[3]]
positions <- c("QB", "RB", "WR", "TE", "DST")
sources <- c("CBS", "ESPN", "FantasyPros", "FantasySharks", "FFToday", "NumberFire", "NFL", "RTSports")

scoring <- ffanalytics::scoring
set_rule <- function(group, rule, sleeper_name) {
  value <- scoring_input[[sleeper_name]]
  if (!is.null(value)) {
    scoring[[group]][[rule]] <<- as.numeric(value)
  }
}
set_rule("pass", "pass_yds", "pass_yd")
set_rule("pass", "pass_tds", "pass_td")
set_rule("pass", "pass_int", "pass_int")
set_rule("rush", "rush_yds", "rush_yd")
set_rule("rush", "rush_tds", "rush_td")
set_rule("rec", "rec", "rec")
set_rule("rec", "rec_yds", "rec_yd")
set_rule("rec", "rec_tds", "rec_td")
set_rule("misc", "fumbles_lost", "fum_lost")

scraped <- ffanalytics::scrape_data(
  src = sources,
  pos = positions,
  season = season,
  week = 0
)
projections <- ffanalytics::projections_table(scraped, scoring_rules = scoring, avg_type = "weighted")
projections <- ffanalytics::add_player_info(projections)

result <- projections[, c("first_name", "last_name", "team", "position", "points", "points_vor", "rank", "pos_rank")]
utils::write.csv(result, output, row.names = FALSE, na = "")
