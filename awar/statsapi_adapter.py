import pandas as pd
import mlbstatsapi
import numpy as np


def fill_play_by_game(mlb:mlbstatsapi.Mlb, game_id: int, df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill the play by play data from 1 game into a pandas dataframe
    """
    plays = mlb.get_game_play_by_play(game_id)
    rows_list = []
    for play in plays.all_plays:
        rows_list.append(
            {
                "game_id": game_id,
                "inning": play.about.inning,
                "side": play.about.half_inning,
                "result": play.result.event_type,
                "home_score": play.result.home_score,
                "away_score": play.result.away_score,
                "balls": play.count.balls,
                "strikes": play.count.strikes,
                "outs": play.count.outs,
                "runner_on_1b": play.matchup.post_on_first.id if play.matchup.post_on_first is not None else None,
                "runner_on_2b": play.matchup.post_on_second.id if play.matchup.post_on_second is not None else None,
                "runner_on_3b": play.matchup.post_on_third.id if play.matchup.post_on_third is not None else None,
                "pitcher_id": play.matchup.pitcher.id,
                "hitter_id": play.matchup.batter.id,
            }
        )
    if df is None:
        return pd.DataFrame(rows_list)
    df = pd.concat([df, pd.DataFrame(rows_list)], ignore_index=True)
    return df


def get_play_by_play_games(mlb:mlbstatsapi.Mlb, year: int=None, date=None) -> pd.DataFrame:
    """
    Ingest the play-by-play data from games to a pandas dataframe.
    If no date is provided, all regular season games for the year will be ingested.
    """
    if date is not None:
        schedule =mlb.get_schedule(date, gameTypes="R")
    elif year is not None:
        schedule =mlb.get_schedule(
            start_date=f"{year}-1-1", end_date=f"{year}-12-31", gameTypes="R"
        )
    else:
        raise AttributeError("Either date or year must be set")
    df = pd.DataFrame()
    for date in schedule.dates:
        for game in date.games:
            df = fill_play_by_game(mlb, game.game_pk, df)
    return df

def runners_on_base(pbp:pd.DataFrame):
    runners_on = np.where(pbp[['runner_on_1b','runner_on_2b','runner_on_3b']].isna(), 0, 1).astype(np.bool_)
    return np.mean(runners_on, axis=0)