import numpy as np
from functools import lru_cache
import pandas as pd
import sqlalchemy as sa


def runners_on_binary(runners_on: np.ndarray):
    runners_on[np.logical_or(runners_on == '', runners_on == 'F')] = 0
    runners_on[runners_on != '0'] = 1
    return runners_on.astype(int)


def S_3_element(values):
    """given the positions of runners on the base paths, returns correct row in the matrix."""
    product = pow(2, values[0]) * pow(3, values[1]) * pow(5, values[2])
    S_3 = np.array([1, 2, 3, 5, 6, 10, 15, 30])
    return np.where(S_3 == product)[0]


def S_3_list(values: np.ndarray):
    products = pow(2, values[:, 0]) * pow(3, values[:, 1]) * pow(5, values[:, 2])
    S_3 = np.array([1, 2, 3, 5, 6, 10, 15, 30])
    inds = []
    for product in products:
        inds.append(np.where(S_3 == product)[0][0])
    return np.array(inds)


def play_score(outs_scores: np.ndarray):
    """Determines number of runners scoring on a given play based on each runner's destination"""
    runner_dests = outs_scores[:, [4, 5, 6, 7]]  # Destinations of batter and runners on each base
    runner_dests[runner_dests <= 3] = 0  # Ignore anyone who doesn't score
    runner_dests[runner_dests > 3] = 1  # Everyone who scores is worth 1 run
    outs_scores[:, 4] = np.sum(runner_dests, axis=1)  # Sum of scores on each play
    return outs_scores


class Guts:
    def __init__(self, engine: sa._engine.URL):
        self.engine = engine

    @lru_cache(maxsize=10)
    def league_avg_pitching(self, season: int):
        query = sa.text('SELECT * FROM league_avg_pitching WHERE "Season" = :year;')
        df = pd.read_sql(
            query,
            params={"year": season},
            con=self.engine,
            index_col="Season",
        )
        if df.empty:
            raise ValueError(f"No league average pitching data for the {season} season")
        return df.iloc[0]

    @lru_cache(maxsize=10)
    def park_factors(self, season):
        query = sa.text('SELECT * FROM park_factors WHERE "Season" = :year;')
        df = pd.read_sql(
            query,
            params={"year": season},
            con=self.engine,
            index_col="Season",
        )
        if df.empty:
            raise ValueError(f"No league park factors for the {season} season")
        return df
