import sqlalchemy as sa
import mlbstatsapi
from .utils import Guts
from . import pitcher


class Stats:
    def __init__(self, url: sa._engine.URL):
        self.engine = sa.create_engine(url)
        self.mlb = mlbstatsapi.Mlb()
        self.guts = Guts(self.engine)
        self.pitcher = pitcher.PitchingCalculator(self.guts, self.mlb)

    def war(self, player_id: int, season: int):
        """
        WAR Claculator

        calculate pitching hitting base running and fielding war for a player in a specific season
        """
        return self.pitcher.pitching_war(player_id, season)
