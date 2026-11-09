import os.path
import mlbstatsapi
import pandas
import pandas as pd
from datetime import datetime
import numpy as np
import os
from utils import Guts


class PitchingCalculator:

    def __init__(
        self,
        guts: Guts,
        mlb_stats: mlbstatsapi.Mlb,
    ):
        """
        Manages calculating pitcher statistics

        Args:
            guts (Guts): guts statisics
            mlb_stats: (mlbstatsapi.Mlb) mlbstatsapi retriever
        """
        self.guts = guts
        self.division_fipr9s = {}
        self.mlbstats = mlb_stats
        self.fipr9_devisions = dict()
        self.fipr9_league = dict()

    def fipc(self, season: int) -> float:
        """
        FIP constant for a given season
        Defined from https://library.fangraphs.com/pitching/fip/

        args:
            season (int): the MLB season to get fip the constant for
        Returns
            float: The fip constant for the requested season.
        """
        season_data = self.guts.league_avg_pitching(season)
        return (
            season_data["ERA"]
            - (
                13 * season_data["HR"]
                + 3 * (season_data["BB"] + season_data["HBP"])
                - 2 * season_data["SO"]
            )
            / season_data["IP"]
        )

    def __reafip_consts(self, season) -> pd.DataFrame:
        pass

    def __reafip(self, player_id: int, season: int) -> float:
        """
        Run Expectancy Adjusted FIP

        this statistic makes an attempt to adjust fip so instead of using static constants FIP the value of home runs,
        walks/hit by pitch, and strikeouts are computed using the per-season run expectancy data.
        """
        player_data = self.mlbstats.get_player_stats(
            player_id,
            stats=["season", "seasonAdvanced"],
            groups=["pitching"],
            season=season,
        )
        basic_stats = player_data["pitching"]["season"].splits[0].stat
        consts = self.__reafip_consts(season)
        return (
            consts["hr"] * basic_stats.home_runs
            + consts["walks"] * (basic_stats.base_on_balls + basic_stats.hit_by_pitch)
            - consts["so"] * basic_stats.strikeouts
        ) / float(basic_stats.innings_pitched)

    def fip(self, player_id: int, season: int) -> float:
        """
        Fielding Independent Pitching
        Defined from https://library.fangraphs.com/pitching/fip/

        Args:
            player_id (int): MLB stats api player id for desired player
            season: MLB season yaar to calculate FIP for
        Returns:
            float: fip
        """
        player_data = self.mlbstats.get_player_stats(
            player_id,
            stats=["season", "seasonAdvanced"],
            groups=["pitching"],
            season=season,
        )
        basic_stats = player_data["pitching"]["season"].splits[0].stat
        return (
            13 * basic_stats.home_runs
            + 3 * (basic_stats.base_on_balls + basic_stats.hit_by_pitch)
            - 2 * basic_stats.strikeouts
        ) / float(basic_stats.innings_pitched) + self.fipc(season)

    def fip_team(self, team_id: int, season: int):
        team_stats = self.mlbstats.get_team_stats(
            team_id,
            stats=["season", "seasonAdvanced"],
            groups=["pitching"],
            **{"season": season},
        )
        pitching_stats = team_stats["pitching"]["season"].splits[0].stat
        return (
            (
                13 * pitching_stats.home_runs
                + 3 * (pitching_stats.base_on_balls + pitching_stats.hit_by_pitch)
                - 2 * pitching_stats.strikeouts
            )
            / float(pitching_stats.innings_pitched)
        ) + self.fipc(season)

    def __calc_fipr9_divisions(self, season: int) -> dict:
        """
        fipr9 for all divisions for a given season
        """
        if season in self.fipr9_devisions.keys():
            return self.fipr9_devisions[season]
        teams = self.mlbstats.get_teams(sport_id=1)
        divisions_fipr9s = {}
        league_avg_pitching = self.guts.league_avg_pitching(season)
        fip_adj = (
            league_avg_pitching.loc["R"] / league_avg_pitching.loc["IP"]
        ) * 9 - league_avg_pitching.loc["ERA"]
        for team in teams:
            fipr9 = self.fip_team(team.id, season) + fip_adj
            if team.division.id not in divisions_fipr9s:
                divisions_fipr9s[team.division.id] = [fipr9]
            else:
                divisions_fipr9s[team.division.id].append(fipr9)
        fipr9s = {}
        for key in divisions_fipr9s.keys():
            fipr9s[key] = np.array(divisions_fipr9s[key]).mean()
        self.fipr9_devisions[season] = fipr9s
        return fipr9s

    def calc_fipr9_league(self, season: int):
        """
        calculate the league average fipr9 for a given season
        """
        if season in self.fipr9_league.keys():
            return self.fipr9_league[season]
        teams = self.mlbstats.get_teams(sport_id=1)
        team_fipr9s = np.empty(30)
        league_avg_pitching = self.guts.league_avg_pitching(season)
        fip_adj = (league_avg_pitching["R"] / league_avg_pitching["IP"]) * 9 - league_avg_pitching[
            "ERA"
        ]
        for i, team in enumerate(teams):
            fipr9 = self.fip_team(team.id, season) + fip_adj
            team_fipr9s[i] = fipr9
        fipr9_season = np.mean(team_fipr9s)
        self.fipr9_league[season] = fipr9_season
        return fipr9_season

    def pitching_war(self, player_id: int, season: int) -> float:
        """
        Calculate WAR for pitchers
        https://library.fangraphs.com/war/calculating-war-pitchers/
        leverage index is not implemented yet.
        Instead of adjusting for league average, we adjust for division averages

        Args:
            player_id (int): MLB stats api id for desired player
            season: season to calculate war for
        """

        player_data = self.mlbstats.get_player_stats(
            player_id,
            stats=["season", "seasonAdvanced"],
            groups=["pitching"],
            season=season,
        )
        if player_data == {}:
            return 0.0
        park_factors = self.guts.park_factors(season)
        team_id = player_data["pitching"]["season"].splits[0].team.id
        team = self.mlbstats.get_team(team_id)
        basic_stats = player_data["pitching"]["season"].splits[0].stat
        league_avg_pitching = self.guts.league_avg_pitching(season)
        # league ra/9
        lg_ra9 = (league_avg_pitching["R"] / league_avg_pitching["IP"]) * 9
        # fip scaled to ra/9
        fip_adjust = lg_ra9 - league_avg_pitching["ERA"]
        fipr9 = self.fip(player_id, season) + fip_adjust
        # park factor adjusted fipr9
        pfipr9 = fipr9 / (park_factors[park_factors["Team"] == team.team_name]["FIP"].item() / 100)

        # Runs Above Average Per 9 scaled to division
        ip = float(basic_stats.innings_pitched)

        if season > 2022:
            # for the 2023 when inter-division games were reduced from 76 to 13
            # and interleague were increased from 20 to 46
            raap9 = self.calc_fipr9_league(season) - pfipr9
            # Dynamic Runs Per Win
            d_rpw = (
                (
                    ((18 - ip / basic_stats.games_pitched) * self.calc_fipr9_league(season))
                    + ((ip / basic_stats.games_pitched) * pfipr9)
                )
                / 18
                + 2
            ) * 1.5
        else:
            raap9 = self.__calc_fipr9_divisions(season)[team.division.id] - pfipr9
            # Dynamic Runs Per Win
            d_rpw = (
                (
                    (
                        (18 - ip / basic_stats.games_pitched)
                        * self.__calc_fipr9_divisions(season)[team.division.id]
                    )
                    + ((ip / basic_stats.games_pitched) * pfipr9)
                )
                / 18
                + 2
            ) * 1.5
        # wins per game above average
        wpgaa = raap9 / d_rpw
        # replacement level
        rl = 0.03 * (1 - basic_stats.games_started / basic_stats.games_played) + 0.12 * (
            basic_stats.games_started / basic_stats.games_played
        )
        # wins per game above replacement
        wpgar = wpgaa + rl
        war_p = wpgar * (ip / 9)
        # leverage index
        # li_multiplier = (1+gm_li)/2
        return war_p


if __name__ == "__main__":
    import sqlalchemy as sa
    import os

    url = sa.URL.create(
        drivername="postgresql",
        username=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database="awar",
    )
    engine = sa.create_engine(url)
    mlb = mlbstatsapi.Mlb()
    guts = Guts(engine)
    pitcher_stats = PitchingCalculator(guts, mlb)
    name = "Shohei Ohtani"
    player_id = mlb.get_people_id(name)[0]
    fip = pitcher_stats.fip(player_id, 2025)
    print(f"{name} fip {fip}")
    war = pitcher_stats.pitching_war(player_id, season=2025)
    print(f"{name} pitching war {war}")
