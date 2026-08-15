import os.path
import mlbstatsapi
import pandas
import pandas as pd
from datetime import datetime
import numpy as np
import os


class PitchingCalculator:
    """
    Pitcher stats calculator
    equations credit to Piper Slowinski at Fangraphs
    """

    def __init__(
        self,
        league_avg_pitching: pandas.DataFrame,
        park_factors: pandas.DataFrame,
        mlb_stats: mlbstatsapi.Mlb,
    ):
        self.pitching = league_avg_pitching
        self.park_factors = park_factors
        self.division_fipr9s = {}
        self.mlbstats = mlb_stats
        self.fipr9_devisions = dict()
        self.fipr9_league = dict()

    def fipc(self, year: int):
        """
        FIP constant
        Defined from https://library.fangraphs.com/pitching/fip/
        """
        year_data = self.pitching.loc[year]
        return (
            year_data.loc["ERA"]
            - (
                13 * year_data.loc["HR"]
                + 3 * (year_data.loc["BB"] + year_data.loc["HBP"])
                - 2 * year_data.loc["SO"]
            )
            / year_data.loc["IP"]
        )

    def __reafip_consts(self, year)-> pd.DataFrame:
        pass

    def reafip(self, player_id: int, year: int) -> float:
        """
        Run Expectancy Adjusted FIP

        this statistic makes an attempt to adjust fip so instead of using static constants FIP the value of home runs,
        walks/hit by pitch, and strikeouts are computed using the per-season run expectancy data.
        """
        player_data = self.mlbstats.get_player_stats(
            player_id,
            stats=["season", "seasonAdvanced"],
            groups=["pitching"],
            season=year,
        )
        basic_stats = player_data["pitching"]["season"].splits[0].stat
        consts = self.__reafip_consts(year)
        return (
            consts["hr"] * basic_stats.home_runs
            + consts["walks"] * (basic_stats.base_on_balls + basic_stats.hit_by_pitch)
            - consts["so"] * basic_stats.strikeouts
        ) / float(basic_stats.innings_pitched)

    def fip(self, player_id: int, year: int):
        """
        Fielding Independent Pitching
        Defined from https://library.fangraphs.com/pitching/fip/
        """
        player_data = self.mlbstats.get_player_stats(
            player_id,
            stats=["season", "seasonAdvanced"],
            groups=["pitching"],
            season=year,
        )
        basic_stats = player_data["pitching"]["season"].splits[0].stat
        return (
            13 * basic_stats.home_runs
            + 3 * (basic_stats.base_on_balls + basic_stats.hit_by_pitch)
            - 2 * basic_stats.strikeouts
        ) / float(basic_stats.innings_pitched) + self.fipc(year)

    def fip_team(self, team_id: int, year: int):
        team_stats = self.mlbstats.get_team_stats(
            team_id,
            stats=["season", "seasonAdvanced"],
            groups=["pitching"],
            **{"season": year},
        )
        pitching_stats = team_stats["pitching"]["season"].splits[0].stat
        return (
            (
                13 * pitching_stats.home_runs
                + 3 * (pitching_stats.base_on_balls + pitching_stats.hit_by_pitch)
                - 2 * pitching_stats.strikeouts
            )
            / float(pitching_stats.innings_pitched)
        ) + self.fipc(year)

    def __calc_fipr9_divisions(self, year):
        """
        fipr9 for all divisions for a given year
        """
        if year in self.fipr9_devisions.keys():
            return self.fipr9_devisions[year]
        teams = self.mlbstats.get_teams(sport_id=1)
        divisions_fipr9s = {}
        fip_adj = (
            self.pitching.loc[year].loc["R"] / self.pitching.loc[year].loc["IP"]
        ) * 9 - self.pitching.loc[year].loc["ERA"]
        for team in teams:
            fipr9 = self.fip_team(team.id, year) + fip_adj
            if team.division.id not in divisions_fipr9s:
                divisions_fipr9s[team.division.id] = [fipr9]
            else:
                divisions_fipr9s[team.division.id].append(fipr9)
        fipr9s = {}
        for key in divisions_fipr9s.keys():
            fipr9s[key] = np.array(divisions_fipr9s[key]).mean()
        self.fipr9_devisions[year] = fipr9s
        return fipr9s

    def calc_fipr9_league(self, year):
        if year in self.fipr9_league.keys():
            return self.fipr9_league[year]
        teams = self.mlbstats.get_teams(sport_id=1)
        team_fipr9s = np.empty(30)
        fip_adj = (
            self.pitching.loc[year].loc["R"] / self.pitching.loc[year].loc["IP"]
        ) * 9 - self.pitching.loc[year].loc["ERA"]
        for i, team in enumerate(teams):
            fipr9 = self.fip_team(team.id, year) + fip_adj
            team_fipr9s[i] = fipr9
        fipr9_year = np.mean(team_fipr9s)
        self.fipr9_league[year] = fipr9_year
        return fipr9_year

    def pitching_war(self, player_id, year: int = datetime.now().year):
        """
        https://library.fangraphs.com/war/calculating-war-pitchers/
        leverage index is not implemented yet.
        Instead of adjusting for league average, we adjust for division averages
        """
        player_data = self.mlbstats.get_player_stats(
            player_id,
            stats=["season", "seasonAdvanced"],
            groups=["pitching"],
            season=year,
        )
        team_id = player_data["pitching"]["season"].splits[0].team.id
        team = self.mlbstats.get_team(team_id)
        basic_stats = player_data["pitching"]["season"].splits[0].stat

        # league ra/9
        lg_ra9 = (
            self.pitching.loc[year].loc["R"] / self.pitching.loc[year].loc["IP"]
        ) * 9
        # fip scaled to ra/9
        fip_adjust = lg_ra9 - self.pitching.loc[year].loc["ERA"]
        fipr9 = self.fip(player_id, year) + fip_adjust
        # park factor adjusted fipr9
        pfipr9 = fipr9 / (
            self.park_factors.loc[2025]
            .loc[self.park_factors["Team"] == team.team_name]["FIP"]
            .item()
            / 100
        )

        # Runs Above Average Per 9 scaled to division
        ip = float(basic_stats.innings_pitched)

        if year > 2022:
            # for the 2023 when inter-division games were reduced from 76 to 13
            # and interleague were increased from 20 to 46
            raap9 = self.calc_fipr9_league(year) - pfipr9
            # Dynamic Runs Per Win
            d_rpw = (
                (
                     (
                        (18 - ip / basic_stats.games_pitched)
                        * self.calc_fipr9_league(year)
                    )
                    + ((ip / basic_stats.games_pitched) * pfipr9)
                )
                / 18
                + 2
            ) * 1.5
        else:
            raap9 = self.__calc_fipr9_divisions(year)[team.division.id] - pfipr9
            # Dynamic Runs Per Win
            d_rpw = (
                (
                    (
                        (18 - ip / basic_stats.games_pitched)
                        * self.__calc_fipr9_divisions(year)[team.division.id]
                    )
                    + ((ip / basic_stats.games_pitched) * pfipr9)
                )
                / 18
                + 2
            ) * 1.5
        # wins per game above average
        wpgaa = raap9 / d_rpw
        # replacement level
        rl = 0.03 * (
            1 - basic_stats.games_started / basic_stats.games_played
        ) + 0.12 * (basic_stats.games_started / basic_stats.games_played)
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
    league_avg_pitching = pd.read_sql(
        'SELECT * FROM league_avg_pitching WHERE "Season" = 2025;',
        con=engine,
        index_col="Season",
    )
    park_factors = pd.read_sql(
        'SELECT * FROM park_factors WHERE "Season" = 2025;',
        con=engine,
        index_col="Season",
    )
    mlb = mlbstatsapi.Mlb()
    pitcher_stats = PitchingCalculator(league_avg_pitching, park_factors, mlb)
    name = "Shohei Ohtani"
    player_id = mlb.get_people_id(name)[0]
    fip = pitcher_stats.fip(player_id, 2025)
    print(f"{name} fip {fip}")
    war = pitcher_stats.pitching_war(player_id, year=2025)
    print(f"{name} pitching war {war}")
