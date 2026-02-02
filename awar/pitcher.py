import mlbstatsapi
import pandas as pd
from datetime import datetime
import statsapi
import numpy as np


class PitchingCalculator:
    """
    Pitcher stats calculator
    equations credit to Piper Slowinski at Fangraphs
    """

    def __init__(self, league_avg_pitching_data_path: str, park_factors_path: str):
        self.pitching = pd.read_csv(league_avg_pitching_data_path, index_col="Year")
        self.park_factor = pd.read_csv(park_factors_path, index_col="Team")
        self.division_fipr9s = {}
        self.mlbstats = mlbstatsapi.Mlb()
        self.fipr9 = dict()

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

    def __calc_divisions_fipr9(self, year):
        if year in self.fipr9.keys():
            return self.fipr9[year]
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
        self.fipr9[year]=fipr9s
        return fipr9s

    def pitching_war(self, player_id, year: int = datetime.now().year):
        """
        https://library.fangraphs.com/war/calculating-war-pitchers/
        leverage index is not implemented yet.
        Instead of adjusting for league average we adjust for division averages

        """
        player_data = self.mlbstats.get_player_stats(
            player_id,
            stats=["season", "seasonAdvanced"],
            groups=["pitching"],
            season=year,
        )
        team_id = player_data["pitching"]["season"].splits[0].team.id
        player_stats = statsapi.player_stat_data(
            player_id, "pitching", "season", 1, year
        )
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
        pfipr9 = fipr9 / (self.park_factor.loc[team.team_name].loc["FIP"] / 100)
        # Runs Above Average Per 9 scaled to division
        if year not in self.division_fipr9s:
            self.division_fipr9s[year] = self.__calc_divisions_fipr9(year)
        raap9 = self.division_fipr9s[year][team.division.id] - pfipr9
        # Dynamic Runs Per Win
        ip = float(basic_stats.innings_pitched)
        d_rpw = (
            (
                (
                    (18 - ip / basic_stats.games_pitched)
                    * self.division_fipr9s[year][team.division.id]
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
        # li = (1+gm_li)/2
        return war_p


if __name__ == "__main__":
    pitcher_stats = PitchingCalculator("./league_avg_pitching.csv", "./pf_2025.csv")
    # name = "Edwin Díaz"
    name = "Shohei Ohtani"
    mlb = mlbstatsapi.Mlb()
    player_id = mlb.get_people_id(name)[0]
    fip = pitcher_stats.fip(player_id, 2025)
    print(f"{name} fip {fip}")
    war = pitcher_stats.pitching_war(player_id, year=2025)
    print(f"{name} pitching war {war}")
