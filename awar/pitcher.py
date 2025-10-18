import pandas as pd
from datetime import datetime
import statsapi

def fipc(year:int, path="./league_avg_pitching.csv"):
    """
    Defined from https://library.fangraphs.com/pitching/fip/
    """
    df = pd.read_csv(path,index_col="Year")
    year_data = df.loc[year]
    return year_data.loc['ERA']-(13*year_data.loc["HR"]+3*(year_data.loc["BB"]+year_data.loc["HBP"])-2*year_data.loc["SO"])/year_data.loc["IP"]


def fip(player_id:int, year:int):
    """
    Defined from https://library.fangraphs.com/pitching/fip/
    """
    player = statsapi.player_stat_data(player_id,'pitching', 'season',1,year)
    pitch_stats = player['stats'][0]['stats']
    return (13*pitch_stats["homeRuns"]+3*(pitch_stats["baseOnBalls"]+pitch_stats["hitByPitch"])-2*pitch_stats["strikeOuts"])/float(pitch_stats["inningsPitched"])+fipc(2025)

def pitcher_war(player_name:str, year:int=datetime.now().year, path="./league_avg_pitching.csv"):
    lgra9 =
    print()