"""
Author: Louis Adamian

The MLB Stats API can only be used for personal non-commercial use.
"""
import statsapi

def get_player_id(full_name:str, season:int)->int:
    return  next(x['id'] for x in statsapi.get('sports_players',{'season':season,'gameType':'W'})['people'] if x['fullName']==full_name)


def play_by_play_data():
