import numpy as np
import pandas as pd
import os
import utils

# Column indices of data in int data matrix
INNING_NUM_INDEX = 0
BTEAM_INDEX = 1
OUTS_INDEX = 2
VIS_SCORE = 3
HOME_SCORE = 4
BAT_DEST = 16
FIRST_DEST = 17
SECOND_DEST = 18
THIRD_DEST = 19
# Column indices of data in str data matrix
ON_FIRST = 3
ON_SECOND = 4
ON_THIRD = 5
GAME_END = 33

BASE_DATA_PATH = os.path.join(os.getcwd(), 'Data', 'Output')


def run_ex(pbp: pd.DataFrame, year:int):
    """
    compute the run expectancy matrix from a play by play DataFrame
    """
    #drop any half innings that are not completed
    last_play_outs = pbp.groupby("half")["outs"].transform("max")
    complete_half = last_play_outs.eq(3)
    # Drop any half inning where a walk-off could occur (bottom half of the last scheduled inning or bottom extra innings)
    walk_off_innings = (pbp["side"].eq("bottom") & (pbp["inning"] >= 9))
    if year==2020 or year==2021:
        # walk off innings for 7 inning games in 2020 and 2021
        walk_off_7_innings = (pbp["side"].eq("bottom") & (pbp["inning"] >= 7) & pbp["7_innings"] == True)
        keep = ~(walk_off_innings & walk_off_7_innings & pbp["inning"] >9) & complete_half
    elif year>=2020:
        #drop innings with ghost runner https://www.mlb.com/glossary/rules/designated-runner
        keep = ~(walk_off_innings & pbp["inning"] >9) & complete_half
    else:
        keep = ~walk_off_innings & complete_half
    pbp = pbp[keep].copy()
    # calculate runs for each PA
    pbp["bat_score"] = np.where(
        pbp["side"].eq("top"), pbp["away_score"], pbp["home_score"]
    )
    prev = pbp.groupby(["game_id", "side"])["bat_score"].shift(1).fillna(0)
    pbp["runs"] = (pbp["bat_score"] - prev).astype(int)
    pbp["half"] = pbp.groupby(["game_id", "inning", "side"], sort=False).ngroup()
    # get on base as a bool
    bases = ["runner_on_1b", "runner_on_2b", "runner_on_3b"]
    on_base_bool = pbp[bases].notna()
    # group the data by half innings
    halves_groups = pbp.groupby("half", sort=False)
    # get the base states before each play
    pre_bases = on_base_bool.groupby(pbp["half"], sort=False).shift(1).fillna(False)
    pre_bases.columns = ["pre_1b", "pre_2b", "pre_3b"]
    pbp["pre_outs"] = halves_groups["outs"].shift(1).fillna(0).astype(int)
    pbp = pd.concat([pbp, pre_bases], axis=1)
    # calculate RE24 for each play
    pbp["re24"] = (
        halves_groups["runs"].transform("sum")
        - halves_groups["runs"].cumsum()
        + pbp["runs"]
    )
    # get the array index for the base state
    pbp["base_idx"] = (
        pbp["pre_1b"].astype(int)
        + 2 * pbp["pre_2b"].astype(int)
        + 4 * pbp["pre_3b"].astype(int)
    )
    # calculate the number of runs for each RE24 cell
    runs = np.zeros((3, 8), dtype=int)
    np.add.at(
        runs,
        (pbp["pre_outs"].to_numpy(), pbp["base_idx"].to_numpy()),
        pbp["re24"].to_numpy(),
    )
    # calculate the number of occurrences of each RE24 cell
    counts = np.zeros((3, 8), dtype=int)
    np.add.at(counts, (pbp["pre_outs"].to_numpy(), pbp["base_idx"].to_numpy()), 1)
    return np.divide(runs, counts)


# TODO these methods should be friendly to splitting leagues by home park. Also intend to remove all non-Ohtani pitcher
#   PAs which should be possible in preprocessing
def build_run_ex_matrix(outs_scores: np.ndarray):
    """Builds the run expectancy matrix dropping all potential walk-off half-innings"""
    # Setup and data manipulation
    base_matrix = np.zeros((3, 8))  # Create empty matrix, 3 possible out states, 8 base occupancy states
    situation_instances = np.zeros(base_matrix.shape)  # Tracks number of times each state combination in the matrix occurs
    outs_scores = utils.play_score(outs_scores)  # Gets number of runners scoring on each play

    # Splits data set into half innings to enable determining number of runs scored after each event in given half
    split_indices = np.where(np.logical_or(outs_scores[:, 1][:-1] != outs_scores[:, 1][1:], outs_scores[:, 0][:-1] != outs_scores[:, 0][1:]))[0] + 1
    innings = np.split(outs_scores, split_indices)

    # Iterate through half innings to determine run scoring after given events
    for inning in innings:
        # Skips half innings in which there is potential for a walk-off
        if (np.isin(1, inning[:, 11]) or inning[0, 0] >= 9) and (int(inning[0, 1]) == 1):
            continue
        matrix, instances = run_ex_inning(inning)  # Gets matrix for given half inning

        # Adds matrix and instances from half inning to season totals
        base_matrix = np.add(base_matrix, matrix)
        situation_instances = np.add(situation_instances, instances)
    return np.divide(base_matrix, situation_instances)


def build_run_ex_matrix_common(outs_scores: np.ndarray):
    """Builds the run expectancy matrix only dropping walk-offs."""
    base_matrix = np.zeros((3, 8))
    situation_instances = np.zeros(base_matrix.shape)
    score_on_play = np.roll(outs_scores[:, 3], -1) - outs_scores[:, 3]
    score_on_play[np.where(score_on_play < 0)] = 0
    outs_scores[:, 3] = score_on_play
    outs_scores = utils.play_score(outs_scores)
    split_indices = np.where(np.logical_or(outs_scores[:, 1][:-1] != outs_scores[:, 1][1:],
                                           outs_scores[:, 0][:-1] != outs_scores[:, 0][1:]))[0] + 1
    innings = np.split(outs_scores, split_indices)
    scores = get_game_scores()
    game_index = 0
    for inning in innings:
        if np.isin(1, inning[:, 11]):
            if scores[game_index] != inning[inning.shape[0]-1, 3]:
                game_index += 1
                continue
            game_index +=1
        matrix, instances = run_ex_inning(inning)
        base_matrix = np.add(base_matrix, matrix)
        situation_instances = np.add(situation_instances, instances)
        print("inning done")
    return np.divide(base_matrix, situation_instances)


def run_ex_inning(inning: np.ndarray):
    """Determines run expectancy contributions for a given inning"""
    base_matrix = np.zeros((3, 8))  # Same as outside base matrix
    instances = np.zeros(base_matrix.shape)  # Same as outside
    # Iterates through each event. Sequence is essential here so I haven't figured out a way to get rid of this loop.
    for event in inning:
        outs = int(event[2])  # Number of outs at play start
        values = (int(event[8]), int(event[9]), int(event[10]))  # Yes/No for 1st-3rd occupied
        occupied_bases = utils.S_3_element(values)  # Returns index in base matrix corresponding to base occupancy state
        instances[outs, occupied_bases] += 1  # Increments instances of base, outs state
        # Event[3] or event[4] depending on scoring style
        """The runs scored on this play are credited to each instance of each base occupancy state, thus they're
        included multiple times if a state has occurred multiple times already"""
        base_matrix += event[4] * instances
    return base_matrix, instances


def stitch_data(str_data: np.ndarray, int_data: np.ndarray):
    """returns an array with inning, batting team, outs, total score, runners on first, second, third, game end flag
    also returns an indexed array of gameIDs for comparison"""
    runners_on = str_data[:, [ON_FIRST, ON_SECOND, ON_THIRD, GAME_END]]
    runners_on_float = utils.runners_on_binary(runners_on)
    int_data[:, VIS_SCORE] += int_data[:, HOME_SCORE]
    outs_scores = int_data[:, [INNING_NUM_INDEX, BTEAM_INDEX, OUTS_INDEX, VIS_SCORE, BAT_DEST, FIRST_DEST, SECOND_DEST, THIRD_DEST]]
    return np.concatenate((outs_scores, runners_on_float), axis=1)


def get_game_scores():
    """Helper method for inferior method of matrix calculations"""
    games = np.genfromtxt(os.path.join(BASE_DATA_PATH, '2019GAME.csv'), delimiter=',', dtype=int)
    scores = games[:, 1] + games[:, 2]
    return scores


if __name__ == '__main__':
    pass
