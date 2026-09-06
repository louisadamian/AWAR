from pandas import DataFrame


class Offensive:
    def __init__(self):
        pass


def __woba(
    stats,
    c_woba,
):
    """
    wOBA: weighted on base average
    """
    simple_stats = stats["hitting"]["season"].splits[0].stat
    ubb = simple_stats.base_on_balls - simple_stats.intentional_walks
    singles = (
        simple_stats.hits - simple_stats.doubles - simple_stats.triples - simple_stats.home_runs
    )

    return (
        c_woba["ubb"] * ubb
        + c_woba["hbp"] * simple_stats.hit_by_pitch
        + c_woba["1b"] * singles
        + c_woba["2b"] * simple_stats.doubles
        + c_woba["3b"] * simple_stats.triples
        + c_woba["hr"] * simple_stats.home_runs
    ) / (
        simple_stats.at_bats
        + simple_stats.base_on_balls
        - simple_stats.intentional_walks
        + simple_stats.sac_flies
        - simple_stats.hit_by_pitch
    )


def __woba_weights(pbp: DataFrame):
    """
    Calculate wOBA weights for a given set of play by play data
    """
    events = ["walk", "hit_by_pitch", "single", "double", "triple", "home_run"]
    exclude = {"intent_walk", "sac_bunt", "sac_bunt_double_play", "catcher_interf"}
    positive = {
        "walk",
        "hit_by_pitch",
        "single",
        "double",
        "triple",
        "home_run",
        "field_error",
        "fielders_choice",
    }
    in_denominator = ~pbp["result"].is_in(exclude)
