import pandas as pd
import sqlalchemy as sa
from sqlalchemy import Column, Integer, Float
from sqlalchemy.orm import Session
import os
from sqlalchemy.orm import declarative_base

Base = declarative_base()

BASE_LABELS = {
    0: "___",  # empty
    1: "1__",  # runner on 1st
    2: "_2_",  # runner on 2nd
    3: "12_",  # 1st and 2nd
    4: "__3",  # runner on 3rd
    5: "1_3",
    6: "_23",
    7: "123",  # loaded
}


def __cell_col(outs: int, base_idx: int):
    return f"re_{outs}out_{BASE_LABELS[base_idx]}"


class RunExpectancy(Base):
    __tablename__ = "run_expectancy"
    year = Column(Integer, primary_key=True)


for outs in range(3):
    for base_idx in range(8):
        setattr(
            RunExpectancy,
            __cell_col(outs, base_idx),
            Column(__cell_col(outs, base_idx), Float),
        )


def write_run_expectancy(engine: sa.engine.Engine, year, re_grid):
    values = {"year": int(year)}
    for outs in range(3):
        for base_idx in range(8):
            values[__cell_col(outs, base_idx)] = float(re_grid[outs, base_idx])

    with Session(engine) as s:
        s.add(RunExpectancy(**values))
        s.commit()


def sql_write_play_by_play(engine: sa.engine.Engine, table: str, pbp: pd.DataFrame):
    """
    Write the play by play data for 1 MLB season to a SQL table
    """
    pbp.to_sql(name=table, con=engine, index=False)


if __name__ == "__main__":
    import statsapi_adapter
    import mlbstatsapi

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
    play_by_play_2026_08_01 = statsapi_adapter.get_play_by_play_games(mlb, date="2026-08-01")
    sql_write_play_by_play(engine, "test-pbp", play_by_play_2026_08_01)
