import pandas as pd
import sqlalchemy as sa
import os


def sql_write_play_by_play(engine: sa.engine.Engine, table: str, pbp: pd.DataFrame):
    """
    Write the play by play data for 1 MLB season to a SQL table
    """
    pbp.to_sql(name=table, con=engine, index=False)

if __name__ == '__main__':
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