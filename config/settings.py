DB_DSN = "postgresql://postgres:123456@localhost:5432/soccer_db"

DATA_ROOT = r"E:\College\Final Project\open-data-master\data"

# (competition_id, season_id) pairs that are in scope.
# La Liga: competition_id=11, seasons 2015/16-2020/21
# UCL 2018/19: competition_id=16, season_id=4
# World Cup 2018: competition_id=43, season_id=3
# World Cup 2022: competition_id=43, season_id=106
#
# StatsBomb season IDs -- confirm against your local competitions.json
# if any IDs differ, update here.
COMPETITIONS = {
    (11, 26),   # La Liga 2015/16
    (11, 27),   # La Liga 2016/17
    (11, 4),    # La Liga 2017/18
    (11, 1),    # La Liga 2018/19
    (11, 42),   # La Liga 2019/20
    (11, 90),   # La Liga 2020/21
    (16, 4),    # UEFA Champions League 2018/19
    (43, 3),    # FIFA World Cup 2018
    (43, 106),  # FIFA World Cup 2022
}

# Open-Meteo base URL for historical weather data
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# Path to the Transfermarkt injuries CSV
# Download from: https://www.kaggle.com/datasets/irrazional/transfermarkt-injuries
TRANSFERMARKT_CSV = r"E:\College\Final Project\transfermarkt.csv"