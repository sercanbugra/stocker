import re
import requests
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from bs4 import BeautifulSoup


# -------------------------------------------------------------------
# 1. Tarihsel veri (football-data.co.uk)
# -------------------------------------------------------------------

def season_code_from_year(year: int) -> str:
    """
    football-data.co.uk sezon kodu:
    2019-2020 -> '1920', 2020-2021 -> '2021', vb.
    """
    yy_start = year % 100
    yy_end = (year + 1) % 100
    return f"{yy_start:02d}{yy_end:02d}"


def get_fd_urls(start_year: int, end_year: int):
    """
    [start_year, end_year) aralığındaki Premier League sezonları için
    football-data.co.uk CSV URL'lerini üretir.
    """
    base = "https://www.football-data.co.uk/mmz4281"
    urls = []
    for year in range(start_year, end_year):
        code = season_code_from_year(year)
        urls.append((year, f"{base}/{code}/E0.csv"))
    return urls


def load_and_preprocess_data_from_csv(start_year: int, end_year: int):
    """
    football-data.co.uk üzerinden tarihsel EPL verisini indirir ve işler.

    Dönüş:
        full_data: DataFrame [Date, Season, HomeTeam, AwayTeam, FTHG, FTAG]
        all_teams: takımların unique listesi (np.array)
    """
    url_info = get_fd_urls(start_year, end_year)
    frames = []

    for season_year, url in url_info:
        try:
            print(f"Loading season {season_year}-{season_year+1} from {url} ...")
            df = pd.read_csv(url)

            required_cols = {"HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date"}
            if not required_cols.issubset(df.columns):
                print(f"  Warning: {url} missing required columns, skipping.")
                continue

            # Tarih formatı genelde dd/mm/yy
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            df["Season"] = str(season_year)

            frames.append(df[["Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]])

        except Exception as e:
            print(f"  Warning: Could not load {url}: {e}")

    if not frames:
        print("Error: No historical data could be loaded from football-data.co.uk")
        return None, None

    full_data = pd.concat(frames, ignore_index=True)
    full_data.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"], inplace=True)

    all_teams = pd.unique(full_data[["HomeTeam", "AwayTeam"]].values.ravel("K"))
    return full_data, all_teams


# -------------------------------------------------------------------
# 2. Model eğitimi
# -------------------------------------------------------------------

def train_models(data: pd.DataFrame, all_teams: np.ndarray):
    """
    İki RandomForest modeli eğitir:
      - model_home: ev sahibi golleri (FTHG)
      - model_away: deplasman golleri (FTAG)

    En güncel sezondaki maçlara 5x sample_weight verilir.
    """
    latest_season = data["Season"].max()
    print(f"Applying 5x weight to the most recent season: {latest_season}")

    data = data.copy()
    data["sample_weight"] = np.where(data["Season"] == latest_season, 5.0, 1.0)

    # Takım isimlerini encode et
    team_encoder = LabelEncoder().fit(all_teams)
    data["HomeTeam_encoded"] = team_encoder.transform(data["HomeTeam"])
    data["AwayTeam_encoded"] = team_encoder.transform(data["AwayTeam"])

    X = data[["HomeTeam_encoded", "AwayTeam_encoded"]]
    y_home = data["FTHG"]
    y_away = data["FTAG"]
    weights = data["sample_weight"]

    print("Training model for home goals...")
    model_home = RandomForestRegressor(
        n_estimators=100, random_state=42, min_samples_leaf=5
    )
    model_home.fit(X, y_home, sample_weight=weights)

    print("Training model for away goals...")
    model_away = RandomForestRegressor(
        n_estimators=100, random_state=42, min_samples_leaf=5
    )
    model_away.fit(X, y_away, sample_weight=weights)

    print("Models trained successfully.")
    return model_home, model_away, team_encoder


# -------------------------------------------------------------------
# 3. Yardımcı: takım isimlerini normalize et
# -------------------------------------------------------------------

def normalize_team_name(name: str) -> str:
    """
    Farklı sitelerdeki küçük isim farklarını düzeltmek için normalizasyon.
    Örnek:
      'Brighton & Hove Albion' / 'Brighton and Hove Albion' -> aynı
    """
    if not isinstance(name, str):
        return ""

    n = name.strip()
    # & -> and
    n = n.replace("&", "and")
    # Çoklu boşlukları tek boşluğa çevir
    n = re.sub(r"\s+", " ", n)
    # FC, A.F.C. gibi son ekleri kaldır (çok agresif olmak istemiyoruz)
    n = re.sub(r"\s+(FC|AFC|F\.C\.)$", "", n, flags=re.IGNORECASE)
    return n.lower()


# -------------------------------------------------------------------
# 4. Lig sıralaması (SkySports Premier League Table)
# -------------------------------------------------------------------

def fetch_league_table_and_strengths():
    """
    SkySports üzerinden güncel Premier League tablosunu çeker
    ve her takım için bir strength (güç) katsayısı üretir.

    Dönüş:
        table_df: DataFrame [Position, Team, TeamNorm]
        strength_map: dict {TeamNorm: strength}
    """
    url = "https://www.skysports.com/premier-league-table"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }
    print(f"\nFetching league table from {url} ...")
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching league table: {e}")
        return pd.DataFrame(columns=["Position", "Team", "TeamNorm"]), {}

    soup = BeautifulSoup(resp.text, "html.parser")

    # "Premier League Table" başlığından sonraki tabloyu bulmaya çalış
    header = None
    for tag in soup.find_all(["h1", "h2", "h3"]):
        if "Premier League Table" in tag.get_text():
            header = tag
            break

    if header:
        table = header.find_next("table")
    else:
        # fallback: ilk tablo
        table = soup.find("table")

    if not table:
        print("Could not find standings table in the page.")
        return pd.DataFrame(columns=["Position", "Team", "TeamNorm"]), {}

    body = table.find("tbody") or table
    rows = body.find_all("tr")

    data = []
    for tr in rows:
        cells = tr.find_all(["td", "th"])
        if len(cells) < 2:
            continue

        pos_text = cells[0].get_text(strip=True)
        team_text = cells[1].get_text(strip=True)

        if not pos_text.isdigit():
            continue

        pos = int(pos_text)
        team = team_text
        team_norm = normalize_team_name(team)

        data.append({"Position": pos, "Team": team, "TeamNorm": team_norm})

    table_df = pd.DataFrame(data).sort_values("Position").reset_index(drop=True)

    if table_df.empty:
        print("Warning: league table parsed as empty.")
        return table_df, {}

    print("Current league table (top -> bottom):")
    print(table_df[["Position", "Team"]])

    # Strength hesabı:
    # En üst sıraya ~1.3, en alt sıraya ~0.7 gibi lineer bir skala verelim.
    n = len(table_df)
    strengths = {}
    max_boost = 1.3
    min_boost = 0.7
    for _, row in table_df.iterrows():
        pos = row["Position"]
        norm = row["TeamNorm"]
        # 1. sıraya 1.3, 20. sıraya 0.7 arasında lineer
        rel = (n - pos) / max(1, n - 1)  # 0 (son) -> 1 (lider)
        strength = min_boost + (max_boost - min_boost) * rel
        strengths[norm] = strength

    return table_df, strengths


# -------------------------------------------------------------------
# 5. Yaklaşan fikstürler (ESPN fixtures)
# -------------------------------------------------------------------

def fetch_upcoming_fixtures_from_web(days_ahead: int = 7,
                                     known_teams_norm: set = None) -> pd.DataFrame:
    """
    ESPN Premier League fixtures sayfasından önümüzdeki 'days_ahead' gün içindeki
    maçların Date, Gameweek, HomeTeam, AwayTeam bilgilerini çeker.

    known_teams_norm: normalize edilmiş takım isimleri set'i
                      (lig tablosundan alınan isimlerle uyum için).
    """
    url = "https://www.espn.co.uk/football/fixtures/_/league/eng.1"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }

    print(f"\nFetching upcoming fixtures from {url} ...")
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching fixtures: {e}")
        return pd.DataFrame(columns=["Date", "Gameweek", "HomeTeam", "AwayTeam"])

    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text("\n")
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    today = datetime.today().date()
    max_date = today + timedelta(days=days_ahead)

    fixtures = []
    current_date = None
    pending_teams = []

    # Tarih satırı regex'i (Monday, December 8, 2025 şeklinde)
    date_pattern = re.compile(
        r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+"
        r"([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4})$"
    )

    def line_to_team(line: str):
        # Satır içindeki takım adını normalize ederek known_teams ile eşleştir
        candidate_norm = normalize_team_name(line)
        if not candidate_norm:
            return None

        if known_teams_norm is not None:
            # SkySports'taki takımlara mümkün olduğunca yaklaş
            if candidate_norm in known_teams_norm:
                return line  # orijinali döndürelim, sonra yeniden map ederiz
            else:
                return None
        return line

    for line in lines:
        # 1) Tarih başlığını yakala
        m = date_pattern.match(line)
        if m:
            weekday, month_name, day_str, year_str = m.groups()
            date_str = f"{day_str} {month_name} {year_str}"
            try:
                current_date = datetime.strptime(date_str, "%d %B %Y").date()
            except ValueError:
                current_date = None
            pending_teams = []
            continue

        # 2) Tarih aralığı içinde miyiz?
        if not current_date or not (today <= current_date <= max_date):
            continue

        # 3) Satır takım ismine benziyor mu?
        maybe_team = line_to_team(line)
        if maybe_team:
            pending_teams.append(maybe_team)

            # 2 takımı görünce bir maç kabul ediyoruz
            if len(pending_teams) >= 2:
                home_team = pending_teams[0]
                away_team = pending_teams[1]
                gw = current_date.isocalendar()[1]  # ISO week -> "Gameweek"

                fixtures.append(
                    {
                        "Date": current_date,
                        "Gameweek": gw,
                        "HomeTeam": home_team,
                        "AwayTeam": away_team,
                    }
                )
                pending_teams = []  # sıradaki maç için sıfırla

    fixtures_df = pd.DataFrame(fixtures)

    if fixtures_df.empty:
        print("No upcoming fixtures found in the next week.")
    else:
        print("Upcoming fixtures (next week) collected from web:")
        print(fixtures_df[["Date", "Gameweek", "HomeTeam", "AwayTeam"]])

    return fixtures_df


# -------------------------------------------------------------------
# 6. Tahmin: lig sıralamasını da ağırlık olarak kullan
# -------------------------------------------------------------------

def predict_with_standings(
    model_home,
    model_away,
    team_encoder: LabelEncoder,
    all_teams: np.ndarray,
    fixtures_df: pd.DataFrame,
    league_table_df: pd.DataFrame,
    strength_map: dict,
):
    """
    Fikstür DataFrame'i (Date, Gameweek, HomeTeam, AwayTeam) için
    lig tablosundaki strength katsayılarını da kullanarak skor tahmini yapar.
    """

    if fixtures_df.empty:
        print("No fixtures to predict.")
        return

    # Lig tablosundaki takımları normalize et ve canonical isim map'i oluştur
    norm_to_table_name = {}
    for _, row in league_table_df.iterrows():
        norm_to_table_name[row["TeamNorm"]] = row["Team"]

    # Fikstürü Canonical (SkySports) isimlerle yeniden yaz
    fixtures_encoded_rows = []
    for _, row in fixtures_df.iterrows():
        home_raw = row["HomeTeam"]
        away_raw = row["AwayTeam"]

        # ESPN vs Sky isim farklarını normalize ederek yaklaştır
        home_norm = normalize_team_name(home_raw)
        away_norm = normalize_team_name(away_raw)

        # Eğer SkySports'ta karşılığı varsa, onu kullan
        home_canonical = norm_to_table_name.get(home_norm, home_raw)
        away_canonical = norm_to_table_name.get(away_norm, away_raw)

        fixtures_encoded_rows.append(
            {
                "Date": row["Date"],
                "Gameweek": row["Gameweek"],
                "HomeTeam": home_canonical,
                "AwayTeam": away_canonical,
            }
        )

    fixtures_clean = pd.DataFrame(fixtures_encoded_rows)

    # >>> ÖNEMLİ FIX: Encoder'ı CANONICAL isimlere göre tekrar genişlet <<<
    fixture_team_names = pd.unique(
        fixtures_clean[["HomeTeam", "AwayTeam"]].values.ravel("K")
    )
    unknown_for_encoder = np.setdiff1d(fixture_team_names, team_encoder.classes_)
    if len(unknown_for_encoder) > 0:
        print(
            "Extending encoder with teams (post-canonical mapping): "
            f"{list(unknown_for_encoder)}"
        )
        updated_classes = np.concatenate([team_encoder.classes_, unknown_for_encoder])
        team_encoder.classes_ = updated_classes

    # Encode et
    home_encoded = team_encoder.transform(fixtures_clean["HomeTeam"])
    away_encoded = team_encoder.transform(fixtures_clean["AwayTeam"])

    X_pred = pd.DataFrame(
        {
            "HomeTeam_encoded": home_encoded,
            "AwayTeam_encoded": away_encoded,
        }
    )

    # Modelden ham (standings'siz) tahminler
    pred_home_raw = model_home.predict(X_pred)
    pred_away_raw = model_away.predict(X_pred)

    # Lig tablosu strength ağırlığı uygula
    print("\n--- Predictions with league-position weighting ---")
    for i, row in fixtures_clean.iterrows():
        match_date = row["Date"].strftime("%Y-%m-%d")
        gw = row["Gameweek"]
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]

        home_norm = normalize_team_name(home_team)
        away_norm = normalize_team_name(away_team)

        home_strength = strength_map.get(home_norm, 1.0)
        away_strength = strength_map.get(away_norm, 1.0)

        # Ortalama güç etrafında normalize et (toplam gol çok sapmasın)
        mean_strength = (home_strength + away_strength) / 2.0
        if mean_strength == 0:
            mean_strength = 1.0

        home_factor = home_strength / mean_strength
        away_factor = away_strength / mean_strength

        # Ham tahminleri güç faktörü ile çarp
        h_raw = pred_home_raw[i]
        a_raw = pred_away_raw[i]

        h_adj = h_raw * home_factor
        a_adj = a_raw * away_factor

        home_goals = int(round(h_adj))
        away_goals = int(round(a_adj))

        print(
            f"GW{gw} | {match_date} | "
            f"{home_team} vs {away_team}: "
            f"Predicted Score -> {home_goals} - {away_goals} "
            f"(raw: {h_raw:.2f}-{a_raw:.2f}, "
            f"strength: {home_strength:.2f}-{away_strength:.2f})"
        )


# -------------------------------------------------------------------
# 7. main
# -------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting the FPL score prediction process with league-table weighting...")

    current_year = datetime.now().year
    # 5 tam sezon için ~6 yıl geriye gidelim
    START_YEAR = current_year - 6
    END_YEAR = current_year  # non-inclusive

    # 1) Tarihsel veri
    historical_data, all_teams = load_and_preprocess_data_from_csv(
        START_YEAR, END_YEAR
    )

    if historical_data is None:
        print("Historical data load failed. Exiting.")
    else:
        # 2) Model eğitimleri
        model_home, model_away, team_encoder = train_models(
            historical_data, all_teams
        )

        # 3) Mevcut lig sıralaması + strength haritası
        league_table_df, strength_map = fetch_league_table_and_strengths()
        if league_table_df.empty or not strength_map:
            print("League table/strengths not available, skipping weighting.")
        else:
            known_teams_norm = set(league_table_df["TeamNorm"].tolist())

            # 4) Önümüzdeki hafta için fikstürler (web'den)
            fixtures_df = fetch_upcoming_fixtures_from_web(
                days_ahead=7,
                known_teams_norm=known_teams_norm,
            )

            # 5) Tahmin + lig sıralamasına göre ağırlıklandırma
            if not fixtures_df.empty:
                predict_with_standings(
                    model_home,
                    model_away,
                    team_encoder,
                    all_teams,
                    fixtures_df,
                    league_table_df,
                    strength_map,
                )

    print("\nProcess finished.")
