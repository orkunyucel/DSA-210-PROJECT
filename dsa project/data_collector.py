from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder, playbyplayv2
import pandas as pd
import numpy as np
import time
import datetime
import os
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Çıktılar için klasör oluşturma
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/data", exist_ok=True)

# NBA API ayarlarını daha güvenilir hale getirmek için patch
def patch_nba_api():
    from nba_api.stats.library import http
    http.TIMEOUT = 120  # 2 dakika timeout

patch_nba_api()

# Daha iyi bir oturum oluşturma ve yeniden deneme stratejisi
def create_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    # Tarayıcı gibi görünmek için User-Agent başlıkları ekleme
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
    ]
    session.headers.update({'User-Agent': random.choice(user_agents)})
    
    return session

def api_call_with_retry(func, max_retries=5, base_delay=1, max_delay=10):
    retries = 0
    delay = base_delay
    
    while retries < max_retries:
        try:
            # API çağrısından önce rastgele bekleme
            time.sleep(random.uniform(1, 2))
            return func()
        except Exception as e:
            print(f"API çağrısı başarısız: {e}. {delay:.2f} saniye sonra yeniden deneniyor...")
            time.sleep(2)  # Sabit 2 saniyelik bekleme
            retries += 1
            # Üssel artış
            delay = min(delay * 1.5, max_delay)
    
    raise Exception("Maksimum yeniden deneme sayısı aşıldı")

# Tüm NBA takımlarını al
nba_teams = teams.get_teams()
# Sadece aktif takımları filtrele
active_teams = [team for team in nba_teams if team['id'] < 1610616000]
# Takım ID'lerini kısaltmalara eşleme sözlüğü oluştur
team_abbr_dict = {team['id']: team['abbreviation'] for team in nba_teams}

# Belirli bir sezon için tüm maçları getir
def get_season_games(season):
    cache_path = f"outputs/data/all_games_{season}.csv"
    
    if os.path.exists(cache_path):
        print(f"Önbelleğe alınmış {season} sezonu maçları kullanılıyor")
        return pd.read_csv(cache_path)
    
    print(f"{season} sezonu için tüm maçlar alınıyor...")
    
    def fetch_games():
        game_finder = api_call_with_retry(
            lambda: leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable="Regular Season"
            )
        )
        return game_finder.get_data_frames()[0]
    
    all_games_df = fetch_games()
    
    # Çiftleri kaldır (her maç her takım için bir kez görünür)
    game_ids = all_games_df['GAME_ID'].unique()
    unique_games = all_games_df.drop_duplicates(subset=['GAME_ID'])
    
    print(f"{season} sezonu için {len(game_ids)} benzersiz maç bulundu")
    
    # Önbelleğe kaydet
    unique_games.to_csv(cache_path, index=False)
    
    # Bekle
    time.sleep(2)
    
    return unique_games

def get_play_by_play(game_id):
    cache_path = f"outputs/data/pbp_{game_id}.csv"
    
    if os.path.exists(cache_path):
        # Dosyanın boş veya bozuk olmadığından emin ol
        try:
            df = pd.read_csv(cache_path)
            if len(df) > 0:
                return df
            # Eğer boşsa, aşağıda yeniden alacağız
        except:
            # Okuma başarısız olursa, yeniden alacağız
            pass
    
    try:
        def fetch_pbp():
            try:
                # Özel başlıklarla oturum oluştur
                session = create_session()
                pbp = playbyplayv2.PlayByPlayV2(game_id=game_id, timeout=30, proxy=None)
                df = pbp.get_data_frames()[0]
                
                # Yanıtın beklenen sütunları olup olmadığını kontrol et
                expected_columns = ['GAME_ID', 'EVENTNUM', 'EVENTMSGTYPE', 'PERIOD', 'PCTIMESTRING']
                for col in expected_columns:
                    if col not in df.columns:
                        print(f"Uyarı: Yanıtta {col} sütunu eksik")
                        return pd.DataFrame()  # Gerekli sütunlar eksikse boş DataFrame döndür
                        
                return df
            except Exception as inner_e:
                print(f"İç API istisnası: {inner_e}")
                # Eski maçlar için, 'veri yok' durumunu düzgünce ele al
                if "resultSet" in str(inner_e):
                    print(f"{game_id} maçı için pozisyon-pozisyon verisi mevcut değil")
                    return pd.DataFrame()
                raise inner_e  # Dış yeniden deneme mantığı için yeniden fırlat
        
        pbp_df = api_call_with_retry(fetch_pbp)
        
        # Veri boş veya eksikse boş DataFrame döndür
        if pbp_df is None or len(pbp_df) == 0:
            print(f"{game_id} maçı için pozisyon-pozisyon verisi mevcut değil")
            # Yeniden denemeyi önlemek için boş bir yer tutucu dosya oluştur
            pd.DataFrame().to_csv(cache_path, index=False)
            return pd.DataFrame()
        
        # Önbelleğe kaydet
        pbp_df.to_csv(cache_path, index=False)
        
        # Bekle
        time.sleep(2)
        
        return pbp_df
    except Exception as e:
        print(f"{game_id} maçı için pozisyon-pozisyon verisi alınırken hata: {e}")
        # Yeniden denemeyi önlemek için boş bir yer tutucu dosya oluştur
        pd.DataFrame().to_csv(cache_path, index=False)
        return pd.DataFrame()  # Boş DataFrame döndür

# 6-0 seriler ve molalar için pozisyon-pozisyon verilerini tara
def scan_for_runs_and_timeouts(play_by_play_data):
    """
    6-0 veya daha iyi serilerden sonra alınan molaları belirlemek için pozisyon-pozisyon verilerini tara.
    
    Döndürülen:
    - Mola bilgileriyle sözlük listesi
    """
    # Veri boşsa boş liste döndür
    if play_by_play_data.empty:
        return []
        
    timeout_data = []
    
    # Maçtaki takımları al
    teams_in_game = []
    for column in ['PLAYER1_TEAM_ID', 'PLAYER2_TEAM_ID', 'PLAYER3_TEAM_ID']:
        if column not in play_by_play_data.columns:
            continue
            
        for value in play_by_play_data[column].dropna().unique():
            if pd.notnull(value) and value not in [0, '0']:
                try:
                    team_id = int(value)
                    if team_id not in teams_in_game and team_id != 0:
                        teams_in_game.append(team_id)
                except:
                    pass
    
    if len(teams_in_game) < 2:
        return []  # Yeterli takım bilgisi yok
    
    # Her takım için, 6-0 serisi yapıp ardından rakibin mola alıp almadığını kontrol et
    for team_id in teams_in_game:
        # Diğer takım rakiptir
        opponent_team_ids = [t for t in teams_in_game if t != team_id]
        if not opponent_team_ids:
            continue
        
        opponent_team_id = opponent_team_ids[0]
        
        # Serileri takip et
        current_run = {'team_id': None, 'points': 0, 'active': False, 'start_event': None}
        score_diff = 0
        home_score = 0
        away_score = 0
        
        # Her pozisyonu kronolojik olarak işle
        for i, play in play_by_play_data.iterrows():
            # Skor varsa güncelle
            if 'SCORE' in play_by_play_data.columns and pd.notnull(play['SCORE']):
                try:
                    score_parts = str(play['SCORE']).split(' - ')
                    if len(score_parts) == 2:
                        away_score, home_score = map(int, score_parts)
                        score_diff = home_score - away_score
                except:
                    pass  # Skor ayrıştırma başarısız olursa atla
            
            # Skor yapan takımın ID'sini al
            scoring_team_id = None
            if 'PLAYER1_TEAM_ID' in play_by_play_data.columns and pd.notnull(play['PLAYER1_TEAM_ID']):
                try:
                    scoring_team_id = int(play['PLAYER1_TEAM_ID'])
                except:
                    pass
            
            # Sayı atılan pozisyonları kontrol et
            if 'EVENTMSGTYPE' in play_by_play_data.columns and play['EVENTMSGTYPE'] == 1:  # Atılan şut
                if scoring_team_id is None:
                    continue
                    
                points_scored = 0
                
                # Atılan sayıyı belirle
                if ('HOMEDESCRIPTION' in play_by_play_data.columns and 'Free Throw' in str(play.get('HOMEDESCRIPTION', ''))) or \
                   ('VISITORDESCRIPTION' in play_by_play_data.columns and 'Free Throw' in str(play.get('VISITORDESCRIPTION', ''))):
                    points_scored = 1
                elif ('HOMEDESCRIPTION' in play_by_play_data.columns and '3PT' in str(play.get('HOMEDESCRIPTION', ''))) or \
                     ('VISITORDESCRIPTION' in play_by_play_data.columns and '3PT' in str(play.get('VISITORDESCRIPTION', ''))):
                    points_scored = 3
                else:
                    points_scored = 2
                    
                # Mevcut seriyi güncelle
                if scoring_team_id == team_id:  # Takip ettiğimiz takım sayı attı
                    if current_run['active'] and current_run['team_id'] == team_id:
                        current_run['points'] += points_scored
                    else:
                        # Yeni seri başlat
                        current_run = {
                            'team_id': team_id, 
                            'points': points_scored, 
                            'active': True,
                            'start_event': play['EVENTNUM']
                        }
                else:  # Rakip sayı attı
                    # Seriyi sıfırla
                    current_run = {'team_id': None, 'points': 0, 'active': False, 'start_event': None}
                
            # Molaları kontrol et
            elif 'EVENTMSGTYPE' in play_by_play_data.columns and play['EVENTMSGTYPE'] == 4:  # Mola
                # Eğer aktif 6-0 veya daha iyi bir seri varsa ve rakip mola aldıysa
                if (current_run['active'] and 
                    current_run['team_id'] == team_id and 
                    current_run['points'] >= 6):
                    
                    # Hangi takımın mola aldığını belirle
                    timeout_called_by = None
                    if 'PLAYER1_TEAM_ID' in play_by_play_data.columns and pd.notnull(play['PLAYER1_TEAM_ID']):
                        try:
                            timeout_called_by = int(play['PLAYER1_TEAM_ID'])
                        except:
                            pass
                    
                    if timeout_called_by is None:
                        # Açıklamadan çıkarım yap
                        if 'HOMEDESCRIPTION' in play_by_play_data.columns and pd.notnull(play['HOMEDESCRIPTION']) and 'TIMEOUT' in str(play['HOMEDESCRIPTION']):
                            # Hangi takımın ev sahibi olduğunu belirle
                            home_team_abbr = team_abbr_dict.get(team_id, '')
                            if home_team_abbr and 'HOMEDESCRIPTION' in play_by_play_data.columns and any(play_by_play_data['HOMEDESCRIPTION'].str.contains(home_team_abbr, na=False)):
                                timeout_called_by = team_id
                            else:
                                timeout_called_by = opponent_team_id
                        elif 'VISITORDESCRIPTION' in play_by_play_data.columns and pd.notnull(play['VISITORDESCRIPTION']) and 'TIMEOUT' in str(play['VISITORDESCRIPTION']):
                            # Hangi takımın deplasman olduğunu belirle
                            visit_team_abbr = team_abbr_dict.get(team_id, '')
                            if visit_team_abbr and 'VISITORDESCRIPTION' in play_by_play_data.columns and any(play_by_play_data['VISITORDESCRIPTION'].str.contains(visit_team_abbr, na=False)):
                                timeout_called_by = team_id
                            else:
                                timeout_called_by = opponent_team_id
                    
                    # Sadece rakibin takımımızın serisi sırasında aldığı molaları kaydet
                    if timeout_called_by == opponent_team_id:
                        # Periyot ve zaman bilgilerini al
                        period = play.get('PERIOD', 0)
                        pc_time = play.get('PCTIMESTRING', '')
                        wc_time = play.get('WCTIMESTRING', None) if 'WCTIMESTRING' in play_by_play_data.columns else None
                    
                        # Mola bilgilerini kaydet
                        timeout_data.append({
                            'game_id': play.get('GAME_ID', ''),
                            'event_num': play.get('EVENTNUM', 0),
                            'run_start_event': current_run['start_event'],
                            'period': period,
                            'pc_time': pc_time,
                            'wc_time': wc_time if pd.notnull(wc_time) else None,
                            'team_id': team_id,  # Seri yapan takım
                            'opponent_team_id': opponent_team_id,  # Mola alan takım
                            'run_points': current_run['points'],
                            'score_diff': score_diff,
                            'home_score': home_score,
                            'away_score': away_score
                        })
                        
                    # Kim mola aldıysa seriyi sıfırla
                    current_run = {'team_id': None, 'points': 0, 'active': False, 'start_event': None}
    
    return timeout_data

# Periyot bazında hücum verimliliğini hesapla (hipoteze göre düzenlendi)
def calculate_period_efficiency(play_by_play_data, team_id, period, event_num, direction='before'):
    """
    Belirli bir periyodun bir kısmında takım için hücum verimliliğini hesapla
    
    Parametreler:
    - play_by_play_data: Pozisyon-pozisyon verileriyle DataFrame
    - team_id: Verimlilik hesaplanacak takım ID'si
    - period: Maç periyodu 
    - event_num: Molanın olay numarası
    - direction: 'before' (periyot başından molaya) veya 'after' (moladan periyot sonuna)
    
    Döndürülen:
    - Hücum verimliliği (pozisyon başına sayı)
    - Ek istatistiklerle sözlük
    """
    # Veri boşsa, varsayılan değerleri döndür
    if play_by_play_data.empty or 'PERIOD' not in play_by_play_data.columns or 'EVENTNUM' not in play_by_play_data.columns:
        empty_stats = {
            'points': 0, 'possessions': 1, 'fgm': 0, 'fga': 0, 'fg_pct': 0,
            'fg3m': 0, 'fg3a': 0, 'fg3_pct': 0, 'ftm': 0, 'fta': 0, 'ft_pct': 0,
            'oreb': 0, 'turnovers': 0, 'true_shooting': 0
        }
        return 0, empty_stats
    
    # team_id'yi sayısal değere dönüştür
    try:
        team_id = int(team_id) if not isinstance(team_id, int) else team_id
    except:
        print(f"Uyarı: Geçersiz team_id {team_id}")
        return 0, {
            'points': 0, 'possessions': 1, 'fgm': 0, 'fga': 0, 'fg_pct': 0,
            'fg3m': 0, 'fg3a': 0, 'fg3_pct': 0, 'ftm': 0, 'fta': 0, 'ft_pct': 0,
            'oreb': 0, 'turnovers': 0, 'true_shooting': 0
        }
    
    # Yöne ve periyoda göre oyunları filtrele
    if direction == 'before':
        # Periyot başından molaya
        relevant_plays = play_by_play_data[
            (play_by_play_data['PERIOD'] == period) & 
            (play_by_play_data['EVENTNUM'] < event_num)
        ].sort_values('EVENTNUM')
    else:  # 'after'
        # Moladan periyot sonuna
        relevant_plays = play_by_play_data[
            (play_by_play_data['PERIOD'] == period) & 
            (play_by_play_data['EVENTNUM'] > event_num)
        ].sort_values('EVENTNUM')
    
    # İlgili pozisyon yoksa, varsayılan değerleri döndür
    if len(relevant_plays) == 0:
        empty_stats = {
            'points': 0, 'possessions': 1, 'fgm': 0, 'fga': 0, 'fg_pct': 0,
            'fg3m': 0, 'fg3a': 0, 'fg3_pct': 0, 'ftm': 0, 'fta': 0, 'ft_pct': 0,
            'oreb': 0, 'turnovers': 0, 'true_shooting': 0
        }
        return 0, empty_stats
    
    # Pozisyonları ve istatistikleri takip et
    possessions = 0
    points = 0
    fgm, fga = 0, 0
    fg3m, fg3a = 0, 0
    ftm, fta = 0, 0
    oreb = 0
    turnovers = 0
    current_possession_team = None
    
    # Pozisyonları işleyerek pozisyonları ve istatistikleri takip et
    for _, play in relevant_plays.iterrows():
        # İlgisiz olay türlerini veya eksik olay türünü atla
        if 'EVENTMSGTYPE' not in play or play['EVENTMSGTYPE'] not in [1, 2, 3, 4, 5, 6, 13]:
            continue
        
        # Bunun rakip takımın aksiyonu olup olmadığını kontrol et
        is_team_action = False
        if 'PLAYER1_TEAM_ID' in play and pd.notnull(play['PLAYER1_TEAM_ID']):
            try:
                is_team_action = int(play['PLAYER1_TEAM_ID']) == team_id
            except:
                pass
        
        # Pozisyon değişiklikleri
        if play['EVENTMSGTYPE'] in [1, 2, 3, 5]:  # Başarılı basket, kaçan şut, serbest atış, top kaybı
            # Pozisyonun değişip değişmediğini belirle
            if is_team_action and current_possession_team != team_id:
                possessions += 1
                current_possession_team = team_id
            
            # Takım için istatistikleri takip et
            if is_team_action:
                if play['EVENTMSGTYPE'] == 1:  # Başarılı şut
                    home_desc = str(play.get('HOMEDESCRIPTION', '')) if 'HOMEDESCRIPTION' in play else ''
                    visitor_desc = str(play.get('VISITORDESCRIPTION', '')) if 'VISITORDESCRIPTION' in play else ''
                    
                    if 'Free Throw' in home_desc or 'Free Throw' in visitor_desc:
                        points += 1
                        ftm += 1
                        fta += 1
                    elif '3PT' in home_desc or '3PT' in visitor_desc:
                        points += 3
                        fgm += 1
                        fga += 1
                        fg3m += 1
                        fg3a += 1
                    else:
                        points += 2
                        fgm += 1
                        fga += 1
                elif play['EVENTMSGTYPE'] == 2:  # Kaçan şut
                    home_desc = str(play.get('HOMEDESCRIPTION', '')) if 'HOMEDESCRIPTION' in play else ''
                    visitor_desc = str(play.get('VISITORDESCRIPTION', '')) if 'VISITORDESCRIPTION' in play else ''
                    
                    if '3PT' in home_desc or '3PT' in visitor_desc:
                        fg3a += 1
                        fga += 1
                    else:
                        fga += 1
                elif play['EVENTMSGTYPE'] == 3:  # Serbest atış
                    home_desc = str(play.get('HOMEDESCRIPTION', '')) if 'HOMEDESCRIPTION' in play else ''
                    visitor_desc = str(play.get('VISITORDESCRIPTION', '')) if 'VISITORDESCRIPTION' in play else ''
                    
                    if 'MISS' in home_desc or 'MISS' in visitor_desc:
                        fta += 1
                    else:
                        ftm += 1
                        fta += 1
                        points += 1
                elif play['EVENTMSGTYPE'] == 5:  # Top kaybı
                    turnovers += 1
            
            # Hücum ribaundlarını takip et (pozisyonu uzatır)
            if (play['EVENTMSGTYPE'] == 4):
                home_desc = str(play.get('HOMEDESCRIPTION', '')) if 'HOMEDESCRIPTION' in play else ''
                visitor_desc = str(play.get('VISITORDESCRIPTION', '')) if 'VISITORDESCRIPTION' in play else ''
                
                if ('REBOUND' in home_desc or 'REBOUND' in visitor_desc) and ('OFF' in home_desc or 'OFF' in visitor_desc):
                    if is_team_action:
                        oreb += 1
                        # Hücum ribaundu yeni bir pozisyon olarak sayılmaz
                        possessions = max(0, possessions - 1)
    
    # Verimlilik metriklerini hesapla
    efficiency = points / max(1, possessions)  # Sıfıra bölmeyi önle
    true_shooting = points / (2 * (fga + 0.44 * fta)) if (fga + 0.44 * fta) > 0 else 0
    
    # Verimliliği ve ek istatistikleri döndür
    stats = {
        'points': points,
        'possessions': possessions if possessions > 0 else 1,  # Sıfıra bölmeyi önle
        'fgm': fgm,
        'fga': fga,
        'fg_pct': fgm / max(1, fga),  # Sıfıra bölmeyi önle
        'fg3m': fg3m,
        'fg3a': fg3a,
        'fg3_pct': fg3m / max(1, fg3a),  # Sıfıra bölmeyi önle
        'ftm': ftm,
        'fta': fta,
        'ft_pct': ftm / max(1, fta),  # Sıfıra bölmeyi önle
        'oreb': oreb,
        'turnovers': turnovers,
        'true_shooting': true_shooting,
    }
    
    return efficiency, stats

# Serili maçları bulma ve molaları analiz etme ana fonksiyonu
def find_and_analyze_timeouts(seasons, max_games_per_season=None, percent_of_games=None):
    all_timeout_results = []
    
    # Kaldığımız yerden devam etme imkanı
    existing_results_path = 'outputs/timeout_analysis_results_partial.csv'
    if os.path.exists(existing_results_path):
        print(f"Mevcut sonuçlar bulundu, kaldığımız yerden devam ediliyor")
        existing_results = pd.read_csv(existing_results_path)
        all_timeout_results = existing_results.to_dict('records')
        
        # İşlediğimiz maçları takip et
        processed_games = set()
        for result in all_timeout_results:
            processed_games.add(result['game_id'])
        
        print(f"Zaten {len(processed_games)} maç işlendi, {len(all_timeout_results)} mola analiz edildi")
    else:
        processed_games = set()
    
    for season in seasons:
        print(f"\n{season} sezonu analiz ediliyor")
        
        # Skor-sıfır serilerinde sınırlı play-by-play verileri olan eski sezonları atla
        if season == "1996-97":
            print(f"Not: {season} sezonu için sınırlı pozisyon-pozisyon verisi mevcut olabilir")
        
        # Sezon için tüm maçları al
        try:
            season_games_df = get_season_games(season)
            
            # Sezonun maçlarının bir yüzdesini al
            if percent_of_games is not None:
               num_games = int(len(season_games_df) * percent_of_games)
               print(f"Toplam {len(season_games_df)} maçtan {num_games} maç analiz ediliyor (%{percent_of_games*100:.0f})")
               season_games_df = season_games_df.sample(num_games, random_state=42)
            # Veya belirli bir sayı al
            elif max_games_per_season and len(season_games_df) > max_games_per_season:
                print(f"Toplam {len(season_games_df)} maçtan rastgele {max_games_per_season} maç ile sınırlandırılıyor")
                season_games_df = season_games_df.sample(max_games_per_season, random_state=42)
            
            # Serileri ve molaları bulmak için her maçı işle
            for i, (_, game) in enumerate(season_games_df.iterrows()):
                game_id = game['GAME_ID']
                
                # Bu maçı daha önce işlediysek atla
                if game_id in processed_games:
                    print(f"Önceden işlenen maç atlanıyor {i+1}/{len(season_games_df)}: {game_id}")
                    continue
                    
                print(f"Maç taranıyor {i+1}/{len(season_games_df)}: {game_id}")
                
                try:
                    # Pozisyon-pozisyon verisini al
                    pbp_df = get_play_by_play(game_id)
                    
                    if pbp_df.empty:
                        print(f"Pozisyon-pozisyon verisi eksik olduğu için {game_id} maçı atlanıyor")
                        processed_games.add(game_id)
                        continue
                    
                    # Serileri ve molaları bul
                    timeouts = scan_for_runs_and_timeouts(pbp_df)
                    
                    if timeouts:
                        print(f"{game_id} maçında {len(timeouts)} ilgili mola bulundu")
                        
                        # Her periyottaki mola olaylarını takip et
                        timeout_events_by_period = {}
                        
                        # Her molayı analiz et
                        for timeout in timeouts:
                            try:
                                period = timeout['period']
                                event_num = timeout['event_num']
                                
                                # Her periyottaki molaları takip et
                                if period not in timeout_events_by_period:
                                    timeout_events_by_period[period] = []
                                timeout_events_by_period[period].append((event_num, timeout))
                                
                                # Rakip takım (seri yapan takım) için periyot bazlı verimliliği hesapla
                                pre_timeout_oe, pre_stats = calculate_period_efficiency(
                                    pbp_df, timeout['team_id'], period, event_num, 'before'
                                )
                                
                                post_timeout_oe, post_stats = calculate_period_efficiency(
                                    pbp_df, timeout['team_id'], period, event_num, 'after'
                                )
                                
                                # Periyot bilgisi
                                if period <= 4:
                                    quarter = f"Q{period}"
                                else:
                                    quarter = f"OT{period-4}"
                                
                                # Molanın etkili olup olmadığını belirle
                                efficiency_change = post_timeout_oe - pre_timeout_oe
                                effective = efficiency_change < 0  # Mola sonrası verimlilik azalırsa etkili
                                
                                # Serinin sona erme durumunu ekle
                                run_terminated = post_stats['points'] < pre_stats['points']
                                
                                # Takım kısaltmalarını al
                                team_abbr = team_abbr_dict.get(timeout['team_id'], "UNK")
                                opponent_abbr = team_abbr_dict.get(timeout['opponent_team_id'], "UNK")
                                
                                # Sonucu kaydet
                                all_timeout_results.append({
                                    'game_id': game_id,
                                    'team_id': timeout['team_id'],  # Seri yapan takım
                                    'team_abbr': team_abbr,
                                    'opponent_team_id': timeout['opponent_team_id'],  # Mola alan takım
                                    'opponent_abbr': opponent_abbr,
                                    'quarter': quarter,
                                    'timeout_time': timeout['pc_time'],
                                    'run_points': timeout['run_points'],
                                    'score_diff': timeout['score_diff'],
                                    'pre_timeout_oe': pre_timeout_oe,
                                    'post_timeout_oe': post_timeout_oe,
                                    'efficiency_change': efficiency_change,
                                    'effective': effective,
                                    'run_terminated': run_terminated,
                                    'pre_timeout_points': pre_stats['points'],
                                    'post_timeout_points': post_stats['points'],
                                    'pre_timeout_possessions': pre_stats['possessions'],
                                    'post_timeout_possessions': post_stats['possessions'],
                                    'pre_timeout_fg_pct': pre_stats['fg_pct'],
                                    'post_timeout_fg_pct': post_stats['fg_pct'],
                                    'pre_timeout_fg3_pct': pre_stats['fg3_pct'],
                                    'post_timeout_fg3_pct': post_stats['fg3_pct'],
                                    'pre_timeout_ts': pre_stats['true_shooting'],
                                    'post_timeout_ts': post_stats['true_shooting'],
                                    'pre_timeout_turnovers': pre_stats['turnovers'],
                                    'post_timeout_turnovers': post_stats['turnovers'],
                                    'season': season,
                                    'multiple_timeouts_in_period': len(timeout_events_by_period.get(period, [])) > 1
                                })
                                
                                # Ara sonuçları kaydet (her 10 moladan sonra)
                                if len(all_timeout_results) % 10 == 0:
                                    temp_df = pd.DataFrame(all_timeout_results)
                                    temp_df.to_csv('outputs/timeout_analysis_results_partial.csv', index=False)
                                    print(f"Ara sonuç kaydedildi: {len(all_timeout_results)} mola analiz edildi")
                                
                            except Exception as e:
                                print(f"{game_id} maçında mola analizi sırasında hata: {e}")
                    
                    # Maçı işlenmiş olarak işaretle
                    processed_games.add(game_id)
                    
                    # Maçlar arasında bekle
                    time.sleep(2)
                
                except Exception as e:
                    print(f"{game_id} maçı işlenirken hata: {e}")
                    time.sleep(2)  # Hata sonrası bekle
            
        except Exception as e:
            print(f"{season} sezonu için maçlar alınırken hata: {e}")
            time.sleep(5)  # Sezon hatası sonrası daha uzun bekle

    # Tüm sonuçları birleştir
    if all_timeout_results:
        combined_df = pd.DataFrame(all_timeout_results)
        
        # Ham sonuçları CSV'ye kaydet
        combined_df.to_csv('outputs/timeout_analysis_results.csv', index=False)
        print(f"Veri toplama tamamlandı! {len(all_timeout_results)} mola analiz edildi.")
        
        return combined_df
    else:
        print("Hiç mola bulunamadı.")
        return pd.DataFrame()

# Ana kısım
if __name__ == "__main__":
# Analiz edilecek sezonları tanımla - 5 sezon, farklı dönemlerden
    seasons_to_analyze = [
        # "1996-97",  # 90'lar - play-by-play verisi sınırlı
        "1999-00",  # 90'ların sonu/2000'lerin başı (96-97'den daha iyi veri)
        "2004-05",  # 2000'lerin başı
        "2010-11",  # 2010'ların başı  
        "2016-17",  # 2010'ların ortası
        "2022-23"   # Güncel dönem
    ]

# Her sezon için maçların %35'ini analiz et
results_df = find_and_analyze_timeouts(seasons_to_analyze, percent_of_games=0.35)