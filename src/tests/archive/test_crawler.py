"""
🧪 Tests Crawler Sentinel2
Tests pour le module crawler
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import feedparser
import numpy as np
import pandas as pd
import pytest
import requests

from src.constants import CONSTANTS
from src.data.crawler import DataCrawler, NewsCrawler, PriceCrawler


@pytest.mark.crawler
class TestPriceCrawler:
    """Tests pour le crawler de prix"""

    def test_price_crawler_initialization(self):
        """Test l'initialisation du crawler de prix"""
        crawler = PriceCrawler()

        assert crawler.session is not None
        assert "User-Agent" in crawler.session.headers
        assert "Mozilla" in crawler.session.headers["User-Agent"]

    @patch("src.data.crawler.yf.Ticker")
    def test_get_yahoo_prices_success(self, mock_ticker_class):
        """Test la récupération réussie des prix Yahoo"""
        # Mock des données Yahoo Finance
        mock_data = pd.DataFrame(
            {
                "Open": [100, 101, 99],
                "High": [102, 103, 101],
                "Low": [99, 100, 98],
                "Close": [101, 102, 100],
                "Volume": [1000, 1100, 900],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="1min"),
        )

        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_data
        mock_ticker_class.return_value = mock_ticker

        crawler = PriceCrawler()
        result = crawler.get_yahoo_prices("SPY")

        assert not result.empty
        assert len(result) == 3
        assert "ts_utc" in result.columns
        assert "ticker" in result.columns
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns
        assert result["ticker"].iloc[0] == "SPY"

    @patch("src.data.crawler.yf.Ticker")
    def test_get_yahoo_prices_empty(self, mock_ticker_class):
        """Test la récupération avec données vides"""
        mock_ticker = Mock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker

        crawler = PriceCrawler()
        result = crawler.get_yahoo_prices("INVALID")

        assert result.empty

    @patch("src.data.crawler.yf.Ticker")
    def test_get_yahoo_prices_missing_columns(self, mock_ticker_class):
        """Test la récupération avec colonnes manquantes"""
        # Mock des données avec colonnes manquantes
        mock_data = pd.DataFrame(
            {
                "Open": [100, 101, 99],
                "High": [102, 103, 101],
                # Missing Low, Close, Volume
            },
            index=pd.date_range("2024-01-01", periods=3, freq="1min"),
        )

        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_data
        mock_ticker_class.return_value = mock_ticker

        crawler = PriceCrawler()
        result = crawler.get_yahoo_prices("SPY")

        assert result.empty

    @patch("src.data.crawler.yf.Ticker")
    def test_get_yahoo_prices_exception(self, mock_ticker_class):
        """Test la gestion d'exception lors de la récupération"""
        mock_ticker_class.side_effect = Exception("API Error")

        crawler = PriceCrawler()
        result = crawler.get_yahoo_prices("SPY")

        assert result.empty

    @patch("src.data.crawler.requests.Session.get")
    def test_get_polygon_prices_success(self, mock_get):
        """Test la récupération réussie des prix Polygon"""
        # Mock de la réponse API
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "OK",
            "results": [
                {
                    "t": 1704067200000,  # timestamp
                    "o": 100.0,  # open
                    "h": 102.0,  # high
                    "l": 99.0,  # low
                    "c": 101.0,  # close
                    "v": 1000,  # volume
                }
            ],
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.dict("os.environ", {"POLYGON_API_KEY": "test_key"}):
            crawler = PriceCrawler()
            result = crawler.get_polygon_prices("SPY")

        assert not result.empty
        assert len(result) == 1
        assert "ts_utc" in result.columns
        assert "ticker" in result.columns
        assert result["ticker"].iloc[0] == "SPY"
        assert result["open"].iloc[0] == 100.0

    @patch("src.data.crawler.requests.Session.get")
    def test_get_polygon_prices_no_api_key(self, mock_get):
        """Test la récupération sans clé API"""
        crawler = PriceCrawler()
        result = crawler.get_polygon_prices("SPY")

        assert result.empty
        mock_get.assert_not_called()

    @patch("src.data.crawler.requests.Session.get")
    def test_get_polygon_prices_api_error(self, mock_get):
        """Test la gestion d'erreur API Polygon"""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "ERROR", "message": "Invalid API key"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.dict("os.environ", {"POLYGON_API_KEY": "invalid_key"}):
            crawler = PriceCrawler()
            result = crawler.get_polygon_prices("SPY")

        assert result.empty


@pytest.mark.crawler
class TestNewsCrawler:
    """Tests pour le crawler de news"""

    def test_news_crawler_initialization(self):
        """Test l'initialisation du crawler de news"""
        crawler = NewsCrawler()

        assert crawler.session is not None
        assert "User-Agent" in crawler.session.headers

    @patch("src.data.crawler.feedparser.parse")
    def test_get_rss_news_success(self, mock_parse):
        """Test la récupération réussie des news RSS"""
        # Mock du feed RSS
        mock_feed = Mock()
        mock_feed.bozo = False
        mock_feed.entries = [
            Mock(
                title="NVIDIA stock rises",
                summary="Strong earnings report",
                link="https://example.com/news1",
                published="2024-01-01T10:00:00Z",
            ),
            Mock(
                title="Market analysis",
                summary="S&P 500 shows growth",
                link="https://example.com/news2",
                published="2024-01-01T11:00:00Z",
            ),
        ]
        mock_parse.return_value = mock_feed

        crawler = NewsCrawler()
        result = crawler.get_rss_news(["https://example.com/feed.rss"])

        assert len(result) == 2
        assert result[0]["title"] == "NVIDIA stock rises"
        assert result[0]["ticker"] == "NVDA"  # Détecté par les mots-clés
        assert result[1]["title"] == "Market analysis"
        assert result[1]["ticker"] == "SPY"  # Détecté par S&P 500

    @patch("src.data.crawler.feedparser.parse")
    def test_get_rss_news_malformed_feed(self, mock_parse):
        """Test la gestion d'un feed RSS malformé"""
        mock_feed = Mock()
        mock_feed.bozo = True
        mock_parse.return_value = mock_feed

        crawler = NewsCrawler()
        result = crawler.get_rss_news(["https://example.com/bad_feed.rss"])

        assert result == []

    @patch("src.data.crawler.feedparser.parse")
    def test_get_rss_news_exception(self, mock_parse):
        """Test la gestion d'exception lors de la récupération RSS"""
        mock_parse.side_effect = Exception("Network error")

        crawler = NewsCrawler()
        result = crawler.get_rss_news(["https://example.com/feed.rss"])

        assert result == []

    @patch("src.data.crawler.requests.Session.get")
    def test_get_newsapi_news_success(self, mock_get):
        """Test la récupération réussie des news NewsAPI"""
        # Mock de la réponse API
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "ok",
            "articles": [
                {
                    "title": "NVIDIA earnings beat expectations",
                    "description": "Strong AI demand drives growth",
                    "content": "Full article content...",
                    "url": "https://example.com/news1",
                    "publishedAt": "2024-01-01T10:00:00Z",
                    "source": {"name": "Reuters"},
                }
            ],
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.dict("os.environ", {"NEWSAPI_KEY": "test_key"}):
            crawler = NewsCrawler()
            result = crawler.get_newsapi_news()

        assert len(result) == 1
        assert result[0]["title"] == "NVIDIA earnings beat expectations"
        assert result[0]["ticker"] == "NVDA"
        assert result[0]["source"] == "Reuters"

    @patch("src.data.crawler.requests.Session.get")
    def test_get_newsapi_news_no_api_key(self, mock_get):
        """Test la récupération sans clé API NewsAPI"""
        crawler = NewsCrawler()
        result = crawler.get_newsapi_news()

        assert result == []
        mock_get.assert_not_called()

    @patch("src.data.crawler.requests.Session.get")
    def test_get_newsapi_news_api_error(self, mock_get):
        """Test la gestion d'erreur API NewsAPI"""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "error", "message": "Invalid API key"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.dict("os.environ", {"NEWSAPI_KEY": "invalid_key"}):
            crawler = NewsCrawler()
            result = crawler.get_newsapi_news()

        assert result == []

    def test_detect_ticker(self):
        """Test la détection de ticker"""
        crawler = NewsCrawler()

        # Test SPY
        assert crawler._detect_ticker("S&P 500 ETF rises") == "SPY"
        assert crawler._detect_ticker("SPDR shows growth") == "SPY"
        assert crawler._detect_ticker("Index performance") == "SPY"

        # Test NVDA
        assert crawler._detect_ticker("NVIDIA stock jumps") == "NVDA"
        assert crawler._detect_ticker("GPU maker reports earnings") == "NVDA"
        assert crawler._detect_ticker("AI company NVIDIA") == "NVDA"

        # Test aucun ticker
        assert crawler._detect_ticker("General market news") is None
        assert crawler._detect_ticker("") is None


@pytest.mark.crawler
class TestDataCrawler:
    """Tests pour le crawler principal"""

    def test_data_crawler_initialization(self):
        """Test l'initialisation du crawler principal"""
        crawler = DataCrawler()

        assert isinstance(crawler.price_crawler, PriceCrawler)
        assert isinstance(crawler.news_crawler, NewsCrawler)

    @patch("src.data.crawler.PriceCrawler.get_yahoo_prices")
    @patch("src.data.crawler.PriceCrawler.get_polygon_prices")
    def test_crawl_prices_success(self, mock_polygon, mock_yahoo):
        """Test le crawling de prix réussi"""
        # Mock des données Yahoo
        mock_yahoo.return_value = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2024-01-01", periods=3, freq="1min"),
                "close": [100, 101, 99],
                "ticker": ["SPY", "SPY", "SPY"],
            }
        )
        mock_polygon.return_value = pd.DataFrame()

        crawler = DataCrawler()
        result = crawler.crawl_prices(["SPY"])

        assert "SPY" in result
        assert len(result["SPY"]) == 3
        mock_yahoo.assert_called_once()
        mock_polygon.assert_not_called()  # Yahoo a réussi

    @patch("src.data.crawler.PriceCrawler.get_yahoo_prices")
    @patch("src.data.crawler.PriceCrawler.get_polygon_prices")
    def test_crawl_prices_yahoo_fallback(self, mock_polygon, mock_yahoo):
        """Test le fallback Yahoo -> Polygon"""
        # Mock Yahoo échoue, Polygon réussit
        mock_yahoo.return_value = pd.DataFrame()
        mock_polygon.return_value = pd.DataFrame(
            {
                "ts_utc": pd.date_range("2024-01-01", periods=2, freq="1min"),
                "close": [100, 101],
                "ticker": ["SPY", "SPY"],
            }
        )

        crawler = DataCrawler()
        result = crawler.crawl_prices(["SPY"])

        assert "SPY" in result
        assert len(result["SPY"]) == 2
        mock_yahoo.assert_called_once()
        mock_polygon.assert_called_once()

    @patch("src.data.crawler.PriceCrawler.get_yahoo_prices")
    @patch("src.data.crawler.PriceCrawler.get_polygon_prices")
    def test_crawl_prices_both_fail(self, mock_polygon, mock_yahoo):
        """Test l'échec des deux sources"""
        mock_yahoo.return_value = pd.DataFrame()
        mock_polygon.return_value = pd.DataFrame()

        crawler = DataCrawler()
        result = crawler.crawl_prices(["SPY"])

        assert result == {}

    @patch("src.data.crawler.NewsCrawler.get_rss_news")
    @patch("src.data.crawler.NewsCrawler.get_newsapi_news")
    def test_crawl_news_rss_only(self, mock_newsapi, mock_rss):
        """Test le crawling de news RSS seulement"""
        mock_rss.return_value = [{"title": "News 1", "ticker": "SPY"}, {"title": "News 2", "ticker": "NVDA"}]
        mock_newsapi.return_value = []

        crawler = DataCrawler()
        result = crawler.crawl_news(newsapi_enabled=False)

        assert len(result) == 2
        assert result[0]["ticker"] == "SPY"
        assert result[1]["ticker"] == "NVDA"
        mock_rss.assert_called_once()
        mock_newsapi.assert_not_called()

    @patch("src.data.crawler.NewsCrawler.get_rss_news")
    @patch("src.data.crawler.NewsCrawler.get_newsapi_news")
    def test_crawl_news_both_sources(self, mock_newsapi, mock_rss):
        """Test le crawling de news avec les deux sources"""
        mock_rss.return_value = [{"title": "RSS News 1", "ticker": "SPY"}]
        mock_newsapi.return_value = [{"title": "NewsAPI News 1", "ticker": "NVDA"}]

        crawler = DataCrawler()
        result = crawler.crawl_news(newsapi_enabled=True)

        assert len(result) == 2
        assert any(news["title"] == "RSS News 1" for news in result)
        assert any(news["title"] == "NewsAPI News 1" for news in result)
        mock_rss.assert_called_once()
        mock_newsapi.assert_called_once()

    @patch("src.data.crawler.NewsCrawler.get_rss_news")
    @patch("src.data.crawler.NewsCrawler.get_newsapi_news")
    def test_crawl_news_deduplication(self, mock_newsapi, mock_rss):
        """Test la déduplication des news"""
        mock_rss.return_value = [{"title": "Same News", "ticker": "SPY"}, {"title": "Unique News 1", "ticker": "NVDA"}]
        mock_newsapi.return_value = [
            {"title": "Same News", "ticker": "SPY"},  # Doublon
            {"title": "Unique News 2", "ticker": "SPY"},
        ]

        crawler = DataCrawler()
        result = crawler.crawl_news(newsapi_enabled=True)

        # Seulement 3 news uniques (pas de doublon)
        assert len(result) == 3
        titles = [news["title"] for news in result]
        assert titles.count("Same News") == 1  # Un seul exemplaire

    @patch("src.data.crawler.DataCrawler.crawl_prices")
    @patch("src.data.crawler.DataCrawler.crawl_news")
    def test_crawl_all(self, mock_crawl_news, mock_crawl_prices):
        """Test le crawling complet"""
        # Mock des résultats
        mock_crawl_prices.return_value = {
            "SPY": pd.DataFrame({"close": [100, 101]}),
            "NVDA": pd.DataFrame({"close": [200, 201]}),
        }
        mock_crawl_news.return_value = [{"title": "News 1", "ticker": "SPY"}, {"title": "News 2", "ticker": "NVDA"}]

        crawler = DataCrawler()
        result = crawler.crawl_all()

        assert "prices" in result
        assert "news" in result
        assert "crawl_time" in result
        assert "total_tickers" in result
        assert "total_news" in result

        assert result["total_tickers"] == 2
        assert result["total_news"] == 2
        assert result["crawl_time"] > 0

        mock_crawl_prices.assert_called_once()
        mock_crawl_news.assert_called_once()

    def test_crawl_all_default_parameters(self):
        """Test le crawling avec paramètres par défaut"""
        with patch.object(CONSTANTS, "TICKERS", ["SPY", "NVDA"]):
            with patch("src.data.crawler.DataCrawler.crawl_prices") as mock_prices:
                with patch("src.data.crawler.DataCrawler.crawl_news") as mock_news:
                    mock_prices.return_value = {}
                    mock_news.return_value = []

                    crawler = DataCrawler()
                    result = crawler.crawl_all()

                    # Vérifier que les paramètres par défaut sont utilisés
                    mock_prices.assert_called_once_with(tickers=["SPY", "NVDA"], price_interval=None, price_period=None)
                    mock_news.assert_called_once_with(rss_urls=None, newsapi_enabled=None, max_items=100)
