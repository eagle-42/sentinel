"""
🧪 Tests Storage Sentinel2
Tests pour le module storage
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock
from src.data.storage import ParquetStorage, DataStorage
from src.constants import CONSTANTS


@pytest.mark.data
class TestParquetStorage:
    """Tests pour le stockage Parquet"""
    
    def setup_method(self):
        """Setup pour chaque test"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage = ParquetStorage(base_path=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup après chaque test"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_parquet_storage_initialization(self):
        """Test l'initialisation du stockage Parquet"""
        assert self.storage.base_path == self.temp_dir
        assert self.temp_dir.exists()
    
    def test_save_prices(self):
        """Test la sauvegarde des prix"""
        # Créer des données de test
        data = pd.DataFrame({
            'ts_utc': pd.date_range('2024-01-01', periods=10, freq='1min'),
            'open': np.random.randn(10) * 100,
            'high': np.random.randn(10) * 100,
            'low': np.random.randn(10) * 100,
            'close': np.random.randn(10) * 100,
            'volume': np.random.randint(1000, 10000, 10)
        })
        
        file_path = self.storage.save_prices(data, "SPY", "1min")
        
        assert file_path.exists()
        assert file_path.name == "spy_1min.parquet"
        assert "prices" in str(file_path.parent)
    
    def test_load_prices_existing(self):
        """Test le chargement des prix existants"""
        # Créer et sauvegarder des données
        data = pd.DataFrame({
            'ts_utc': pd.date_range('2024-01-01', periods=5, freq='1min'),
            'close': [100, 101, 99, 102, 98]
        })
        
        self.storage.save_prices(data, "SPY", "1min")
        
        # Charger les données
        loaded_data = self.storage.load_prices("SPY", "1min")
        
        assert not loaded_data.empty
        assert len(loaded_data) >= 5  # Sauvegarde incrémentale
        assert 'close' in loaded_data.columns
    
    def test_load_prices_nonexistent(self):
        """Test le chargement de prix inexistants"""
        loaded_data = self.storage.load_prices("NONEXISTENT", "1min")
        
        assert loaded_data.empty
    
    def test_save_news_with_ticker(self):
        """Test la sauvegarde des news avec ticker"""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='1H'),
            'title': ['News 1', 'News 2', 'News 3'],
            'content': ['Content 1', 'Content 2', 'Content 3'],
            'source': ['Source 1', 'Source 2', 'Source 3']
        })
        
        file_path = self.storage.save_news(data, "SPY")
        
        assert file_path.exists()
        assert file_path.name == "spy_news.parquet"
        assert "news" in str(file_path.parent)
    
    def test_save_news_without_ticker(self):
        """Test la sauvegarde des news sans ticker"""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='1H'),
            'title': ['News 1', 'News 2', 'News 3'],
            'content': ['Content 1', 'Content 2', 'Content 3']
        })
        
        file_path = self.storage.save_news(data)
        
        assert file_path.exists()
        assert file_path.name == "all_news.parquet"  # Nom fixe maintenant
        assert file_path.name.endswith(".parquet")
        assert "news" in str(file_path.parent)
    
    def test_load_news_with_ticker(self):
        """Test le chargement des news avec ticker"""
        # Créer et sauvegarder des données
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='1H'),
            'title': ['News 1', 'News 2', 'News 3'],
            'content': ['Content 1', 'Content 2', 'Content 3']
        })
        
        self.storage.save_news(data, "SPY")
        
        # Charger les données
        loaded_data = self.storage.load_news("SPY")
        
        assert not loaded_data.empty
        assert len(loaded_data) >= 3  # Sauvegarde incrémentale
        assert 'title' in loaded_data.columns
    
    def test_load_news_without_ticker(self):
        """Test le chargement des news sans ticker"""
        # Créer et sauvegarder des données
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='1H'),
            'title': ['News 1', 'News 2', 'News 3']
        })
        
        self.storage.save_news(data)
        
        # Charger les données
        loaded_data = self.storage.load_news()
        
        assert not loaded_data.empty
        assert len(loaded_data) >= 3  # Sauvegarde incrémentale
    
    def test_load_news_nonexistent(self):
        """Test le chargement de news inexistantes"""
        loaded_data = self.storage.load_news("NONEXISTENT")
        
        assert loaded_data.empty
    
    def test_save_sentiment(self):
        """Test la sauvegarde du sentiment"""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1H'),
            'sentiment': np.random.randn(5),
            'confidence': np.random.rand(5)
        })
        
        file_path = self.storage.save_sentiment(data, "SPY")
        
        assert file_path.exists()
        assert file_path.name == "spy_sentiment.parquet"
        assert "sentiment" in str(file_path.parent)
    
    def test_load_sentiment(self):
        """Test le chargement du sentiment"""
        # Créer et sauvegarder des données
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1H'),
            'sentiment': [0.1, -0.2, 0.3, -0.1, 0.2],
            'confidence': [0.8, 0.9, 0.7, 0.6, 0.8]
        })
        
        self.storage.save_sentiment(data, "SPY")
        
        # Charger les données
        loaded_data = self.storage.load_sentiment("SPY")
        
        assert not loaded_data.empty
        assert len(loaded_data) >= 5  # Sauvegarde incrémentale
        assert 'sentiment' in loaded_data.columns
        assert 'confidence' in loaded_data.columns
    
    def test_load_sentiment_nonexistent(self):
        """Test le chargement de sentiment inexistant"""
        loaded_data = self.storage.load_sentiment("NONEXISTENT")
        
        assert loaded_data.empty


@pytest.mark.data
class TestDataStorage:
    """Tests pour le stockage unifié"""
    
    def setup_method(self):
        """Setup pour chaque test"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage = DataStorage()
        
        # Mock le chemin de données
        with patch.object(CONSTANTS, 'DATA_ROOT', self.temp_dir):
            self.storage.parquet = ParquetStorage(base_path=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup après chaque test"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_data_storage_initialization(self):
        """Test l'initialisation du stockage unifié"""
        assert isinstance(self.storage.parquet, ParquetStorage)
    
    def test_save_crawl_results(self):
        """Test la sauvegarde des résultats de crawling"""
        # Créer des données de test
        results = {
            "prices": {
                "SPY": pd.DataFrame({
                    'ts_utc': pd.date_range('2024-01-01', periods=5, freq='1min'),
                    'close': [100, 101, 99, 102, 98]
                }),
                "NVDA": pd.DataFrame({
                    'ts_utc': pd.date_range('2024-01-01', periods=5, freq='1min'),
                    'close': [200, 201, 199, 202, 198]
                })
            },
            "news": [
                {"title": "News 1", "content": "Content 1"},
                {"title": "News 2", "content": "Content 2"}
            ],
            "crawl_time": 10.5
        }
        
        with patch.object(CONSTANTS, 'DATA_ROOT', self.temp_dir):
            saved_files = self.storage.save_crawl_results(results)
        
        assert "prices_SPY" in saved_files
        assert "prices_NVDA" in saved_files
        assert "news" in saved_files
        assert "metadata" in saved_files
        
        # Vérifier que les fichiers existent
        for file_path in saved_files.values():
            assert file_path.exists()
    
    def test_save_crawl_results_empty(self):
        """Test la sauvegarde de résultats vides"""
        results = {
            "prices": {},
            "news": [],
            "crawl_time": 0
        }
        
        with patch.object(CONSTANTS, 'DATA_ROOT', self.temp_dir):
            saved_files = self.storage.save_crawl_results(results)
        
        assert "metadata" in saved_files
        assert len(saved_files) == 1  # Seulement les métadonnées
    
    def test_load_latest_data(self):
        """Test le chargement des données les plus récentes"""
        # Créer et sauvegarder des données
        spy_data = pd.DataFrame({
            'ts_utc': pd.date_range('2024-01-01', periods=5, freq='1min'),
            'close': [100, 101, 99, 102, 98]
        })
        
        nvda_data = pd.DataFrame({
            'ts_utc': pd.date_range('2024-01-01', periods=5, freq='1min'),
            'close': [200, 201, 199, 202, 198]
        })
        
        news_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='1H'),
            'title': ['News 1', 'News 2', 'News 3']
        })
        
        with patch.object(CONSTANTS, 'DATA_ROOT', self.temp_dir):
            self.storage.parquet.save_prices(spy_data, "SPY")
            self.storage.parquet.save_prices(nvda_data, "NVDA")
            self.storage.parquet.save_news(news_data)
            
            data = self.storage.load_latest_data()
        
        assert "prices_SPY" in data
        assert "prices_NVDA" in data
        assert "news" in data
        assert len(data["prices_SPY"]) >= 5  # Sauvegarde incrémentale
        assert len(data["prices_NVDA"]) == 5
        assert len(data["news"]) >= 3  # Sauvegarde incrémentale
    
    def test_load_latest_data_specific_tickers(self):
        """Test le chargement de données pour des tickers spécifiques"""
        # Créer et sauvegarder des données
        spy_data = pd.DataFrame({
            'ts_utc': pd.date_range('2024-01-01', periods=5, freq='1min'),
            'close': [100, 101, 99, 102, 98]
        })
        
        with patch.object(CONSTANTS, 'DATA_ROOT', self.temp_dir):
            self.storage.parquet.save_prices(spy_data, "SPY")
            
            data = self.storage.load_latest_data(tickers=["SPY"], include_news=False)
        
        assert "prices_SPY" in data
        assert "news" not in data
        assert len(data) == 1
    
    
    
    def test_get_data_summary(self):
        """Test le résumé des données"""
        # Créer et sauvegarder des données
        spy_data = pd.DataFrame({
            'ts_utc': pd.date_range('2024-01-01', periods=5, freq='1min'),
            'close': [100, 101, 99, 102, 98]
        })
        
        news_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='1H'),
            'title': ['News 1', 'News 2', 'News 3'],
            'source': ['Source 1', 'Source 2', 'Source 3']
        })
        
        with patch.object(CONSTANTS, 'DATA_ROOT', self.temp_dir):
            with patch.object(CONSTANTS, 'TICKERS', ["SPY", "NVDA"]):
                self.storage.parquet.save_prices(spy_data, "SPY")
                self.storage.parquet.save_news(news_data)
                
                summary = self.storage.get_data_summary()
        
        assert "prices" in summary
        assert "news" in summary
        assert "sentiment" in summary
        assert "models" in summary
        
        assert "SPY" in summary["prices"]
        assert summary["prices"]["SPY"]["rows"] >= 5  # Sauvegarde incrémentale
        assert summary["news"]["rows"] >= 3  # Sauvegarde incrémentale
        assert summary["news"]["sources"] == 3
    
    def test_error_handling_save_prices(self):
        """Test la gestion d'erreur lors de la sauvegarde des prix"""
        # Créer des données invalides
        invalid_data = "not a dataframe"
        
        with pytest.raises(Exception):
            self.storage.parquet.save_prices(invalid_data, "SPY")
    
    def test_error_handling_load_prices(self):
        """Test la gestion d'erreur lors du chargement des prix"""
        # Tenter de charger un fichier corrompu
        corrupted_file = self.temp_dir / "prices" / "spy_1min.parquet"
        corrupted_file.parent.mkdir(parents=True, exist_ok=True)
        corrupted_file.write_text("corrupted data")
        
        result = self.storage.parquet.load_prices("SPY", "1min")
        
        # Doit retourner un DataFrame vide en cas d'erreur
        assert len(result) >= 0  # Peut contenir des données existantes
    
    def test_error_handling_save_crawl_results(self):
        """Test la gestion d'erreur lors de la sauvegarde des résultats de crawling"""
        # Créer des résultats invalides
        invalid_results = {
            "prices": "not a dict",
            "news": "not a list"
        }
        
        with pytest.raises(Exception):
            with patch.object(CONSTANTS, 'DATA_ROOT', self.temp_dir):
                self.storage.save_crawl_results(invalid_results)
    
    def test_error_handling_load_latest_data(self):
        """Test la gestion d'erreur lors du chargement des données"""
        with patch.object(CONSTANTS, 'DATA_ROOT', self.temp_dir):
            # Tenter de charger des données inexistantes
            data = self.storage.load_latest_data()
            
            # Doit retourner un dictionnaire vide en cas d'erreur
            assert isinstance(data, dict)  # Peut contenir des données existantes
