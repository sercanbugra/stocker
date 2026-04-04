import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the functions from the scorer script
from scorer import (
    get_data_urls,
    load_and_preprocess_data,
    train_models,
    predict_next_gameweek,
)

class TestFPLScorer(unittest.TestCase):
    """Unit tests for the FPL scorer script."""

    def test_get_data_urls(self):
        """Test that the URL generation is correct."""
        urls = get_data_urls(2020, 2022)
        self.assertEqual(len(urls), 2)
        self.assertIn("https://www.football-data.co.uk/mmz4281/2021/E0.csv", urls)
        self.assertIn("https://www.football-data.co.uk/mmz4281/2122/E0.csv", urls)

    @patch("pandas.read_csv")
    def test_load_and_preprocess_data(self, mock_read_csv):
        """Test the data loading and preprocessing logic."""
        # Create mock CSV data
        mock_df1 = pd.DataFrame({
            "Date": ["01/01/2021"], "HomeTeam": ["Team A"], "AwayTeam": ["Team B"],
            "FTHG": [1], "FTAG": [0]
        })
        mock_df2 = pd.DataFrame({
            "Date": ["02/01/2021"], "HomeTeam": ["Team C"], "AwayTeam": ["Team D"],
            "FTHG": [2], "FTAG": [2]
        })
        mock_read_csv.side_effect = [mock_df1, mock_df2]

        urls = ["fake_url_1", "fake_url_2"]
        data, all_teams = load_and_preprocess_data(urls)

        # Check that the data is combined and processed correctly
        self.assertEqual(data.shape[0], 2)
        self.assertEqual(data["Season"].nunique(), 2)
        self.assertEqual(len(all_teams), 4)
        self.assertIn("Team A", all_teams)
        self.assertTrue(pd.api.types.is_datetime64_ns_dtype(data['Date']))

    @patch("pandas.read_csv", return_value=pd.DataFrame())
    def test_load_data_failure(self, mock_read_csv):
        """Test that the function handles failure to load any data."""
        mock_read_csv.side_effect = Exception("Failed to load")
        data, all_teams = load_and_preprocess_data(["bad_url"])
        self.assertIsNone(data)
        self.assertIsNone(all_teams)

    def test_train_models(self):
        """Test the model training function."""
        # Create a sample DataFrame for training
        data = pd.DataFrame({
            "Date": pd.to_datetime(["2022-08-01", "2023-08-01"]),
            "Season": ["2122", "2223"],
            "HomeTeam": ["Team A", "Team C"], "AwayTeam": ["Team B", "Team D"],
            "FTHG": [1, 3], "FTAG": [1, 0]
        })
        all_teams = ["Team A", "Team B", "Team C", "Team D"]

        model_home, model_away, encoder = train_models(data, all_teams)

        # Verify that models and an encoder are returned
        self.assertIsNotNone(model_home)
        self.assertIsNotNone(model_away)
        self.assertIsNotNone(encoder)
        # Check if the latest season has a higher weight
        self.assertEqual(data["sample_weight"].iloc[1], 5)

    @patch("pandas.read_csv")
    @patch("scorer.datetime")
    def test_predict_next_gameweek(self, mock_datetime, mock_read_csv):
        """Test the prediction of the next gameweek's scores."""
        # Mock the current time to be before the fixture date
        now = datetime(2024, 1, 1)
        mock_datetime.now.return_value = now
        
        # Create mock fixture data
        next_gw_date = now + timedelta(days=5)
        mock_fixtures_df = pd.DataFrame({
            "Div": ["E0", "E1"],
            "Date": [next_gw_date.strftime("%d/%m/%y"), "06/01/24"],
            "HomeTeam": ["Team A", "Team X"],
            "AwayTeam": ["Team B", "Team Y"]
        })
        mock_read_csv.return_value = mock_fixtures_df

        # Mock trained models and encoder
        mock_model_home = MagicMock()
        mock_model_home.predict.return_value = np.array([2.1])
        mock_model_away = MagicMock()
        mock_model_away.predict.return_value = np.array([0.9])
        
        mock_encoder = MagicMock()
        mock_encoder.transform = lambda x: {"Team A": 0, "Team B": 1}[x[0]]
        mock_encoder.classes_ = ["Team A", "Team B"]

        all_teams = ["Team A", "Team B"]

        # Run the prediction function
        with patch("builtins.print") as mock_print:
             predict_next_gameweek(mock_model_home, mock_model_away, mock_encoder, all_teams)
             
             # Check if the correct prediction is printed
             mock_print.assert_any_call("Team A vs Team B: Predicted Score -> 2 - 1")

    @patch("pandas.read_csv")
    @patch("scorer.datetime")
    def test_predict_gameweek_with_new_team(self, mock_datetime, mock_read_csv):
        """Test prediction when a new (e.g., promoted) team is in the fixtures."""
        now = datetime(2024, 1, 1)
        mock_datetime.now.return_value = now
        
        next_gw_date = now + timedelta(days=5)
        mock_fixtures_df = pd.DataFrame({
            "Div": ["E0"],
            "Date": [next_gw_date.strftime("%d/%m/%y")],
            "HomeTeam": ["New Team"],
            "AwayTeam": ["Team A"]
        })
        mock_read_csv.return_value = mock_fixtures_df
        
        mock_model_home = MagicMock()
        mock_model_home.predict.return_value = np.array([1.0])
        mock_model_away = MagicMock()
        mock_model_away.predict.return_value = np.array([1.0])

        # Initial encoder only knows about Team A and Team B
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder().fit(["Team A", "Team B"])
        
        all_teams = ["Team A", "Team B"]

        with patch("builtins.print") as mock_print:
            predict_next_gameweek(mock_model_home, mock_model_away, encoder, all_teams)
            
            # Check that the encoder was extended
            self.assertIn("New Team", encoder.classes_)
            # Check that a prediction was made
            mock_print.assert_any_call("New Team vs Team A: Predicted Score -> 1 - 1")


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
