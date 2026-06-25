import argparse
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import main


class MarkowitzAnalysisTest(unittest.TestCase):
    def test_calculate_daily_returns(self):
        prices = pd.DataFrame(
            {
                "AAA": [100.0, 110.0, 121.0],
                "BBB": [50.0, 55.0, 60.5],
            },
            index=pd.date_range("2024-01-01", periods=3),
        )

        returns = main.calculate_daily_returns(prices)

        self.assertEqual(list(returns.columns), ["AAA", "BBB"])
        np.testing.assert_allclose(returns["AAA"].to_numpy(), [0.10, 0.10])
        np.testing.assert_allclose(returns["BBB"].to_numpy(), [0.10, 0.10])

    def test_portfolio_metrics_annualizes_values(self):
        expected_returns = pd.Series({"AAA": 0.001, "BBB": 0.0005})
        covariance = pd.DataFrame(
            [[0.0001, 0.00002], [0.00002, 0.00005]],
            index=["AAA", "BBB"],
            columns=["AAA", "BBB"],
        )
        weights = np.array([0.60, 0.40])

        annual_return, annual_risk, sharpe = main.portfolio_metrics(
            weights,
            expected_returns,
            covariance,
            risk_free_rate=0.02,
        )

        self.assertGreater(annual_return, 0)
        self.assertGreater(annual_risk, 0)
        self.assertAlmostEqual(sharpe, (annual_return - 0.02) / annual_risk)

    def test_discover_assets_from_manual_configuration(self):
        with TemporaryDirectory() as temporary_dir:
            data_dir = Path(temporary_dir)
            (data_dir / "aaa.xlsx").touch()
            (data_dir / "bbb.xlsx").touch()

            assets = main.discover_assets(
                data_dir,
                "*.xlsx",
                ["aaa=aaa.xlsx", "bbb=bbb.xlsx"],
            )

        self.assertEqual([asset.ticker for asset in assets], ["AAA", "BBB"])

    def test_full_analysis_generates_report_and_chart(self):
        with TemporaryDirectory() as temporary_dir:
            data_dir = Path(temporary_dir)
            dates = pd.date_range("2024-01-01", periods=80, freq="B")
            base = pd.DataFrame({"Date": dates})
            for ticker, start, step in [
                ("AAA", 100, 0.35),
                ("BBB", 80, 0.20),
                ("CCC", 50, -0.03),
            ]:
                data = base.copy()
                data["Close"] = start + np.arange(len(dates)) * step
                data.to_excel(data_dir / f"{ticker}_year1.xlsx", index=False)

            args = argparse.Namespace(
                data_dir=data_dir,
                pattern="*_year1.xlsx",
                asset=[],
                output=data_dir / "reports" / "analysis.xlsx",
                chart=data_dir / "reports" / "frontier.png",
                portfolios=250,
                frontier_points=20,
                risk_free_rate=0.0,
                seed=7,
            )

            output_path, chart_path, summary = main.run_analysis(args)

            self.assertTrue(output_path.exists())
            self.assertTrue(chart_path.exists())
            self.assertEqual(len(summary), 3)
            self.assertIn("Carteira", summary.columns)


if __name__ == "__main__":
    unittest.main()
