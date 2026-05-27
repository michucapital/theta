from thetadata_http import ThetaHttpClient
from config import CONFIG
import pandas as pd
import json
from typing import Optional

class ThetaDownloader:
    """Downloads SPY spot and options data from ThetaData"""

    def __init__(self):
        self.client = ThetaHttpClient(
            base_url=CONFIG.base_url,
            timeout_seconds=CONFIG.timeout_seconds
        )

        # Ensure data directories exist
        CONFIG.raw_data_path.mkdir(parents=True, exist_ok=True)
        CONFIG.merged_data_path.mkdir(parents=True, exist_ok=True)

    def _parse_json_response(self, response_bytes: bytes) -> tuple[pd.DataFrame, Optional[str]]:
        """
        Parse ThetaData JSON response for SPOT data.

        Returns:
            (DataFrame, next_page_url or None)
        """
        # Decode and parse JSON
        response_text = response_bytes.decode('utf-8')
        data = json.loads(response_text)

        # Extract column names from header
        columns = data['header']['format']

        # Extract data rows from response
        rows = data['response']

        # Create DataFrame
        df = pd.DataFrame(rows, columns=columns)

        # Check for pagination
        next_page = data['header'].get('next_page', None)

        return df, next_page

    def _parse_options_response(self, response_bytes: bytes) -> pd.DataFrame:
        """
        Parse nested options response from ThetaData.

        Structure:
        {
            "response": [
                {
                    "contract": {"strike": 500000, "right": "C", "expiration": 20260114, "root": "SPY"},
                    "ticks": [[ms_of_day, sequence, size, condition, price, date], ...]
                },
                ...
            ]
        }

        Returns:
            Flattened DataFrame with contract info + tick data
        """
        response_text = response_bytes.decode('utf-8')
        data = json.loads(response_text)

        tick_columns = data['header']['format']  # ['ms_of_day', 'sequence', 'size', 'condition', 'price', 'date']
        all_rows = []

        # Iterate through each contract
        for contract_data in data['response']:
            contract = contract_data['contract']
            ticks = contract_data['ticks']

            # Extract contract details
            strike_raw = contract['strike']
            strike = strike_raw / 1000.0  # Convert from thousandths to dollars
            right = contract['right']  # 'C' or 'P'
            expiration = contract['expiration']
            root = contract['root']

            # Expand each tick and add contract info
            for tick in ticks:
                row = {
                    # Tick data
                    'ms_of_day': tick[0],
                    'sequence': tick[1],
                    'size': tick[2],
                    'condition': tick[3],  # This is the condition/flags field!
                    'price': tick[4],
                    'date': tick[5],
                    # Contract data
                    'strike': strike,
                    'right': right,
                    'expiration': expiration,
                    'root': root
                }
                all_rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(all_rows)
        return df

    def _download_with_pagination(self, path: str, params: dict, data_type: str) -> pd.DataFrame:
        """
        Download data and handle pagination automatically.

        Args:
            path: API endpoint path
            params: Query parameters
            data_type: Description for progress messages (e.g., "Spot Trades")

        Returns:
            Complete DataFrame with all pages concatenated
        """
        all_dataframes = []
        page_num = 1
        next_page = None

        print(f"Downloading {data_type}... (handling pagination)")

        # First request
        response_bytes = self.client.get_bytes(path=path, params=params)
        print(f"  Page {page_num}: {len(response_bytes):,} bytes")

        df, next_page = self._parse_json_response(response_bytes)
        print(f"  Page {page_num}: {len(df):,} rows")
        all_dataframes.append(df)

        # Follow pagination if exists
        while next_page:
            page_num += 1
            print(f"  Fetching page {page_num}...")

            # Extract just the path and params from next_page URL
            # Format: http://127.0.0.1:25510/v2/page/4
            if '/page/' in next_page:
                page_path = next_page.split('25510')[1]  # Get everything after port
                response_bytes = self.client.get_bytes(path=page_path, params={})
                print(f"  Page {page_num}: {len(response_bytes):,} bytes")

                df, next_page = self._parse_json_response(response_bytes)
                print(f"  Page {page_num}: {len(df):,} rows")
                all_dataframes.append(df)
            else:
                print(f"  Warning: Unexpected next_page format: {next_page}")
                break

        # Concatenate all pages
        print(f"✓ Downloaded {page_num} page(s), concatenating...")
        final_df = pd.concat(all_dataframes, ignore_index=True)
        return final_df

    def download_spot_trades(self, date_str: str) -> pd.DataFrame:
        """
        Download SPY spot tick trades for a given date.

        Args:
            date_str: Date in YYYYMMDD format (e.g., "20260114")

        Returns:
            DataFrame with all SPY tick trades for that date
        """
        print(f"\n{'='*60}")
        print(f"Downloading SPY Spot Trades for {date_str}")
        print(f"{'='*60}")

        path = "/v2/hist/stock/trade"
        params = {
            "root": "SPY",
            "start_date": date_str,
            "end_date": date_str
        }

        print(f"Requesting: {self.client.base_url}{path}")
        print(f"Parameters: {params}")

        try:
            # Download with automatic pagination handling
            df = self._download_with_pagination(path, params, "SPY Spot Trades")

            print(f"✓ Total rows: {len(df):,}")
            print(f"✓ Columns: {list(df.columns)}")

            return df

        except Exception as e:
            print(f"\n✗ DOWNLOAD FAILED")
            print(f"Error: {e}")
            raise

    def download_options_trades(self, date_str: str) -> pd.DataFrame:
        """
        Download SPY 0DTE options trades for a given date.

        Args:
            date_str: Date in YYYYMMDD format (e.g., "20260114")

        Returns:
            DataFrame with all SPY 0DTE option trades for that date
        """
        print(f"\n{'='*60}")
        print(f"Downloading SPY 0DTE Options for {date_str}")
        print(f"{'='*60}")

        path = "/bulk_hist/option/trade"
        params = {
            "root": "SPY",
            "exp": date_str,  # expiration = trade date for 0DTE
            "start_date": date_str,
            "end_date": date_str
        }

        print(f"Requesting: {self.client.base_url}{path}")
        print(f"Parameters: {params}")
        print("Downloading 0DTE options (nested structure)...")

        try:
            # Download data
            response_bytes = self.client.get_bytes(path=path, params=params)
            print(f"✓ Downloaded {len(response_bytes):,} bytes")

            # Parse nested options response
            print("Parsing nested contract/ticks structure...")
            df = self._parse_options_response(response_bytes)

            print(f"✓ Parsed {len(df):,} total ticks")
            print(f"✓ Columns: {list(df.columns)}")

            # Show unique strikes and rights
            print(f"✓ Unique strikes: {df['strike'].nunique()}")
            print(f"✓ Calls vs Puts: {df['right'].value_counts().to_dict()}")

            return df

        except Exception as e:
            print(f"\n✗ DOWNLOAD FAILED")
            print(f"Error: {e}")
            raise


def preview_dataframe(df: pd.DataFrame, name: str, n_rows: int = 5):
    """Print a nice preview of a DataFrame"""
    print(f"\n{'='*60}")
    print(f"{name} Preview")
    print(f"{'='*60}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    print(f"\nColumn Names and Types:")
    for col in df.columns:
        print(f"  • {col}: {df[col].dtype}")

    print(f"\nFirst {n_rows} rows:")
    print(df.head(n_rows).to_string())

    print(f"\nLast {n_rows} rows:")
    print(df.tail(n_rows).to_string())
