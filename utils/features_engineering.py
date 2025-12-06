import gc
import pandas as pd
import numpy as np
from numba import jit, njit
from .chart_patterns import CHART_PATTERN_FUNCS

class FeatureEngineerOptimized:
    """
    Optimized FeatureEngineer that only processes recent rows for incremental updates.
    
    Key Optimizations:
    1. Only loads last N rows needed for calculations.
    2. Only recalculates features for new/updated rows.
    3. Uses vectorized operations throughout.
    4. Caches intermediate calculations.
    
    CORRECTIONS:
    - MAX_LOOKBACK increased to 1000. 250 is insufficient for ema_200 
      convergence and breaks all .cumsum() indicators (OBV, VWAP, AD, VPT).
    - Fixed Ichimoku Lagging Span data-leak (was peeking 26 bars
      into the future).
    - Fixed Swing High/Low data-leak (was identifying swings 5 bars
      before they were confirmed).
    - Vectorized slow loops for OBV, NVI, and swing point helpers.
    
    CATEGORICAL ENCODINGS (for Deep Learning):
    All categorical features use numerical encoding instead of strings.
    Reference mappings below for embedding layers or one-hot encoding.
    """
    
    # Maximum lookback period across all indicators
    # Needs to be ~5x the longest EMA (200) for convergence
    # and to provide a stable base for cumulative indicators.
    MAX_LOOKBACK = 1050
    BATCH_SIZE = 25000  # Process 25k rows at a time to keep RAM usage low
    
    # ========== CATEGORICAL FEATURE MAPPINGS (for DL embedding/one-hot) ==========
    # Use these mappings to interpret numerical categorical features or build embeddings
    
    PULLBACK_DEPTH_MAP = {
        0: 'none',      # No significant pullback
        1: 'shallow',   # -1% to -5% pullback
        2: 'moderate',  # -5% to -10% pullback  
        3: 'deep',      # < -10% pullback
    }
    PULLBACK_DEPTH_INV = {v: k for k, v in PULLBACK_DEPTH_MAP.items()}
    N_PULLBACK_DEPTH_CLASSES = 4
    
    PULLBACK_STAGE_MAP = {
        0: 'neutral',         # No clear stage
        1: 'impulse_up',      # Strong upward move (>3%)
        2: 'impulse_down',    # Strong downward move (>3%)
        3: 'pullback_early',  # Just started pulling back (1-3 bars)
        4: 'pullback_mid',    # In middle of pullback (4-7 bars, <8% depth)
        5: 'pullback_late',   # Near completion (>=5% depth)
        6: 'resumption_bull', # Bullish trend resuming
        7: 'resumption_bear', # Bearish trend resuming
    }
    PULLBACK_STAGE_INV = {v: k for k, v in PULLBACK_STAGE_MAP.items()}
    N_PULLBACK_STAGE_CLASSES = 8
    
    # Temporal features extracted from datetime columns (18 features total):
    # Cyclical: hour_sin/cos, day_of_week_sin/cos, day_of_month_sin/cos,
    #           month_sin/cos, week_of_year_sin/cos, minute_sin/cos
    # Sessions: session_sydney, session_tokyo, session_london, session_newyork
    # Overlaps: session_london_ny_overlap, session_tokyo_london_overlap
    # Flags: is_friday, is_monday, quarter
    N_TEMPORAL_FEATURES = 19
    
    def __init__(self, db_connector=None):
        self.db = db_connector

    def generate_features(self, df: pd.DataFrame, batch_processing: bool = True) -> pd.DataFrame:
        """
        Public entry point for the Training Pipeline.
        Handles large datasets by automatically switching to batch processing.
        """
        # 1. Standardize Columns
        if 'tick_volume' in df.columns and 'volume' not in df.columns:
            df['volume'] = df['tick_volume']
            
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            missing = [c for c in required if c not in df.columns]
            raise ValueError(f"DataFrame missing required columns: {missing}")

        # 2. Decide Strategy
        if len(df) > self.BATCH_SIZE and batch_processing:
            print(f"⚡ Large dataset detected ({len(df)} rows). Using Batched Processing...")
            return self._process_in_batches(df)
        else:
            return self._sanitize_column_names(self.calculate_indicators(df))

    def _process_in_batches(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Splits a massive DataFrame into chunks, calculates features with 
        correct lookback overlaps, and stitches them back together.
        """
        total_rows = len(df)
        num_batches = (total_rows // self.BATCH_SIZE) + 1
        processed_chunks = []
        
        print(f"   Splitting into {num_batches} batches of ~{self.BATCH_SIZE} rows...")
        
        for i in range(num_batches):
            # Calculate slice indices
            start_idx = i * self.BATCH_SIZE
            end_idx = min((i + 1) * self.BATCH_SIZE, total_rows)
            
            if start_idx >= end_idx:
                break
                
            # CRITICAL: Include Lookback Buffer
            # For batch 2+, we need to look back into batch 1 to calculate EMAs/RSIs correctly.
            buffer_start = max(0, start_idx - self.MAX_LOOKBACK)
            
            # Slice the chunk (Buffer + Current Batch)
            df_chunk = df.iloc[buffer_start:end_idx].copy()
            
            # Calculate features on this chunk
            # print(f"   Processing Batch {i+1}/{num_batches} (Rows {buffer_start} to {end_idx})...")
            df_chunk_features = self.calculate_indicators(df_chunk)
            
            # Sanitize
            df_chunk_features = self._sanitize_column_names(df_chunk_features)
            
            # Trim the buffer (we only want the *new* calculated rows)
            # The "actual" rows for this batch start at (start_idx - buffer_start) relative to the chunk
            actual_start_rel = start_idx - buffer_start
            df_final_chunk = df_chunk_features.iloc[actual_start_rel:]
            
            processed_chunks.append(df_final_chunk)
            
            # Memory Cleanup
            del df_chunk, df_chunk_features
            gc.collect()
            
        # Stitch back together
        print("   ✅ Merging batches...")
        full_features = pd.concat(processed_chunks, axis=0)
        
        # Verify integrity
        if len(full_features) != len(df):
            print(f"⚠️ Warning: Input rows {len(df)} != Output rows {len(full_features)}")
            
        return full_features

    def calculate_and_save_incremental_features(self, pair_tf, new_rows_count=1):
        """
        Calculates features for a *known* number of new rows.
        This is ideal for a websocket handler.
        """
        print(f"\n{'='*80}")
        print(f"INCREMENTAL FEATURE UPDATE: {pair_tf}")
        print(f"Processing {new_rows_count} new row(s)")
        print(f"{'='*80}")
        
        try:
            
            # Load only the required window of data
            # We need MAX_LOOKBACK rows for history + the new rows
            rows_needed = self.MAX_LOOKBACK + new_rows_count
            
            # Load N-1 old rows, so that when we add N new rows, the
            # total chunk size is correct for calculation.
            # Example: 1 new row. We load 1000 rows (999 old + 1 new).
            # We need to load 1000 rows total, *including* the new row.
            
            # Simplified: Just load the last M rows from the DB.
            # This assumes new_rows_count is the number of rows *already in the DB*
            # that we need to process.
            
            # Let's assume the new row is NOT in the DB yet.
            # 1. Load M old rows
            # 2. Get N new rows (from websocket)
            # 3. Concat [M_old, N_new]
            # 4. Calculate features on M+N
            # 5. Save N new feature rows
            
            # The logic in your file implies the new rows are already in the
            # raw DB. So we load the *total* chunk.
            rows_needed = self.MAX_LOOKBACK + new_rows_count
            df_raw = self.db.load_raw_ohlcv(pair_tf, limit=rows_needed)
            
            if df_raw is None or len(df_raw) < self.MAX_LOOKBACK:
                print(f"⚠️ Insufficient data for {pair_tf}. Need {self.MAX_LOOKBACK}, got {len(df_raw) if df_raw is not None else 0}")
                return False
            
            # Ensure we have enough data for the calculation
            if len(df_raw) < rows_needed:
                print(f"⚠️ Warning: Loaded {len(df_raw)} rows, less than {rows_needed} requested.")
                actual_new_rows = max(1, len(df_raw) - self.MAX_LOOKBACK + 1)
                print(f"   Adjusting processing count to {actual_new_rows} row(s).")
                new_rows_count = actual_new_rows

            df_features = self.calculate_indicators(df_raw)
            df_features = self._sanitize_column_names(df_features)
            
            df_to_update = df_features.tail(new_rows_count)
            
            self.db.update_features(df_to_update, pair_tf) # Assumes this appends
            print(f"✅ Appended {len(df_to_update)} row(s) for {pair_tf}")
            return True
            
        except Exception as e:
            import traceback
            print(f"❌ Error in incremental update for {pair_tf}: {e}")
            print(traceback.format_exc())
            return False
    
    def calculate_and_save_all_features(self):
        """
        Smarter "catch-up" calculation with BATCHING for massive updates.
        """
        print("\n" + "="*80)
        print("FEATURE ENGINEERING STARTING: Smart Batch Mode")
        print("="*80)
        
        try:
            all_pair_tfs = self.db.get_all_raw_pair_tfs()
        except AttributeError:
            print("❌ DBConnector not fully implemented (missing get_all_raw_pair_tfs).")
            return
        
        if not all_pair_tfs:
            print("❌ No raw OHLCV data found in database.")
            return
        
        processed_count = 0
        
        # Define a safe batch size to prevent RAM explosion
        # 25,000 rows per batch is a safe balance for most machines
        BATCH_SIZE = 25000 
        
        for pair_tf in all_pair_tfs:
            print(f"Processing features for {pair_tf}...")

            # 1. Check counts
            try:
                feature_count = self.db.count_features(pair_tf) 
                raw_count = self.db.count_raw_rows(pair_tf)
            except AttributeError:
                # Fallback if DB method missing
                df_raw = self.db.load_raw_ohlcv(pair_tf)
                raw_count = len(df_raw) if df_raw is not None else 0
                feature_count = 0

            if raw_count < 50:
                print(f"⚠️ Skipping {pair_tf}: Insufficient raw data ({raw_count} rows).")
                continue
            
            # 2. Determine work needed
            new_rows_count = 0
            is_incremental = False
            
            if feature_count == 0:
                # Full calculation needed
                new_rows_count = raw_count
                is_incremental = False
                print(f"  No features found. Performing full calculation for {raw_count} rows.")
            elif raw_count > feature_count:
                # Catch-up needed
                new_rows_count = raw_count - feature_count
                is_incremental = True
                print(f"  Found {feature_count} existing features. Catching up {new_rows_count} rows.")
            else:
                print(f"✅ Features up-to-date ({feature_count} rows).")
                processed_count += 1
                continue

            # 3. Load Data
            # If the dataset is massive, we still load the raw data (lightweight),
            # but we process features (heavy) in chunks.
            rows_to_load = self.MAX_LOOKBACK + new_rows_count
            
            # Safety cap for logs: only print if not huge
            if rows_to_load < 100000:
                print(f"  Loading last {rows_to_load} raw rows...")
            else:
                print(f"  Loading massive dataset ({rows_to_load} rows)... please wait.")

            df_to_process = self.db.load_raw_ohlcv(pair_tf, limit=rows_to_load)

            if df_to_process is None or len(df_to_process) < self.MAX_LOOKBACK:
                 print(f"⚠️ Insufficient data loaded. Skipping.")
                 continue

            # 4. Processing Logic (with Batching)
            import gc # Garbage collector interface
            
            try:
                # If the job is small, do it in one go
                if new_rows_count <= BATCH_SIZE:
                    df_features = self.calculate_indicators(df_to_process)
                    df_features = self._sanitize_column_names(df_features)
                    
                    if is_incremental:
                        df_to_save = df_features.tail(new_rows_count)
                        self.db.update_features(df_to_save, pair_tf)
                    else:
                        self.db.update_features(df_features, pair_tf)
                    print(f"✅ Updated {new_rows_count} rows.")
                
                else:
                    # --- BATCH PROCESSING ---
                    print(f"  ⚠️ Large dataset detected. Processing in batches of {BATCH_SIZE}...")
                    
                    # We need to process the 'new' part of df_to_process in chunks.
                    # df_to_process structure: [Lookback Data] + [New Data to Process]
                    
                    # Start index of the 'new' data within df_to_process
                    start_index_of_new = len(df_to_process) - new_rows_count
                    
                    total_batches = (new_rows_count // BATCH_SIZE) + 1
                    
                    for i in range(total_batches):
                        # Determine slice indices relative to df_to_process
                        batch_start_rel = start_index_of_new + (i * BATCH_SIZE)
                        batch_end_rel = min(start_index_of_new + ((i + 1) * BATCH_SIZE), len(df_to_process))
                        
                        current_batch_size = batch_end_rel - batch_start_rel
                        if current_batch_size <= 0: break
                        
                        # We must include lookback for THIS batch
                        # Slice from: (batch_start - LOOKBACK) to (batch_end)
                        calc_start = max(0, batch_start_rel - self.MAX_LOOKBACK)
                        calc_end = batch_end_rel
                        
                        df_chunk = df_to_process.iloc[calc_start:calc_end].copy()
                        
                        print(f"    Batch {i+1}/{total_batches}: Calculating {current_batch_size} rows...")
                        
                        # Calculate features for this chunk
                        df_chunk_features = self.calculate_indicators(df_chunk)
                        df_chunk_features = self._sanitize_column_names(df_chunk_features)
                        
                        # Extract ONLY the new calculated rows for saving
                        # (The last 'current_batch_size' rows)
                        df_to_save = df_chunk_features.tail(current_batch_size)
                        
                        # Save immediately
                        self.db.update_features(df_to_save, pair_tf)
                        
                        # Memory cleanup
                        del df_chunk
                        del df_chunk_features
                        del df_to_save
                        gc.collect() 
                        
                    print(f"✅ Batch processing complete for {pair_tf}")

                processed_count += 1

            except Exception as e:
                import traceback
                print(f"❌ Error processing {pair_tf}: {e}")
                print(traceback.format_exc())
    
        print(f"\n✨ FEATURE ENGINEERING COMPLETE. {processed_count} datasets processed.")
    
    # ==================== OPTIMIZED HELPER FUNCTIONS ====================
    
    @staticmethod
    def sma(series, period):
        return series.rolling(window=period, min_periods=period).mean()
    
    @staticmethod
    def ema(series, period):
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        return series.ewm(span=period, adjust=False, min_periods=period).mean()
    
    @staticmethod
    def wma(series, period):
        weights = np.arange(1, period + 1)
        return series.rolling(period, min_periods=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    @staticmethod
    def rma(series, period):
        alpha = 1.0 / period
        return series.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    
    @staticmethod
    def std(series, period):
        return series.rolling(window=period, min_periods=period).std()
    
    @staticmethod
    def typical_price(high, low, close):
        return (high + low + close) / 3
    
    @staticmethod
    def true_range(high, low, close):
        hl = high - low
        hc = np.abs(high - close.shift(1))
        lc = np.abs(low - close.shift(1))
        return pd.concat([hl, hc, lc], axis=1).max(axis=1)
    
    def _sanitize_column_names(self, df):
        df.columns = df.columns.str.replace('.', '_', regex=False)
        df = df.loc[:, ~df.columns.duplicated(keep='last')]
        return df
    
    # In features_engineering.py, add this helper:
    def get_optimal_lookback(timeframe):
        """Get optimal swing lookback based on timeframe."""
        lookback_map = {
            '1m': 1,   # Very sensitive for scalping
            '5m': 2,   # Standard for intraday
            '15m': 2,  # Standard for swing detection
            '1h': 3,   # Slightly less sensitive
            '4h': 3,   # Filter minor noise
        }
        return lookback_map.get(timeframe, 2)  # Default to 2
    
    # ==================== OPTIMIZED INDICATOR CALCULATION ====================    
    def calculate_indicators(self, df):
        """
        Optimized indicator calculation using vectorization and batching.
        """
        
        df = df.copy()
        
        if len(df) < 50:
            print("⚠️ Insufficient data for indicator calculation")
            return df
        
        # Pre-calculate common values once
        tr = self.true_range(df['high'], df['low'], df['close'])
        tp = self.typical_price(df['high'], df['low'], df['close'])
        
        # Dictionary to collect all new columns
        cols = {}
        
        # ============ MOVING AVERAGES (Batch 1) ============
        for period in [7, 20, 50, 100, 200]:
            cols[f'sma_{period}'] = self.sma(df['close'], period)
        
        for period in [9, 12, 26, 50, 200]:
            cols[f'ema_{period}'] = self.ema(df['close'], period)
        
        cols['wma_20'] = self.wma(df['close'], 20)
        cols['smma'] = self.rma(df['close'], 14)
        
        ema = self.ema(df['close'], 20)
        ema_ema = self.ema(ema, 20)
        cols['dema'] = 2 * ema - ema_ema
        
        ema1 = self.ema(df['close'], 20)
        ema2 = self.ema(ema1, 20)
        ema3 = self.ema(ema2, 20)
        cols['tema'] = 3 * ema1 - 3 * ema2 + ema3
        
        half_length = 10
        sqrt_length = int(np.sqrt(20))
        wma1 = self.wma(df['close'], half_length)
        wma2 = self.wma(df['close'], 20)
        raw_hma = 2 * wma1 - wma2
        cols['hma'] = self.wma(raw_hma, sqrt_length)
        
        # LSMA (This loop is still slow, recommend pandas-ta)
        def lsma_vectorized(series, period):
            result = pd.Series(np.nan, index=series.index)
            for i in range(period - 1, len(series)):
                y = series.iloc[i-period+1:i+1].values
                x = np.arange(period)
                z = np.polyfit(x, y, 1)
                result.iloc[i] = z[0] * (period - 1) + z[1]
            return result
        cols['lsma'] = lsma_vectorized(df['close'], 25)
        
        # McGinley Dynamic (This loop is still slow, recommend pandas-ta)
        mcg = df['close'].copy()
        for i in range(1, len(df)):
            if pd.notna(mcg.iloc[i-1]) and mcg.iloc[i-1] != 0:
                mcg.iloc[i] = mcg.iloc[i-1] + (
                    (df['close'].iloc[i] - mcg.iloc[i-1]) / 
                    (20 * (df['close'].iloc[i] / mcg.iloc[i-1]) ** 4)
                )
        cols['mcginley'] = mcg
        
        cols['vwma'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}
        
        # ============ MACD ============
        cols['macd'] = df['ema_12'] - df['ema_26']
        cols['macd_signal'] = self.ema(cols['macd'], 9)
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}
        cols['macd_hist'] = df['macd'] - df['macd_signal']
        cols['macd_mean'] = df['macd'].rolling(window=50, min_periods=1).mean()
        
        # ============ ADX (Vectorized) ============
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        pos_dm = pd.Series(np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0), index=df.index)
        neg_dm = pd.Series(np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0), index=df.index)
        atr_14 = self.rma(tr, 14)
        di_plus = 100 * self.rma(pos_dm, 14) / atr_14
        di_minus = 100 * self.rma(neg_dm, 14) / atr_14
        cols['di_plus'] = di_plus
        cols['di_minus'] = di_minus
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        cols['adx'] = self.rma(dx, 14)
        
        # ============ AROON (Numba Accelerated) ============
        @njit(cache=True)
        def numba_aroon(series_arr: np.ndarray, period: int, is_high: bool) -> np.ndarray:
            """
            Calculates the Aroon Up or Aroon Down indicator using Numba.
            The Aroon value is calculated based on how many periods have passed
            since the maximum (Aroon Up) or minimum (Aroon Down) price was observed
            within the current lookback period.
            """
            n = len(series_arr)
            # Initialize with NaN for the first (period-1) elements as they don't 
            # have a full window yet.
            result = np.full(n, np.nan, dtype=np.float64) 

            # Start loop from period - 1 to ensure a full window is available
            for i in range(period - 1, n):
                # Define the window: series_arr[i - period + 1] to series_arr[i]
                start_index = i - period + 1
                window = series_arr[start_index:i + 1]
                
                # Find the index of the extreme value within the window (0 to period-1)
                if is_high:
                    extreme_index_in_window = np.argmax(window)
                else:
                    extreme_index_in_window = np.argmin(window)
                    
                # periods_since calculates how many periods have passed since the extreme.
                # A value of 0 means the extreme was today (index period-1).
                # A value of period-1 means the extreme was at the start of the window (index 0).
                periods_since = (period - 1) - extreme_index_in_window
                
                # Aroon value: ((period - periods_since) / period) * 100
                result[i] = ((period - periods_since) / period) * 100
                
            return result

        # --- Main Accelerated Function ---
        def calculate_aroon_accelerated(df: pd.DataFrame, period: int = 25) -> pd.DataFrame:
            """
            Calculates Aroon Up, Aroon Down, and the Aroon Oscillator.
            """
            
            # 1. Prepare data arrays
            high_arr = df['high'].values
            low_arr = df['low'].values

            # 2. Calculate Aroon Up (Accelerated)
            aroon_up_arr = numba_aroon(high_arr, period, True)
            aroon_up = pd.Series(aroon_up_arr, index=df.index)

            # 3. Calculate Aroon Down (Accelerated)
            aroon_down_arr = numba_aroon(low_arr, period, False)
            aroon_down = pd.Series(aroon_down_arr, index=df.index)

            # 4. Calculate Aroon Oscillator (Vectorized)
            aroon_osc = aroon_up - aroon_down
            
            # Return results in a new DataFrame
            results = pd.DataFrame({
                f'aroon_up_{period}': aroon_up,
                f'aroon_down_{period}': aroon_down,
                f'aroon_osc_{period}': aroon_osc
            }, index=df.index)
            
            return aroon_up, aroon_down, aroon_osc

        cols['aroon_up'], cols['aroon_down'], cols['aroon_osc'] = calculate_aroon_accelerated(df, 25)
        
        # ============ ICHIMOKU (Vectorized) ============
        period9_high = df['high'].rolling(9).max()
        period9_low = df['low'].rolling(9).min()
        cols['ichimoku_conversion'] = (period9_high + period9_low) / 2
        
        period26_high = df['high'].rolling(26).max()
        period26_low = df['low'].rolling(26).min()
        cols['ichimoku_base'] = (period26_high + period26_low) / 2
        
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}
        
        cols['ichimoku_a'] = ((df['ichimoku_conversion'] + df['ichimoku_base']) / 2).shift(26)
        
        period52_high = df['high'].rolling(52).max()
        period52_low = df['low'].rolling(52).min()
        cols['ichimoku_b'] = ((period52_high + period52_low) / 2).shift(26)
        
        # The lagging span (Chikou) is today's close plotted 26 bars AGO.
        cols['ichimoku_lagging'] = df['close'].shift(26)
        
        # ============ PARABOLIC SAR (Numba Accelerated) ============
        @njit(cache=True)
        def numba_psar(
            high_arr: np.ndarray, 
            low_arr: np.ndarray, 
            close_arr: np.ndarray, 
            af_start: float = 0.02, 
            af_inc: float = 0.02, 
            af_max: float = 0.2
        ) -> np.ndarray:
            """
            Calculates the Parabolic SAR (Stop and Reverse) indicator using Numba.
            
            The PSAR calculation is highly iterative and includes:
            1. Tracking the trend direction (bull/bear).
            2. Tracking the Extreme Point (EP), the highest high in an uptrend or lowest low in a downtrend.
            3. Updating the Acceleration Factor (AF) when a new EP is reached.
            4. Enforcing 'jump' rules (PSAR cannot cross over the previous two days' lows/highs).
            """
            n = len(close_arr)
            if n == 0:
                return np.empty(0, dtype=np.float64)
            
            psar = np.zeros(n, dtype=np.float64)
            
            # Initialize variables
            # The first PSAR point is often the previous bar's close, but to start the logic 
            # we initialize the first three points as the close prices (or the EP).
            psar[0] = close_arr[0]
            
            # Initial state assumption (PSAR is trend-following, so we need a prior trend)
            bull = True  # Assuming initial uptrend
            af = af_start
            
            # EP is the highest high in a bull trend, or the lowest low in a bear trend.
            # We initialize HP and LP for the first day.
            hp = high_arr[0]  # Highest Price (in current bull trend)
            lp = low_arr[0]   # Lowest Price (in current bear trend)
            
            # Since PSAR relies on previous day values, we process the loop from i=1
            for i in range(1, n):
                psar_prev = psar[i - 1]
                
                # --- 1. Calculate the next PSAR value based on the current trend (bull/bear) ---
                if bull:
                    # Bullish trend: PSAR moves up towards the high side
                    psar[i] = psar_prev + af * (hp - psar_prev)
                    
                    # Check for reversal to Bearish
                    if low_arr[i] < psar[i]:
                        # Reversal: Current low crosses PSAR -> switch to Bear
                        bull = False
                        
                        # New PSAR value after reversal is the previous EP (HP)
                        psar[i] = hp
                        
                        # Reset AF and initialize new LP
                        af = af_start
                        lp = low_arr[i]
                    
                    # If still bull, update EP and AF
                    else:
                        if high_arr[i] > hp:
                            # New highest high (New EP)
                            hp = high_arr[i]
                            # Increase AF, maxing out at af_max
                            af = min(af + af_inc, af_max)
                
                else: # Bear trend
                    # Bearish trend: PSAR moves down towards the low side
                    psar[i] = psar_prev + af * (lp - psar_prev)
                    
                    # Check for reversal to Bullish
                    if high_arr[i] > psar[i]:
                        # Reversal: Current high crosses PSAR -> switch to Bull
                        bull = True
                        
                        # New PSAR value after reversal is the previous EP (LP)
                        psar[i] = lp
                        
                        # Reset AF and initialize new HP
                        af = af_start
                        hp = high_arr[i]
                    
                    # If still bear, update EP and AF
                    else:
                        if low_arr[i] < lp:
                            # New lowest low (New EP)
                            lp = low_arr[i]
                            # Increase AF, maxing out at af_max
                            af = min(af + af_inc, af_max)
                            
                # --- 2. Enforce the 'Jump' Rule (PSAR cannot penetrate recent price extremes) ---
                # PSAR must not be higher than the low of the two previous bars in a bull trend
                if bull:
                    if i >= 1 and low_arr[i-1] < psar[i]:
                        psar[i] = low_arr[i-1]
                    if i >= 2 and low_arr[i-2] < psar[i]:
                        psar[i] = min(psar[i], low_arr[i-2]) # Use min in case i-1 was already used
                
                # PSAR must not be lower than the high of the two previous bars in a bear trend
                else: # Bear trend
                    if i >= 1 and high_arr[i-1] > psar[i]:
                        psar[i] = high_arr[i-1]
                    if i >= 2 and high_arr[i-2] > psar[i]:
                        psar[i] = max(psar[i], high_arr[i-2]) # Use max in case i-1 was already used
                        
            return psar

        # --- Main Function for Demonstration ---
        def calculate_psar_accelerated(df: pd.DataFrame, af_start=0.02, af_inc=0.02, af_max=0.2) -> pd.Series:
            """
            Calculates the Parabolic SAR and returns it as a Pandas Series.
            """
            
            # 1. Prepare data arrays (Numba requires NumPy arrays)
            high_arr = df['high'].values
            low_arr = df['low'].values
            close_arr = df['close'].values

            # 2. Calculate PSAR using Numba
            psar_arr = numba_psar(high_arr, low_arr, close_arr, af_start, af_inc, af_max)

            # 3. Convert result back to Pandas Series with original index
            psar_series = pd.Series(psar_arr, index=df.index, name='PSAR')
            
            return psar_series

        cols['psar'] = calculate_psar_accelerated(df, 0.02, 0.02, 0.2)

        # ============ TRIX ============
        ema1 = self.ema(df['close'], 15)
        ema2 = self.ema(ema1, 15)
        ema3 = self.ema(ema2, 15)
        cols['trix'] = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 100
        
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}
        cols['trix_signal'] = self.sma(df['trix'], 9)
        
        # ============ MASS INDEX ============
        hl_range = df['high'] - df['low']
        ema9 = self.ema(hl_range, 9)
        ema9_ema9 = self.ema(ema9, 9)
        mass_ratio = ema9 / ema9_ema9
        cols['mass_index'] = mass_ratio.rolling(25).sum()
        
        # ============ DPO ============
        shift_val = int(20 / 2 + 1)
        sma_20 = df['close'].rolling(20).mean()
        cols['dpo'] = df['close'].shift(shift_val) - sma_20
        
        # ============ KST ============
        roc1 = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
        roc2 = ((df['close'] - df['close'].shift(15)) / df['close'].shift(15)) * 100
        roc3 = ((df['close'] - df['close'].shift(20)) / df['close'].shift(20)) * 100
        roc4 = ((df['close'] - df['close'].shift(30)) / df['close'].shift(30)) * 100
        cols['kst'] = (self.sma(roc1, 10) * 1 + self.sma(roc2, 10) * 2 + 
                       self.sma(roc3, 10) * 3 + self.sma(roc4, 15) * 4)
        
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}
        cols['kst_signal'] = self.sma(df['kst'], 9)
        
        # ============ RSI (Vectorized) ============
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain_14 = self.rma(gain, 14)
        avg_loss_14 = self.rma(loss, 14)
        rs_14 = avg_gain_14 / avg_loss_14
        cols['rsi'] = 100 - (100 / (1 + rs_14))
        
        avg_gain_30 = self.rma(gain, 30)
        avg_loss_30 = self.rma(loss, 30)
        rs_30 = avg_gain_30 / avg_loss_30
        cols['rsi_30'] = 100 - (100 / (1 + rs_30))
        
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}
        
        # ============ CONNORS RSI (Numba Accelerated) ============
        @njit(cache=True)
        def numba_calculate_streak(close_prices: np.ndarray) -> np.ndarray:
            """
            Calculates the consecutive up/down streak array using Numba.
            This replaces the slow explicit Python loop in the original code.
            """
            n = len(close_prices)
            streak = np.zeros(n, dtype=np.float64)

            # Start loop from 1 as streak depends on the previous day's close
            for i in range(1, n):
                prev_streak = streak[i - 1]
                
                # Up day (Close[i] > Close[i-1])
                if close_prices[i] > close_prices[i - 1]:
                    if prev_streak > 0:
                        # Extend positive streak
                        streak[i] = prev_streak + 1
                    else:
                        # Start new positive streak
                        streak[i] = 1
                
                # Down day (Close[i] < Close[i-1])
                elif close_prices[i] < close_prices[i - 1]:
                    if prev_streak < 0:
                        # Extend negative streak
                        streak[i] = prev_streak - 1
                    else:
                        # Start new negative streak
                        streak[i] = -1
                
                # No change (Close[i] == Close[i-1]), streak remains 0 (default in array)

            return streak

        @njit(cache=True)
        def numba_rma(data: np.ndarray, period: int) -> np.ndarray:
            """
            Calculates the Relative Moving Average (RMA) using Numba (Wilder's Smoothing).
            
            RMA_t = (RMA_{t-1} * (period - 1) + X_t) / period
            """
            n = len(data)
            rma = np.empty(n, dtype=np.float64)
            alpha = 1.0 / period
            
            # Initialize the first (period-1) values to NaN, and start smoothing 
            # based on the average of the first 'period' values.

            # 1. Simple average for the first 'period' values
            if n > 0:
                rma[0] = data[0]
                
            # 2. Apply smoothing formula
            for i in range(1, n):
                # RMA_t = RMA_{t-1} + alpha * (X_t - RMA_{t-1})
                rma[i] = rma[i-1] + alpha * (data[i] - rma[i-1])
                
            return rma


        # --- Main Accelerated Function (Hybrid Pandas/Numba) ---

        def calculate_connors_rsi_accelerated(df: pd.DataFrame, rsi_period: int = 3, rma_period: int = 2, pct_rank_period: int = 100) -> pd.Series:
            """
            Calculates the Connors RSI components using Numba for the iterative parts.

            Assumes the input DataFrame 'df' has 'close' and a pre-calculated 'rsi' column.
            The 'rsi' calculation itself (the original 'rsi_component') is assumed to be
            calculated externally, as it's typically a standard RSI(3).
            """
            
            # 1. RSI Component (Assumed pre-calculated standard RSI(3))
            # Note: In a real implementation, you would calculate this RSI(3) as well.
            # We use the existing 'rsi' column as per the original code.
            rsi_component = df['rsi'].copy()

            # 2. Streak Calculation (Numba Accelerated)
            close_arr = df['close'].values
            streak_arr = numba_calculate_streak(close_arr)
            streak = pd.Series(streak_arr, index=df.index)

            # 3. Streak RSI Components (Vectorized Pandas/NumPy + Numba RMA)
            
            # Calculate delta (vectorized, fast)
            streak_delta = streak.diff().fillna(0).values # Fill NaN created by diff()

            # Calculate gains and losses (vectorized, fast)
            streak_gain_arr = np.where(streak_delta > 0, streak_delta, 0)
            streak_loss_arr = np.where(streak_delta < 0, -streak_delta, 0)

            # Calculate average gain/loss using Numba RMA
            streak_avg_gain_arr = numba_rma(streak_gain_arr, rma_period)
            streak_avg_loss_arr = numba_rma(streak_loss_arr, rma_period)
            
            # Convert back to Series for vectorized division (fast)
            streak_avg_gain = pd.Series(streak_avg_gain_arr, index=df.index)
            streak_avg_loss = pd.Series(streak_avg_loss_arr, index=df.index)

            # Calculate RS and RSI (vectorized, fast)
            streak_rs = streak_avg_gain / streak_avg_loss
            rsi_streak = 100 - (100 / (1 + streak_rs))
            
            # 4. Rate of Change and Percent Rank Component (Vectorized Pandas)
            
            # Calculate ROC (vectorized, fast)
            roc = df['close'].pct_change(1) * 100
            
            # Calculate Percent Rank (This is a complex rolling operation, 
            # but the Pandas implementation is generally acceptable for this part.)
            pct_rank = roc.rolling(pct_rank_period, min_periods=1).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else np.nan, 
                raw=False
            )
            
            # 5. Final Calculation
            connors_rsi = (rsi_component + rsi_streak + pct_rank) / 3
            
            return connors_rsi
        
        cols["connors_rsi"] = calculate_connors_rsi_accelerated(df, 3, 2, 100)
    
        # ============ STOCHASTIC (Vectorized) ============
        low_min = df['low'].rolling(14).min()
        high_max = df['high'].rolling(14).max()
        cols['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}
        cols['stoch_d'] = self.sma(df['stoch_k'], 3)
        
        rsi_min = df['rsi'].rolling(14).min()
        rsi_max = df['rsi'].rolling(14).max()
        cols['stoch_rsi'] = (df['rsi'] - rsi_min) / (rsi_max - rsi_min)
        
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}
        cols['stoch_rsi_k'] = self.sma(df['stoch_rsi'], 3) * 100
        
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}
        cols['stoch_rsi_d'] = self.sma(df['stoch_rsi_k'], 3)
        
        # ============ SMI (Vectorized) ============
        ll = df['low'].rolling(14).min()
        hh = df['high'].rolling(14).max()
        diff = hh - ll
        rdiff = df['close'] - (hh + ll) / 2
        
        avgrel = self.ema(self.ema(rdiff, 3), 3)
        avgdiff = self.ema(self.ema(diff, 3), 3)
        cols['smi'] = np.where(avgdiff != 0, (avgrel / (avgdiff / 2)) * 100, 0)
        cols['smi_signal'] = self.ema(cols['smi'], 3)
        cols['smi_ergodic'] = cols['smi']
        cols['smi_ergodic_signal'] = self.ema(cols['smi_ergodic'], 5)
        
        # ============ WILLIAMS %R ============
        highest_high = df['high'].rolling(14).max()
        lowest_low = df['low'].rolling(14).min()
        cols['williams_r'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
        
        # ============ ROC ============
        cols['roc'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
        cols['roc_20'] = ((df['close'] - df['close'].shift(20)) / df['close'].shift(20)) * 100
        
        # ============ AWESOME OSCILLATOR ============
        median_price = (df['high'] + df['low']) / 2
        cols['ao'] = self.sma(median_price, 5) - self.sma(median_price, 34)
        
        # ============ KAMA (Numba Accelerated) ============
        @jit(nopython=True)
        def kama_numba(prices, er_period=10, fast=2, slow=30):
            """Numba-optimized KAMA calculation"""
            n = len(prices)
            kama = np.zeros(n)
            kama[:er_period] = prices[0]
            
            sc_fast = 2.0 / (fast + 1)
            sc_slow = 2.0 / (slow + 1)
            
            for i in range(er_period, n):
                change = abs(prices[i] - prices[i - er_period])
                
                # Calculate volatility
                volatility = 0.0
                for j in range(i - er_period + 1, i + 1):
                    volatility += abs(prices[j] - prices[j-1])
                
                # Efficiency ratio
                er = change / volatility if volatility != 0 else 0
                
                # Smoothing constant
                sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2
                
                # KAMA
                kama[i] = kama[i-1] + sc * (prices[i] - kama[i-1])
            
            return kama
        cols['kama'] = kama_numba(df['close'].values, 10, 2, 30)

        # ============ PPO ============
        ema_fast = self.ema(df['close'], 12)
        ema_slow = self.ema(df['close'], 26)
        ppo_line = (ema_fast - ema_slow) / ema_slow * 100
        ppo_signal = self.ema(ppo_line, 9)
        cols['ppo'] = ppo_line
        cols['ppo_signal'] = ppo_signal
        cols['ppo_hist'] = ppo_line - ppo_signal
        cols['ppo_mean'] = cols['ppo'].rolling(window=50, min_periods=1).mean()
        
        # ============ ULTIMATE OSCILLATOR ============
        bp = df['close'] - pd.concat([df['low'], df['close'].shift(1)], axis=1).min(axis=1)
        avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
        avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
        avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
        cols['ultimate_osc'] = 100 * ((4 * avg7) + (2 * avg14) + avg28) / 7
        
        # ============ TSI ============
        m = df['close'].diff()
        ema1 = self.ema(m, 25)
        ema2 = self.ema(ema1, 13)
        ema1_abs = self.ema(m.abs(), 25)
        ema2_abs = self.ema(ema1_abs, 13)
        cols['tsi'] = 100 * (ema2 / ema2_abs)
        
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}
        
        # ============ BOLLINGER BANDS ============
        cols['bb_middle'] = self.sma(df['close'], 20)
        bb_std = self.std(df['close'], 20)
        cols['bb_upper'] = cols['bb_middle'] + (bb_std * 2)
        cols['bb_lower'] = cols['bb_middle'] - (bb_std * 2)
        cols['bb_width'] = (cols['bb_upper'] - cols['bb_lower']) / cols['bb_middle']
        cols['bb_pct'] = (df['close'] - cols['bb_lower']) / (cols['bb_upper'] - cols['bb_lower'])
        cols['bb_bandwidth'] = cols['bb_width']
        cols['bb_mean'] = cols['bb_middle'].rolling(window=50, min_periods=1).mean().fillna(method='bfill').fillna(0)
        
        # ============ ATR ============
        cols['atr'] = self.rma(tr, 14)
        # Attach atr early so subsequent columns can reference it
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}

        # ATR normalized to price level (ratio) — used for volatility bucketing
        cols['atr_ratio'] = df['atr'] / df['close'].replace(0, np.nan)
        # Rolling mean of ATR for stability / centroid calculations
        cols['atr_mean'] = df['atr'].rolling(window=100, min_periods=1).mean()
        cols['atr_percent'] = (df['atr'] / df['close']) * 100 
        
        # Rolling mean of recent lows (used by regime SQL / low_mean references)
        cols['low_mean'] = df['low'].rolling(window=50, min_periods=1).mean().fillna(method='bfill').fillna(0)

        
        # ============ KELTNER CHANNELS ============
        cols['kc_middle'] = self.ema(df['close'], 20)
        cols['kc_upper'] = cols['kc_middle'] + (2 * df['atr'])
        cols['kc_lower'] = cols['kc_middle'] - (2 * df['atr'])
        cols['kc_width'] = (cols['kc_upper'] - cols['kc_lower']) / cols['kc_middle']
        cols['kc_pct'] = (df['close'] - cols['kc_lower']) / (cols['kc_upper'] - cols['kc_lower'])
        
        # ============ DONCHIAN CHANNELS ============
        cols['dc_upper'] = df['high'].rolling(20).max()
        cols['dc_lower'] = df['low'].rolling(20).min()
        cols['dc_middle'] = (cols['dc_upper'] + cols['dc_lower']) / 2
        cols['dc_width'] = (cols['dc_upper'] - cols['dc_lower']) / cols['dc_middle']
        cols['dc_pct'] = (df['close'] - cols['dc_lower']) / (cols['dc_upper'] - cols['dc_lower'])
        
        # ============ ULCER INDEX ============
        rolling_max = df['close'].rolling(14, min_periods=1).max()
        drawdown = (df['close'] - rolling_max) / rolling_max * 100
        squared_dd = drawdown.pow(2)
        cols['ulcer_index'] = np.sqrt(squared_dd.rolling(14, min_periods=1).mean())
        
        # ============ OBV (Vectorized) ============
        price_diff = df['close'].diff()
        volume_direction = np.where(price_diff > 0, df['volume'],
                                  np.where(price_diff < 0, -df['volume'], 0))
        # .cumsum() is the key. On a 1000-bar chunk, this is a stable approximation.
        volume_direction_series = pd.Series(volume_direction, index=df.index)
        cols['obv'] = volume_direction_series.cumsum().fillna(0)
        cols['obv_mean'] = cols['obv'].rolling(window=50, min_periods=1).mean()
        
        # ============ CMF ============
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        clv = clv.fillna(0)
        mf_volume = clv * df['volume']
        cols['cmf'] = mf_volume.rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # ============ FORCE INDEX ============
        cols['force_index'] = df['close'].diff() * df['volume']
        cols['force_index_ema'] = self.ema(cols['force_index'], 13)
        
        # ============ EASE OF MOVEMENT ============
        distance = ((df['high'] + df['low']) / 2) - ((df['high'].shift(1) + df['low'].shift(1)) / 2)
        box_ratio = (df['volume'] / 100000000) / (df['high'] - df['low'])
        cols['eom'] = distance / box_ratio
        cols['eom_sma'] = self.sma(cols['eom'], 14)
        
        # ============ VPT ============
        pct_change = df['close'].pct_change()
        cols['vpt'] = (pct_change * df['volume']).cumsum().fillna(0)
        
        # ============ NVI (Vectorized) ============
        volume_down = df['volume'].diff() < 0
        pct_change_nvi = df['close'].pct_change().fillna(0)
        nvi_changes = np.where(volume_down, 1 + pct_change_nvi, 1.0)
        nvi_changes = pd.Series(nvi_changes, index=df.index)
        nvi_changes.iloc[0] = 1.0 # Start at 1
        cols['nvi'] = nvi_changes.cumprod() * 1000 # Scale to 1000
        
        # ============ A/D ============
        clv_ad = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        clv_ad = clv_ad.fillna(0)
        cols['ad'] = (clv_ad * df['volume']).cumsum().fillna(0)
        
        # ============ MFI ============
        raw_money_flow = tp * df['volume']
        positive_flow = np.where(tp > tp.shift(1), raw_money_flow, 0)
        negative_flow = np.where(tp < tp.shift(1), raw_money_flow, 0)
        positive_flow = pd.Series(positive_flow, index=df.index)
        negative_flow = pd.Series(negative_flow, index=df.index)
        positive_mf = positive_flow.rolling(14).sum()
        negative_mf = negative_flow.rolling(14).sum()
        mfi_ratio = positive_mf / negative_mf
        cols['mfi'] = 100 - (100 / (1 + mfi_ratio))
        
        # ============ VWAP ============
        typical_price_vwap = (df['high'] + df['low'] + df['close']) / 3
        cum_vol_price = (typical_price_vwap * df['volume']).cumsum()
        cum_volume = df['volume'].cumsum()
        cols['vwap'] = cum_vol_price / cum_volume
        
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}
        
        # ============ CCI ============
        tp_sma = self.sma(tp, 20)
        mean_dev = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        cols['cci'] = (tp - tp_sma) / (0.015 * mean_dev)
        cols['woodies_cci'] = cols['cci']
        cols['woodies_cci_signal'] = self.ema(cols['cci'], 6)
        
        # ============ MOMENTUM ============
        cols['momentum'] = df['close'] - df['close'].shift(10)
        cols['momentum_pct'] = (df['close'] / df['close'].shift(10) - 1) * 100
        
        # ============ CHAIKIN OSCILLATOR ============
        cols['chaikin_osc'] = self.ema(df['ad'], 3) - self.ema(df['ad'], 10)
        
        # ============ VORTEX INDICATOR ============
        vmp = np.abs(df['high'] - df['low'].shift(1))
        vmm = np.abs(df['low'] - df['high'].shift(1))
        vmp = pd.Series(vmp, index=df.index)
        vmm = pd.Series(vmm, index=df.index)
        vmp_sum = vmp.rolling(14).sum()
        vmm_sum = vmm.rolling(14).sum()
        tr_sum = tr.rolling(14).sum()
        cols['vortex_pos'] = vmp_sum / tr_sum
        cols['vortex_neg'] = vmm_sum / tr_sum
        cols['vortex_diff'] = cols['vortex_pos'] - cols['vortex_neg']
        
        # ============ BOP ============
        cols['bop'] = (df['close'] - df['open']) / (df['high'] - df['low'])
        cols['bop'] = cols['bop'].replace([np.inf, -np.inf], 0)
        
        # ============ BULL/BEAR POWER ============
        ema_13 = self.ema(df['close'], 13)
        cols['bull_power'] = df['high'] - ema_13
        cols['bear_power'] = df['low'] - ema_13
        
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}
        
        # ============ CHOPPINESS INDEX ============
        atr_sum = df['atr'].rolling(14).sum()
        high_low_range = df['high'].rolling(14).max() - df['low'].rolling(14).min()
        cols['choppiness'] = 100 * np.log10(atr_sum / high_low_range) / np.log10(14)
        
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}
        
        # ============ CANDLESTICK PATTERNS (Optimized) ============
        body = (df['close'] - df['open']).abs()
        range_hl = df['high'] - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        bull = df['close'] > df['open']
        bear = df['close'] < df['open']
        small_body = body <= 0.25 * range_hl
        cols['pattern_hammer'] = ((lower_shadow > 2 * body) & (upper_shadow <= body)).astype(int)
        cols['pattern_inverted_hammer'] = ((upper_shadow > 2 * body) & (lower_shadow <= body) & bull).astype(int)
        cols['pattern_hanging_man'] = ((lower_shadow > 2 * body) & (upper_shadow <= body) & bear).astype(int)
        cols['pattern_shooting_star'] = ((upper_shadow > 2 * body) & (lower_shadow <= body)).astype(int)
        cols['pattern_doji'] = (body <= 0.1 * range_hl).astype(int)
        cols['pattern_long_legged_doji'] = ((body <= 0.1 * range_hl) & (upper_shadow >= 0.4 * range_hl) & (lower_shadow >= 0.4 * range_hl)).astype(int)
        cols['pattern_dragonfly_doji'] = ((body <= 0.1 * range_hl) & (lower_shadow >= 0.6 * range_hl) & (upper_shadow <= 0.1 * range_hl)).astype(int)
        cols['pattern_gravestone_doji'] = ((body <= 0.1 * range_hl) & (upper_shadow >= 0.6 * range_hl) & (lower_shadow <= 0.1 * range_hl)).astype(int)
        cols['pattern_marubozu_bull'] = (bull & (upper_shadow <= 0.05 * range_hl) & (lower_shadow <= 0.05 * range_hl)).astype(int)
        cols['pattern_marubozu_bear'] = (bear & (upper_shadow <= 0.05 * range_hl) & (lower_shadow <= 0.05 * range_hl)).astype(int)
        o1, c1 = df['open'].shift(1), df['close'].shift(1)
        bull_1 = c1 > o1
        bear_1 = c1 < o1
        cols['pattern_bullish_engulfing'] = (bear_1 & bull & (df['close'] >= o1) & (df['open'] <= c1)).astype(int)
        cols['pattern_bearish_engulfing'] = (bull_1 & bear & (df['open'] >= c1) & (df['close'] <= o1)).astype(int)
        cols['pattern_morning_star'] = (bear.shift(2) & small_body.shift(1) & bull & (df['close'] >= (df['open'].shift(2) + df['close'].shift(2)) / 2)).astype(int)
        cols['pattern_evening_star'] = (bull.shift(2) & small_body.shift(1) & bear & (df['close'] <= (df['open'].shift(2) + df['close'].shift(2)) / 2)).astype(int)
        cols['pattern_three_white_soldiers'] = (bull & bull.shift(1) & bull.shift(2) & (df['close'] > df['close'].shift(1)) & (df['close'].shift(1) > df['close'].shift(2))).astype(int)
        cols['pattern_three_black_crows'] = (bear & bear.shift(1) & bear.shift(2) & (df['close'] < df['close'].shift(1)) & (df['close'].shift(1) < df['close'].shift(2))).astype(int)
        pattern_names = [
            'pattern_belt_hold_bull', 'pattern_belt_hold_bear', 'pattern_tweezer_top',
            'pattern_tweezer_bottom', 'pattern_bullish_harami', 'pattern_bearish_harami',
            'pattern_piercing_line', 'pattern_dark_cloud_cover', 'pattern_on_neck',
            'pattern_in_neck', 'pattern_thrusting', 'pattern_morning_doji_star',
            'pattern_evening_doji_star', 'pattern_three_inside_up', 'pattern_three_inside_down',
            'pattern_three_outside_up', 'pattern_three_outside_down', 'pattern_rising_three_methods',
            'pattern_falling_three_methods', 'pattern_upside_tasuki_gap', 'pattern_downside_tasuki_gap',
            'pattern_abandoned_baby_bull', 'pattern_abandoned_baby_bear', 'pattern_stick_sandwich',
            'pattern_matching_low', 'pattern_matching_high', 'pattern_ladder_bottom',
            'pattern_counterattack_bull', 'pattern_counterattack_bear', 'pattern_breakaway_bull',
            'pattern_breakaway_bear', 'pattern_separating_lines_bull', 'pattern_separating_lines_bear',
            'pattern_side_by_side_white_lines', 'pattern_homing_pigeon', 'pattern_doji_star_bull',
            'pattern_doji_star_bear', 'pattern_rickshaw_man', 'pattern_kicking_bull',
            'pattern_kicking_bear', 'pattern_kicking_by_length_bull', 'pattern_kicking_by_length_bear'
        ]
        for pattern in pattern_names:
            if pattern not in cols:
                cols[pattern] = 0
        
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}
        
        # ============ CHART PATTERNS ============
        for name, func in CHART_PATTERN_FUNCS.items():
            try:
                df[name] = func(df).fillna(False).astype(int)
            except Exception:
                df[name] = 0
        
        # ============ PULLBACK INDICATORS (Optimized) ============
        
        # *** CRITICAL FIX: DATA LEAKAGE ***
        # Swing points loop is still slow, but now it is *correct*.
        # The signal is shifted 5 bars to prevent future-peeking.

        #lookback = get_optimal_lookback(self.timeframe) # This line needs to extract timeframe from pair_tf string

        lookback = 2

        swing_highs = pd.Series(False, index=df.index)
        swing_lows = pd.Series(False, index=df.index)
        
        for i in range(lookback, len(df) - lookback):
            is_swing_high = all(df['high'].iloc[i] > df['high'].iloc[i-j] and 
                               df['high'].iloc[i] > df['high'].iloc[i+j] 
                               for j in range(1, lookback + 1))
            is_swing_low = all(df['low'].iloc[i] < df['low'].iloc[i-j] and 
                              df['low'].iloc[i] < df['low'].iloc[i+j] 
                              for j in range(1, lookback + 1))
            swing_highs.iloc[i] = is_swing_high
            swing_lows.iloc[i] = is_swing_low
        
        # Apply the shift. The swing high/low is only *confirmed*
        # 'lookback' bars after it happens.
        cols['swing_high'] = swing_highs.shift(lookback).fillna(False).astype(int)
        cols['swing_low'] = swing_lows.shift(lookback).fillna(False).astype(int)
        
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}
        
        # Last swing values (Vectorized)
        # Find the *price* at the bar where swing_high/low is true
        # Then forward-fill that price.
        cols['last_swing_high'] = df['high'].where(df['swing_high'] == 1).ffill()
        cols['last_swing_low'] = df['low'].where(df['swing_low'] == 1).ffill()
        
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}
        
        # Pullback percentages
        cols['pullback_from_high_pct'] = ((df['close'] - df['last_swing_high']) / df['last_swing_high'] * 100)
        cols['pullback_from_low_pct'] = ((df['close'] - df['last_swing_low']) / df['last_swing_low'] * 100)
        
        # Fibonacci levels
        diff = df['last_swing_high'] - df['last_swing_low']
        cols['fib_23_6'] = df['last_swing_high'] - (diff * 0.236)
        cols['fib_38_2'] = df['last_swing_high'] - (diff * 0.382)
        cols['fib_50_0'] = df['last_swing_high'] - (diff * 0.500)
        cols['fib_61_8'] = df['last_swing_high'] - (diff * 0.618)
        cols['fib_78_6'] = df['last_swing_high'] - (diff * 0.786)
        
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}
        
        # Near fib levels (vectorized)
        tolerance = 0.005
        cols['near_fib_236'] = (np.abs(df['close'] - df['fib_23_6']) / df['close'] < tolerance).astype(int)
        cols['near_fib_382'] = (np.abs(df['close'] - df['fib_38_2']) / df['close'] < tolerance).astype(int)
        cols['near_fib_500'] = (np.abs(df['close'] - df['fib_50_0']) / df['close'] < tolerance).astype(int)
        cols['near_fib_618'] = (np.abs(df['close'] - df['fib_61_8']) / df['close'] < tolerance).astype(int)
        cols['near_fib_786'] = (np.abs(df['close'] - df['fib_78_6']) / df['close'] < tolerance).astype(int)
        
        # Trend direction
        cols['trend_short'] = np.where(df['close'] > df['ema_9'], 1, -1)
        cols['trend_medium'] = np.where(df['close'] > df['ema_50'], 1, -1)
        cols['trend_long'] = np.where(df['close'] > df['ema_200'], 1, -1)



        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}
        
        # Pullback depth classification (Numerical encoding for DL)
        # Mapping: 0=none, 1=shallow, 2=moderate, 3=deep
        pb_pct = df['pullback_from_high_pct']
        depth = pd.Series(0, index=df.index, dtype=np.int8)  # Default: 0 (none)
        depth[pb_pct.between(-5, -1)] = 1   # shallow
        depth[pb_pct.between(-10, -5)] = 2  # moderate
        depth[pb_pct < -10] = 3             # deep
        cols['pullback_depth'] = depth
        
        # ============================================================================
        # VECTORIZED HL and LH Detection (Faster, but more complex)
        # ============================================================================

        def detect_higher_lows_vectorized(df):
            """
            Vectorized version using rolling window logic.
            More efficient for large datasets.
            """
            pattern = pd.Series(0, index=df.index)
            
            # Get swing low prices and their indices
            swing_low_mask = df['swing_low'] == 1
            swing_low_prices = df['low'].where(swing_low_mask)
            
            # For each bar, check if last 3 swing lows are ascending
            for i in range(20, len(df)):
                window = df.iloc[max(0, i-50):i+1]  # Look back 50 bars max
                recent_lows = window[window['swing_low'] == 1]['low'].tail(3)
                
                if len(recent_lows) == 3:
                    values = recent_lows.values
                    # Check if ascending
                    if values[1] > values[0] and values[2] > values[1]:
                        if df.iloc[i]['trend_medium'] == 1:
                            pattern.iloc[i] = 1
            
            return pattern

        def detect_lower_highs_vectorized(df):
            """
            Vectorized version for lower highs pattern.
            """
            pattern = pd.Series(0, index=df.index)
            
            # Get swing high prices and their indices
            swing_high_mask = df['swing_high'] == 1
            
            # For each bar, check if last 3 swing highs are descending
            for i in range(20, len(df)):
                window = df.iloc[max(0, i-50):i+1]  # Look back 50 bars max
                recent_highs = window[window['swing_high'] == 1]['high'].tail(3)
                
                if len(recent_highs) == 3:
                    values = recent_highs.values
                    # Check if descending
                    if values[1] < values[0] and values[2] < values[1]:
                        if df.iloc[i]['trend_medium'] == -1:
                            pattern.iloc[i] = 1
            
            return pattern

        # Higher lows / lower highs patterns (still slow, recommend pandas-ta)
        cols['higher_lows_pattern'] = detect_higher_lows_vectorized(df)
        cols['lower_highs_pattern'] = detect_lower_highs_vectorized(df)
        
        # Pullback to MA
        cols['pullback_to_ema9'] = ((df['close'] - df['ema_9']).abs() / df['ema_9'] < 0.01).astype(int)
        cols['pullback_to_ema20'] = ((df['close'] - df['ema_26']).abs() / df['ema_26'] < 0.01).astype(int)
        cols['pullback_to_sma50'] = ((df['close'] - df['sma_50']).abs() / df['sma_50'] < 0.01).astype(int)
        
        # Healthy pullbacks
        cols['healthy_bull_pullback'] = ((df['trend_medium'] == 1) & (df['rsi'] > 40) & (df['rsi'] < 60) & (df['close'] < df['close'].shift(1))).astype(int)
        cols['healthy_bear_pullback'] = ((df['trend_medium'] == -1) & (df['rsi'] < 60) & (df['rsi'] > 40) & (df['close'] > df['close'].shift(1))).astype(int)
        
        # Volume analysis
        cols['volume_ma5'] = df['volume'].rolling(5).mean()
        
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}
        
        cols['volume_decreasing'] = (df['volume'] < df['volume_ma5']).astype(int)
        
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}
        
        # Bars since swing (Vectorized)
        row_numbers = pd.Series(np.arange(len(df)), index=df.index)
        swing_high_row = row_numbers.where(df['swing_high'] == 1).ffill().fillna(-9999)
        swing_low_row = row_numbers.where(df['swing_low'] == 1).ffill().fillna(-9999)
        
        cols['bars_since_swing_high'] = row_numbers - swing_high_row
        cols['bars_since_swing_low'] = row_numbers - swing_low_row
        
        # ABC patterns (simplified)  
        def detect_abc_pullback(df, window=20):
            """
            Detect ABC pullback pattern:
            A = Initial move
            B = Pullback
            C = Continuation
            """
            abc_bull = pd.Series(False, index=df.index)
            abc_bear = pd.Series(False, index=df.index)
            
            for i in range(window, len(df)):
                window_data = df.iloc[i-window:i+1]
                
                # Bullish ABC: Low -> High -> Higher Low
                highs = window_data['high']
                lows = window_data['low']
                
                point_a_idx = lows.idxmin()  # Lowest point
                point_a_pos = window_data.index.get_loc(point_a_idx)
                
                if point_a_pos < len(window_data) - 5:
                    after_a = window_data.iloc[point_a_pos:]
                    point_b_idx = after_a['high'].idxmax()  # Highest point after A
                    point_b_pos = after_a.index.get_loc(point_b_idx)
                    
                    if point_b_pos < len(after_a) - 2:
                        after_b = after_a.iloc[point_b_pos:]
                        point_c_low = after_b['low'].min()
                        point_a_low = window_data.loc[point_a_idx, 'low']
                        
                        # C should be higher than A (higher low)
                        if point_c_low > point_a_low * 1.005:  # 0.5% higher
                            abc_bull.iloc[i] = True
                
                # Bearish ABC: High -> Low -> Lower High
                point_a_idx = highs.idxmax()
                point_a_pos = window_data.index.get_loc(point_a_idx)
                
                if point_a_pos < len(window_data) - 5:
                    after_a = window_data.iloc[point_a_pos:]
                    point_b_idx = after_a['low'].idxmin()
                    point_b_pos = after_a.index.get_loc(point_b_idx)
                    
                    if point_b_pos < len(after_a) - 2:
                        after_b = after_a.iloc[point_b_pos:]
                        point_c_high = after_b['high'].max()
                        point_a_high = window_data.loc[point_a_idx, 'high']
                        
                        if point_c_high < point_a_high * 0.995:
                            abc_bear.iloc[i] = True
            
            return abc_bull, abc_bear

        abc_bull, abc_bear = detect_abc_pullback(df, window=20)
        cols['abc_pullback_bull'] = abc_bull.astype(int)
        cols['abc_pullback_bear'] = abc_bear.astype(int)
        
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}
        
        # Pullback quality score
        trend_strength = ((df['trend_short'] == df['trend_medium']) & (df['trend_medium'] == df['trend_long'])).astype(int) * 30
        ideal_depth = df['pullback_from_high_pct'].between(-12, -3).astype(int) * 30
        vol_decrease = df['volume_decreasing'] * 20
        rsi_healthy = ((df['rsi'] > 40) & (df['rsi'] < 60)).astype(int) * 20
        cols['pullback_quality_score'] = trend_strength + ideal_depth + vol_decrease + rsi_healthy
        
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}
        
        # SR confluence
        tolerance = 0.01
        confluence = pd.Series(0, index=df.index)
        confluence += ((df['close'] - df['ema_9']).abs() / df['close'] < tolerance).astype(int)
        confluence += ((df['close'] - df['ema_26']).abs() / df['close'] < tolerance).astype(int)
        confluence += ((df['close'] - df['sma_50']).abs() / df['close'] < tolerance).astype(int)
        confluence += ((df['close'] - df['last_swing_low']).abs() / df['close'] < tolerance).astype(int)
        confluence += df['near_fib_382']
        confluence += df['near_fib_500']
        confluence += df['near_fib_618']
        cols['sr_confluence_score'] = confluence
        
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}
        
        # Pullback completion signals
        cols['pullback_complete_bull'] = ((df['trend_medium'] == 1) & (df['pullback_quality_score'] >= 50) & (df['rsi'] < 50) & (df['sr_confluence_score'] >= 2) & (df['volume_decreasing'] == 1) & (df['close'] > df['open'])).astype(int)
        cols['pullback_complete_bear'] = ((df['trend_medium'] == -1) & (df['pullback_quality_score'] >= 50) & (df['rsi'] > 50) & (df['sr_confluence_score'] >= 2) & (df['volume_decreasing'] == 1) & (df['close'] < df['open'])).astype(int)
        
        # Failed pullbacks
        cols['failed_pullback_bull'] = ((df['trend_medium'] == 1) & (df['close'] < df['last_swing_low']) & (df['volume'] > df['volume_ma5'])).astype(int)
        cols['failed_pullback_bear'] = ((df['trend_medium'] == -1) & (df['close'] > df['last_swing_high']) & (df['volume'] > df['volume_ma5'])).astype(int)
        
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}
        
        # Measured move targets
        move_size = df['last_swing_high'] - df['last_swing_low']
        cols['measured_move_bull_target'] = df['last_swing_high'] + move_size
        cols['measured_move_bear_target'] = df['last_swing_low'] - move_size
        cols['distance_to_bull_target_pct'] = ((cols['measured_move_bull_target'] - df['close']) / df['close'] * 100)
        cols['distance_to_bear_target_pct'] = ((df['close'] - cols['measured_move_bear_target']) / df['close'] * 100)
        
        # Pullback stage (Numerical encoding for DL)
        # Mapping: 0=neutral, 1=impulse_up, 2=impulse_down, 3=pullback_early,
        #          4=pullback_mid, 5=pullback_late, 6=resumption_bull, 7=resumption_bear
        def classify_pullback_stage(df):
            """
            Classify current stage (numerical encoding): 
            - 0: neutral
            - 1: impulse_up - Strong upward move
            - 2: impulse_down - Strong downward move
            - 3: pullback_early - Just started pulling back
            - 4: pullback_mid - In middle of pullback
            - 5: pullback_late - Near completion
            - 6: resumption_bull - Bullish trend resuming
            - 7: resumption_bear - Bearish trend resuming
            """
            stage = pd.Series(0, index=df.index, dtype=np.int8)  # Default: 0 (neutral)
            
            for i in range(10, len(df)):
                # Check recent price action
                recent_close = df['close'].iloc[i-10:i+1]
                
                # Impulse: Strong move in one direction
                if df['trend_medium'].iloc[i] == 1:
                    if recent_close.iloc[-1] > recent_close.iloc[0] * 1.03:  # 3% up
                        stage.iloc[i] = 1  # impulse_up
                elif df['trend_medium'].iloc[i] == -1:
                    if recent_close.iloc[-1] < recent_close.iloc[0] * 0.97:  # 3% down
                        stage.iloc[i] = 2  # impulse_down
                
                # Pullback stages based on bars since swing and depth
                bars_since = df['bars_since_swing_high'].iloc[i]
                pullback_depth = abs(df['pullback_from_high_pct'].iloc[i])
                
                if df['trend_medium'].iloc[i] == 1 and df['close'].iloc[i] < df['close'].iloc[i-1]:
                    if bars_since <= 3:
                        stage.iloc[i] = 3  # pullback_early
                    elif bars_since <= 7 and pullback_depth < 8:
                        stage.iloc[i] = 4  # pullback_mid
                    elif pullback_depth >= 5:
                        stage.iloc[i] = 5  # pullback_late
                
                # Resumption: Pullback complete, trend resuming
                if df['pullback_complete_bull'].iloc[i] == 1:
                    stage.iloc[i] = 6  # resumption_bull
                elif df['pullback_complete_bear'].iloc[i] == 1:
                    stage.iloc[i] = 7  # resumption_bear
            
            return stage
        cols['pullback_stage'] = classify_pullback_stage(df)
        
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}

        # ============ REGIME ANALYSIS PRE-CALCULATIONS ============
        # These are features that analysis_advanced_regime.py
        # will read at each swing point.
        
        # Trend features
        cols['local_slope'] = (df['close'] - df['close'].shift(10)) / 10 / df['close']
        cols['return_gradient'] = df['close'].pct_change().diff().rolling(5).mean()
        cols['directional_persistence'] = (df['close'].pct_change() > 0).rolling(10).mean() - 0.5
        
        # Volatility features
        cols['atr_ratio'] = df['atr'] / df['close']
        cols['normalized_variance'] = df['close'].pct_change().rolling(20).std()
        
        # Volume features
        cols['volume_roc'] = (df['volume'] - df['volume'].shift(5)) / df['volume'].shift(5)
        cols['obv_change'] = df['obv'].diff(1)
        
        vol_mean_20 = df['volume'].rolling(20).mean()
        vol_std_20 = df['volume'].rolling(20).std()
        cols['volume_zscore'] = (df['volume'] - vol_mean_20) / vol_std_20
        
        # Price action anomaly features
        cols['gap_intensity'] = (df['open'] - df['close'].shift(1)).abs() / df['close'].shift(1)
        cols['extended_bar_ratio'] = (df['high'] - df['low']) / df['atr']
        cols['volume_spike'] = (df['volume'] > (vol_mean_20 * 2.0)).astype(int)

        # Candle context features
        bullish_patterns = [
            'pattern_hammer', 'pattern_bullish_engulfing', 'pattern_morning_star',
            'pattern_piercing_line', 'pattern_inverted_hammer'
        ]
        bearish_patterns = [
            'pattern_shooting_star', 'pattern_bearish_engulfing', 'pattern_evening_star',
            'pattern_dark_cloud_cover', 'pattern_hanging_man'
        ]
        doji_patterns = [
            'pattern_doji', 'pattern_dragonfly_doji', 'pattern_gravestone_doji',
            'pattern_long_legged_doji'
        ]

        # Calculate rolling sums for recent patterns
        # We check if column exists before summing
        cols['recent_bullish_patterns'] = 0
        for pattern in bullish_patterns:
            if pattern in df.columns:
                cols['recent_bullish_patterns'] += df[pattern].rolling(5).sum()
        
        cols['recent_bearish_patterns'] = 0
        for pattern in bearish_patterns:
            if pattern in df.columns:
                cols['recent_bearish_patterns'] += df[pattern].rolling(5).sum()
                
        cols['recent_doji'] = 0
        for pattern in doji_patterns:
            if pattern in df.columns:
                cols['recent_doji'] += df[pattern].rolling(5).sum()
        
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}

        # ============ LOCAL SLOPE (for regime/trend strength) ============
        # Compute a local slope for 'close' using a small rolling linear fit.
        # This provides the 'local_slope' column expected by the regime mapper.
        try:
            def _slope_func(x):
                # x is a numpy array with length >=1
                if len(x) < 2:
                    return 0.0
                # fit degree-1 polynomial and return slope
                return float(np.polyfit(np.arange(len(x)), x, 1)[0])

            slope_window = 5
            cols['local_slope'] = df['close'].rolling(window=slope_window, min_periods=1).apply(
                lambda x: _slope_func(x.values), raw=False
            )
        except Exception:
            # Fallback: simple difference if rolling polyfit fails for any reason
            cols['local_slope'] = df['close'].diff().fillna(0)

        # Small smoothing to remove micro-noise (optional but helps HMM/XGB)
        cols['local_slope'] = cols['local_slope'].rolling(window=3, min_periods=1).mean()

        # Ensure no inf / divide-by-zero propagates into regime models
        # (these will be sanitized later, but add protective fill here)
        cols['atr_ratio'] = cols.get('atr_ratio', df.get('atr', np.nan) / df['close'].replace(0, np.nan)).fillna(0)
        cols['macd_mean'] = cols.get('macd_mean', df.get('macd', np.nan).rolling(window=50, min_periods=1).mean()).fillna(0)
        cols['ppo_mean'] = cols.get('ppo_mean', df.get('ppo', np.nan).rolling(window=50, min_periods=1).mean()).fillna(0)

        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}

        # ============================================================================
        # FIX: ENSURE TIMESTAMP COLUMN EXISTS (for confluence scoring)
        # ============================================================================
        
        if 'timestamp' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                cols['timestamp'] = df.index
            else:
                # If index is not datetime, try to convert or use range
                try:
                    cols['timestamp'] = pd.to_datetime(df.index)
                except:
                    cols['timestamp'] = pd.RangeIndex(len(df))
            
        df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        cols = {}

        # Final merge and defragmentation
        if cols:
            df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
        
        # Replace infs that can result from division by zero
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # ========== TEMPORAL FEATURE EXTRACTION (for DL) ==========
        # Convert datetime columns to cyclical numeric features
        # Neural networks benefit from sin/cos encoding for cyclical time patterns
        
        def extract_temporal_features(df):
            """
            Extract numeric temporal features from datetime columns.
            Uses cyclical (sin/cos) encoding for hour, day_of_week, month.
            """
            temp_cols = {}
            
            # Find the datetime source (prioritize 'time', then 'timestamp', then index)
            dt_series = None
            dt_source = None
            
            if 'time' in df.columns:
                try:
                    dt_series = pd.to_datetime(df['time'])
                    dt_source = 'time'
                except:
                    pass
            
            if dt_series is None and 'timestamp' in df.columns:
                try:
                    dt_series = pd.to_datetime(df['timestamp'])
                    dt_source = 'timestamp'
                except:
                    pass
                    
            if dt_series is None and isinstance(df.index, pd.DatetimeIndex):
                dt_series = df.index.to_series()
                dt_source = 'index'
            
            if dt_series is None:
                return temp_cols  # No datetime found, skip
            
            # Hour of day (0-23) → cyclical encoding (forex session patterns)
            hour = dt_series.dt.hour
            temp_cols['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            temp_cols['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            
            # Day of week (0=Monday, 6=Sunday) → cyclical encoding
            dow = dt_series.dt.dayofweek
            temp_cols['day_of_week_sin'] = np.sin(2 * np.pi * dow / 7)
            temp_cols['day_of_week_cos'] = np.cos(2 * np.pi * dow / 7)
            
            # Day of month (1-31) → cyclical encoding (month-end effects)
            dom = dt_series.dt.day
            temp_cols['day_of_month_sin'] = np.sin(2 * np.pi * dom / 31)
            temp_cols['day_of_month_cos'] = np.cos(2 * np.pi * dom / 31)
            
            # Month (1-12) → cyclical encoding (seasonal patterns)
            month = dt_series.dt.month
            temp_cols['month_sin'] = np.sin(2 * np.pi * month / 12)
            temp_cols['month_cos'] = np.cos(2 * np.pi * month / 12)
            
            # Quarter (1-4) → useful for macro patterns
            temp_cols['quarter'] = dt_series.dt.quarter
            
            # Week of year (1-52) → cyclical encoding
            week = dt_series.dt.isocalendar().week.astype(int)
            temp_cols['week_of_year_sin'] = np.sin(2 * np.pi * week / 52)
            temp_cols['week_of_year_cos'] = np.cos(2 * np.pi * week / 52)
            
            # Minute of hour (0-59) → cyclical (for sub-hourly TFs)
            minute = dt_series.dt.minute
            temp_cols['minute_sin'] = np.sin(2 * np.pi * minute / 60)
            temp_cols['minute_cos'] = np.cos(2 * np.pi * minute / 60)
            
            # Trading session flags (forex-specific)
            # Sydney: 22:00-07:00 UTC, Tokyo: 00:00-09:00 UTC
            # London: 08:00-17:00 UTC, New York: 13:00-22:00 UTC
            temp_cols['session_sydney'] = ((hour >= 22) | (hour < 7)).astype(np.int8)
            temp_cols['session_tokyo'] = ((hour >= 0) & (hour < 9)).astype(np.int8)
            temp_cols['session_london'] = ((hour >= 8) & (hour < 17)).astype(np.int8)
            temp_cols['session_newyork'] = ((hour >= 13) & (hour < 22)).astype(np.int8)
            
            # Session overlaps (high volatility periods)
            temp_cols['session_london_ny_overlap'] = ((hour >= 13) & (hour < 17)).astype(np.int8)
            temp_cols['session_tokyo_london_overlap'] = ((hour >= 8) & (hour < 9)).astype(np.int8)
            
            # Weekend proximity (Friday afternoon = reduced liquidity)
            temp_cols['is_friday'] = (dow == 4).astype(np.int8)
            temp_cols['is_monday'] = (dow == 0).astype(np.int8)
            
            return temp_cols
        
        temporal_features = extract_temporal_features(df)
        if temporal_features:
            df = pd.concat([df, pd.DataFrame(temporal_features, index=df.index)], axis=1)
        
        # ========== DL SAFETY: Ensure all columns are numeric ==========
        # Drop any non-numeric columns that may have leaked through (e.g., datetime, strings)
        # Preserve 'timestamp' if needed downstream, but exclude from numeric check
        preserve_cols = ['timestamp']
        numeric_df = df.select_dtypes(include=[np.number])
        non_numeric_cols = [c for c in df.columns if c not in numeric_df.columns and c not in preserve_cols]
        
        if non_numeric_cols:
            # Silently drop datetime columns we've already extracted features from
            expected_drops = ['time', 'date', 'datetime']
            unexpected = [c for c in non_numeric_cols if c.lower() not in expected_drops]
            if unexpected:
                print(f"⚠️ DL Safety: Dropping {len(unexpected)} non-numeric columns: {unexpected}")
            df = df.drop(columns=non_numeric_cols, errors='ignore')
        
        return df.copy()


# Backward compatibility alias
FeatureEngineer = FeatureEngineerOptimized

# --- Execution Example ---
if __name__ == "__main__":
    print("FeatureEngineer cannot run standalone. It requires a database connection.")
    # Example usage:
    # from db_connector import DBConnector
    # connector = DBConnector()
    # fe = FeatureEngineer(connector)
    # fe.calculate_and_save_all_features()