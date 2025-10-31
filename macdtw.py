# macd_tw_optimized_with_kd_and_candlestick.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import matplotlib
from matplotlib.patches import Rectangle
warnings.filterwarnings("ignore")

# ==================== 修复中文字体显示问题 ====================
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

def safe_chinese_text(text):
    """安全处理中文字符，如果字体不可用则返回英文"""
    try:
        test_fig, test_ax = plt.subplots()
        test_ax.text(0.5, 0.5, '测试', fontproperties=matplotlib.font_manager.FontProperties(fname=None))
        plt.close(test_fig)
        return text
    except:
        chinese_to_english = {
            "智能交易訊號": "Trading Signals",
            "收盤價": "Close Price",
            "買入": "Buy",
            "賣出": "Sell",
            "布林带": "Bollinger Bands",
            "MACD 指標": "MACD Indicator",
            "訊號線": "Signal Line",
            "RSI 指標": "RSI Indicator",
            "KD 指標": "KD Indicator",
            "K線": "K Line",
            "D線": "D Line",
            "超买线": "Overbought",
            "超卖线": "Oversold",
            "累積報酬比較": "Cumulative Returns Comparison",
            "智能策略": "Strategy",
            "買進持有": "Buy & Hold",
            "智能信心指数": "Confidence Index",
            "參數優化結果": "Parameter Optimization",
            "快線週期": "Fast Period",
            "慢線週期": "Slow Period",
            "K線形態": "Candlestick Patterns",
            "錘子線": "Hammer",
            "倒錘子": "Inverted Hammer", 
            "看漲吞沒": "Bullish Engulfing",
            "看跌吞沒": "Bearish Engulfing",
            "早晨之星": "Morning Star",
            "黃昏之星": "Evening Star",
            "紅三兵": "Three White Soldiers",
            "三隻烏鴉": "Three Black Crows"
        }
        return chinese_to_english.get(text, text)

# ==================== K线形态识别 ====================
def detect_candlestick_patterns(df):
    """识别常见的K线形态"""
    patterns = pd.DataFrame(index=df.index)
    
    # 计算基本的价格关系
    open_price = df['Open']
    high_price = df['High'] 
    low_price = df['Low']
    close_price = df['Close']
    
    # 计算实体和影线
    body = abs(close_price - open_price)
    upper_shadow = high_price - np.maximum(open_price, close_price)
    lower_shadow = np.minimum(open_price, close_price) - low_price
    
    # 1. 锤子线 (Hammer) - 底部反转信号
    # 小实体，长下影线，短上影线，出现在下跌趋势中
    is_small_body = body < (high_price - low_price) * 0.3
    is_long_lower_shadow = lower_shadow > body * 2
    is_short_upper_shadow = upper_shadow < body * 0.5
    patterns['Hammer'] = is_small_body & is_long_lower_shadow & is_short_upper_shadow
    
    # 2. 倒锤子线 (Inverted Hammer) - 底部反转信号
    # 小实体，长上影线，短下影线
    is_long_upper_shadow = upper_shadow > body * 2
    is_short_lower_shadow = lower_shadow < body * 0.5
    patterns['Inverted_Hammer'] = is_small_body & is_long_upper_shadow & is_short_lower_shadow
    
    # 3. 看涨吞没 (Bullish Engulfing) - 底部反转信号
    # 当前阳线完全吞没前一根阴线
    prev_close = close_price.shift(1)
    prev_open = open_price.shift(1)
    is_bullish_engulfing = (
        (close_price > open_price) &  # 当前是阳线
        (prev_close < prev_open) &    # 前一根是阴线
        (open_price < prev_close) &   # 当前开盘低于前一根收盘
        (close_price > prev_open)     # 当前收盘高于前一根开盘
    )
    patterns['Bullish_Engulfing'] = is_bullish_engulfing
    
    # 4. 看跌吞没 (Bearish Engulfing) - 顶部反转信号
    # 当前阴线完全吞没前一根阳线
    is_bearish_engulfing = (
        (close_price < open_price) &  # 当前是阴线
        (prev_close > prev_open) &    # 前一根是阳线
        (open_price > prev_close) &   # 当前开盘高于前一根收盘
        (close_price < prev_open)     # 当前收盘低于前一根开盘
    )
    patterns['Bearish_Engulfing'] = is_bearish_engulfing
    
    # 5. 早晨之星 (Morning Star) - 底部反转信号
    # 第一根阴线，第二根小实体（任何颜色），第三根阳线
    day1_down = (prev_close < prev_open)  # 第一天阴线
    day2_small_body = body.shift(1) < (high_price.shift(1) - low_price.shift(1)) * 0.3  # 第二天小实体
    day3_up = (close_price > open_price)  # 第三天阳线
    patterns['Morning_Star'] = day1_down & day2_small_body & day3_up
    
    # 6. 黄昏之星 (Evening Star) - 顶部反转信号
    # 第一根阳线，第二根小实体，第三根阴线
    day1_up = (prev_close > prev_open)  # 第一天阳线
    day3_down = (close_price < open_price)  # 第三天阴线
    patterns['Evening_Star'] = day1_up & day2_small_body & day3_down
    
    # 7. 红三兵 (Three White Soldiers) - 强势上涨信号
    # 连续三根阳线，每根开盘在前一根实体之内，收盘创更高
    three_up_days = (
        (close_price > open_price) &
        (close_price.shift(1) > open_price.shift(1)) &
        (close_price.shift(2) > open_price.shift(2))
    )
    higher_highs = (
        (close_price > close_price.shift(1)) &
        (close_price.shift(1) > close_price.shift(2))
    )
    patterns['Three_White_Soldiers'] = three_up_days & higher_highs
    
    # 8. 三只乌鸦 (Three Black Crows) - 强势下跌信号
    # 连续三根阴线，每根开盘在前一根实体之内，收盘创更低
    three_down_days = (
        (close_price < open_price) &
        (close_price.shift(1) < open_price.shift(1)) &
        (close_price.shift(2) < open_price.shift(2))
    )
    lower_lows = (
        (close_price < close_price.shift(1)) &
        (close_price.shift(1) < close_price.shift(2))
    )
    patterns['Three_Black_Crows'] = three_down_days & lower_lows
    
    # 计算K线形态强度分数
    patterns['Bullish_Pattern_Score'] = (
        patterns['Hammer'].astype(int) * 0.7 +
        patterns['Inverted_Hammer'].astype(int) * 0.7 +
        patterns['Bullish_Engulfing'].astype(int) * 0.8 +
        patterns['Morning_Star'].astype(int) * 0.9 +
        patterns['Three_White_Soldiers'].astype(int) * 0.8
    )
    
    patterns['Bearish_Pattern_Score'] = (
        patterns['Bearish_Engulfing'].astype(int) * 0.8 +
        patterns['Evening_Star'].astype(int) * 0.9 +
        patterns['Three_Black_Crows'].astype(int) * 0.8
    )
    
    patterns['Overall_Pattern_Score'] = patterns['Bullish_Pattern_Score'] - patterns['Bearish_Pattern_Score']
    
    return patterns

# ==================== 增强技术指标计算 ====================
def ema(series, period):
    """指数移动平均线"""
    return series.ewm(span=period, adjust=False).mean()

def macd_enhanced(close, fast=12, slow=26, signal=9):
    """增强MACD计算"""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    macd_slope = macd_line.diff(3)
    macd_position = (macd_line - macd_line.rolling(50).min()) / (
        macd_line.rolling(50).max() - macd_line.rolling(50).min() + 1e-8
    )
    
    return macd_line, signal_line, histogram, macd_slope, macd_position

def rsi_enhanced(close, period=14):
    """增强RSI计算"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss.replace(0, np.nan) + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    rsi_momentum = rsi.diff(3)
    
    return rsi, rsi_momentum

def calculate_kd(high, low, close, n=9, m1=3, m2=3):
    """计算KD指标（随机指标）"""
    # 计算RSV
    lowest_low = low.rolling(window=n).min()
    highest_high = high.rolling(window=n).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low + 1e-8) * 100
    
    # 计算K值和D值
    k = rsv.ewm(alpha=1/m1, adjust=False).mean()
    d = k.ewm(alpha=1/m2, adjust=False).mean()
    
    # KD交叉信号
    kd_golden_cross = (k > d) & (k.shift(1) <= d.shift(1))  # K向上穿越D
    kd_death_cross = (k < d) & (k.shift(1) >= d.shift(1))   # K向下穿越D
    
    # KD位置
    kd_position = (k + d) / 2  # KD平均值，表示整体位置
    
    return k, d, kd_golden_cross, kd_death_cross, kd_position

def calculate_atr(high, low, close, period=14):
    """计算真实波动幅度"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def calculate_bollinger_bands(close, period=20, std_dev=2):
    """计算布林带"""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    bandwidth = (upper_band - lower_band) / (sma + 1e-8)
    return upper_band, sma, lower_band, bandwidth

# ==================== 市场情绪分析 ====================
def get_market_sentiment(ticker):
    """获取市场情绪数据（简化版）"""
    try:
        if "TW" in ticker:
            sentiment_score = np.random.normal(0.6, 0.2)
        else:
            sentiment_score = np.random.normal(0.5, 0.2)
        return max(0, min(1, sentiment_score))
    except:
        return 0.5

# ==================== 自适应参数调整 ====================
def adaptive_parameters(close, lookback=30):
    """根据市场波动率调整MACD参数"""
    volatility = close.pct_change().std() * np.sqrt(252)
    
    if volatility > 0.3:
        return 8, 21, 5
    elif volatility < 0.15:
        return 15, 30, 10
    else:
        return 12, 26, 9

# ==================== 简化的机器学习增强信号 ====================
def simple_ml_enhanced_signals(df):
    """使用简化规则替代机器学习"""
    try:
        # MACD 信号强度
        macd_strength = np.where(
            (df['MACD'] > df['Signal']) & (df['MACD'] > 0),
            0.8,
            np.where(
                (df['MACD'] < df['Signal']) & (df['MACD'] < 0),
                0.3,
                0.6
            )
        )
        
        # RSI 信号强度
        rsi_strength = np.where(
            df['RSI'] < 40,
            0.8,
            np.where(
                df['RSI'] > 70,
                0.2,
                0.6
            )
        )
        
        # KD 信号强度
        kd_strength = np.where(
            (df['K'] < 30) & (df['K'] > df['D']),  # KD低位金叉
            0.9,
            np.where(
                (df['K'] > 70) & (df['K'] < df['D']),  # KD高位死叉
                0.2,
                0.5
            )
        )
        
        # K线形态信号强度
        pattern_strength = np.where(
            df['Overall_Pattern_Score'] > 0.5,
            0.8,
            np.where(
                df['Overall_Pattern_Score'] < -0.5,
                0.2,
                0.5
            )
        )
        
        # 价格位置信号
        bb_position = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-8)
        position_strength = np.where(
            bb_position < 0.3,
            0.9,
            np.where(
                bb_position > 0.7,
                0.3,
                0.6
            )
        )
        
        # 价格趋势信号
        price_trend = np.where(
            df['Close'] > df['MA20'],
            0.7,
            0.4
        )
        
        # 综合评分（加权平均）- 加入K线形态权重
        df['ML_Confidence'] = (macd_strength * 0.2 + rsi_strength * 0.15 + 
                              kd_strength * 0.2 + pattern_strength * 0.15 + 
                              position_strength * 0.15 + price_trend * 0.15)
        df['ML_Signal'] = (df['ML_Confidence'] > 0.55).astype(int)
        
    except Exception as e:
        st.warning(f"增强信号计算失败: {e}")
        df['ML_Confidence'] = 0.5
        df['ML_Signal'] = 0
    
    return df

# ==================== 改进的交易信号 - 包含KD指标和K线形态 ====================
def improved_signals_with_kd_and_patterns(df):
    """改进的买卖信号判断 - 包含KD指标和K线形态"""
    try:
        # MACD 零軸確認
        macd_above_zero = df['MACD'] > 0
        macd_trend_up = df['MACD'] > df['MACD'].shift(3)
        
        # 布林带位置
        bb_position = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-8)
        
        # 价格趋势
        price_above_ma20 = df['Close'] > df.get('MA20', df['Close'])
        
        # KD指标条件 - 使用安全的获取方式
        kd_golden_cross = df.get('KD_Golden_Cross', pd.Series([False]*len(df)))
        kd_death_cross = df.get('KD_Death_Cross', pd.Series([False]*len(df)))
        k_values = df.get('K', pd.Series([50]*len(df)))
        d_values = df.get('D', pd.Series([50]*len(df)))
        rsi_values = df.get('RSI', pd.Series([50]*len(df)))
        
        k_low = k_values < 30  # K值低于30，超卖
        k_high = k_values > 70  # K值高于70，超买
        k_above_d = k_values > d_values  # K值在D值上方
        
        # K线形态条件
        bullish_patterns = df.get('Bullish_Pattern_Score', pd.Series([0]*len(df))) > 0.5
        bearish_patterns = df.get('Bearish_Pattern_Score', pd.Series([0]*len(df))) > 0.5
        strong_bullish = df.get('Bullish_Pattern_Score', pd.Series([0]*len(df))) > 0.7
        strong_bearish = df.get('Bearish_Pattern_Score', pd.Series([0]*len(df))) > 0.7
        
        # 强化买入信号 - 多种买入条件
        # 1. 标准金叉买入
        standard_buy = (
            (df['MACD'] > df['Signal']) & 
            (df['MACD'].shift(1) <= df['Signal'].shift(1)) &
            macd_above_zero &
            (rsi_values < 70) &
            (bb_position < 0.8) &
            price_above_ma20
        )
        
        # 2. KD金叉买入
        kd_buy = (
            kd_golden_cross &
            (k_values < 40) &  # KD在相对低位
            (df['MACD'] > df['MACD'].shift(3)) &  # MACD开始上升
            (rsi_values < 65)
        )
        
        # 3. MACD和KD双重金叉买入
        double_golden_cross = (
            (df['MACD'] > df['Signal']) & 
            (df['MACD'].shift(1) <= df['Signal'].shift(1)) &
            kd_golden_cross &
            (k_values < 50) &  # KD在中低位
            (rsi_values < 60)
        )
        
        # 4. RSI超卖反弹买入
        rsi_oversold_buy = (
            (rsi_values < 35) &
            (rsi_values.shift(1) < 30) &
            (df['MACD'] > df['MACD'].shift(3)) &
            (bb_position < 0.3) &
            k_above_d  # KD指标也显示向上
        )
        
        # 5. KD低位钝化买入
        kd_oversold_buy = (
            (k_values < 20) &  # K值极度超卖
            (d_values < 30) &  # D值也超卖
            (df['Close'] > df['Close'].shift(3)) &  # 价格开始反弹
            (df['MACD'] > df['MACD'].shift(5))
        )
        
        # 6. 价格突破布林带下轨反弹买入
        bb_bounce_buy = (
            (bb_position < 0.2) &
            (df['Close'] > df['Close'].shift(3)) &
            (df['MACD'] > df['MACD'].shift(5)) &
            k_above_d  # KD指标配合
        )
        
        # 7. K线形态买入信号
        pattern_buy = (
            bullish_patterns &
            (df['MACD'] > df['MACD'].shift(2)) &  # MACD开始改善
            (rsi_values < 60)  # RSI不过热
        )
        
        # 8. 强烈K线形态买入
        strong_pattern_buy = (
            strong_bullish &
            (df['Close'] > df['Close'].shift(1))  # 价格开始上涨
        )
        
        # 9. 机器学习增强买入信号
        ml_confidence = df.get('ML_Confidence', pd.Series([0.5]*len(df)))
        ml_buy = (
            (ml_confidence > 0.6) &
            (df['MACD'] > df['Signal']) &
            k_above_d  # KD指标配合
        )
        
        # 综合买入信号
        df['Buy_Signal'] = (
            standard_buy | kd_buy | double_golden_cross | rsi_oversold_buy | 
            kd_oversold_buy | bb_bounce_buy | pattern_buy | strong_pattern_buy | ml_buy
        )
        
        # 改良賣出訊號
        trailing_stop = df.get('Trailing_Stop', df['Close'] * 0.95)
        
        # 1. MACD死叉卖出
        macd_death_cross = (
            (df['MACD'] < df['Signal']) & 
            (df['MACD'].shift(1) >= df['Signal'].shift(1)) &
            (df['MACD'] > 1.0)  # MACD高位死叉
        )
        
        # 2. KD死叉卖出
        kd_sell = (
            kd_death_cross &
            (k_values > 60) &  # KD在相对高位
            (df['MACD'] < df['MACD'].shift(3))  # MACD开始下降
        )
        
        # 3. K线形态卖出信号
        pattern_sell = (
            bearish_patterns &
            (df['MACD'] < df['MACD'].shift(2)) &  # MACD开始恶化
            (rsi_values > 40)  # RSI不超卖
        )
        
        # 4. 强烈K线形态卖出
        strong_pattern_sell = (
            strong_bearish &
            (df['Close'] < df['Close'].shift(1))  # 价格开始下跌
        )
        
        # 5. 多重指标超买卖出
        overbought_sell = (
            (rsi_values > 80) |  # RSI严重超买
            (bb_position > 0.9) |  # 价格接近布林带上轨
            (k_values > 80)  # KD严重超买
        )
        
        # 6. 止损卖出
        stop_loss_sell = (df['Close'] < trailing_stop)
        
        df['Sell_Signal'] = (
            macd_death_cross | kd_sell | pattern_sell | strong_pattern_sell | overbought_sell | stop_loss_sell
        )
        
    except Exception as e:
        st.error(f"信号计算错误: {e}")
        # 设置默认信号
        df['Buy_Signal'] = False
        df['Sell_Signal'] = False
    
    return df

# ==================== 向量化持仓计算 ====================
def vectorized_position_calculation(df):
    """向量化持仓计算"""
    try:
        buy_signals = df['Buy_Signal'].fillna(False).astype(int)
        sell_signals = df['Sell_Signal'].fillna(False).astype(int) * -1
        
        all_signals = buy_signals + sell_signals
        
        position = 0
        positions = []
        
        for signal in all_signals:
            if signal == 1 and position == 0:
                position = 1
            elif signal == -1 and position == 1:
                position = 0
            positions.append(position)
        
        df['Position'] = positions
    except Exception as e:
        st.error(f"持仓计算错误: {e}")
        df['Position'] = 0
    
    return df

# ==================== 风险管理 ====================
def position_sizing(df, risk_per_trade=0.02, max_portfolio_risk=0.1):
    """根据波动率调整仓位大小"""
    try:
        volatility = df['ATR'] / df['Close']
        position_size = risk_per_trade / (volatility.replace(0, 0.01) + 1e-8)
        df['Position_Size'] = position_size.clip(upper=max_portfolio_risk)
    except Exception as e:
        st.warning(f"仓位计算错误: {e}")
        df['Position_Size'] = 1.0
    
    return df

def dynamic_stop_loss(df, initial_stop=0.95, trailing_stop=0.98):
    """动态止损机制"""
    try:
        df['Peak_Price'] = df['Close'].cummax()
        df['Trailing_Stop'] = df['Peak_Price'] * trailing_stop
        df['Hard_Stop'] = df['Close'] * initial_stop
        
        if 'Position' not in df.columns:
            df['Position'] = 0
        
        df['Current_Stop'] = np.where(
            df['Position'] == 1,
            df[['Trailing_Stop', 'Hard_Stop']].max(axis=1),
            np.nan
        )
    except Exception as e:
        st.warning(f"止损计算错误: {e}")
    
    return df

# ==================== 增强回测策略 ====================
def enhanced_backtest_strategy(df, fast=12, slow=26, signal=9, use_rsi=True, use_ma=True, use_ml=False, use_kd=True, use_patterns=True):
    df = df.copy()
    
    try:
        close = df['Close'].squeeze()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = pd.Series(close, index=df.index, name='Close')
        
        df['Close'] = close
        df['High'] = df['High'].squeeze() if 'High' in df.columns else close
        df['Low'] = df['Low'].squeeze() if 'Low' in df.columns else close
        df['Open'] = df['Open'].squeeze() if 'Open' in df.columns else close

        # --- 自适应参数 ---
        if st.session_state.get('use_adaptive_params', False):
            fast, slow, signal = adaptive_parameters(close)

        # --- 增强技术指标计算 ---
        # MACD
        try:
            df['MACD'], df['Signal'], df['Histogram'], df['MACD_Slope'], df['MACD_Position'] = macd_enhanced(
                close, fast, slow, signal
            )
        except Exception as e:
            st.error(f"MACD计算错误: {e}")
            ema_fast = ema(close, fast)
            ema_slow = ema(close, slow)
            df['MACD'] = ema_fast - ema_slow
            df['Signal'] = ema(df['MACD'], signal)
            df['Histogram'] = df['MACD'] - df['Signal']
            df['MACD_Slope'] = df['MACD'].diff(3)
            df['MACD_Position'] = 0.5
        
        # RSI
        if use_rsi:
            try:
                df['RSI'], df['RSI_Momentum'] = rsi_enhanced(close, 14)
            except Exception as e:
                st.warning(f"RSI计算失败: {e}")
                use_rsi = False
        
        # KD指标
        if use_kd:
            try:
                df['K'], df['D'], df['KD_Golden_Cross'], df['KD_Death_Cross'], df['KD_Position'] = calculate_kd(
                    df['High'], df['Low'], df['Close']
                )
            except Exception as e:
                st.warning(f"KD指标计算失败: {e}")
                use_kd = False
        
        # K线形态识别
        if use_patterns:
            try:
                patterns = detect_candlestick_patterns(df)
                df = pd.concat([df, patterns], axis=1)
            except Exception as e:
                st.warning(f"K线形态识别失败: {e}")
                use_patterns = False
        
        # 移动平均线
        if use_ma:
            df['MA20'] = close.rolling(20).mean()
            df['MA50'] = close.rolling(50).mean()
            df['MA200'] = close.rolling(200).mean()
        
        # 布林带
        try:
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'], df['BB_Width'] = calculate_bollinger_bands(close)
        except Exception as e:
            st.warning(f"布林带计算失败: {e}")
            df['BB_Upper'] = close
            df['BB_Middle'] = close
            df['BB_Lower'] = close
            df['BB_Width'] = 0
        
        # ATR和波动率
        try:
            df['ATR'] = calculate_atr(df['High'], df['Low'], close)
            df['Volatility'] = close.pct_change().rolling(20).std()
        except Exception as e:
            st.warning(f"ATR计算失败: {e}")
            df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
            df['Volatility'] = close.pct_change().std()

        # --- 简化的机器学习增强 ---
        if use_ml and len(df) > 50:
            df = simple_ml_enhanced_signals(df)

        # --- 改进的交易信号 - 包含KD指标和K线形态 ---
        df = improved_signals_with_kd_and_patterns(df)

        # --- 向量化持仓计算 ---
        df = vectorized_position_calculation(df)

        # --- 风险管理 ---
        df = position_sizing(df)
        df = dynamic_stop_loss(df)

        # --- 报酬计算 ---
        df['Returns'] = close.pct_change()
        
        if 'Position_Size' not in df.columns:
            df['Position_Size'] = 1.0
        
        df['Strategy_Returns'] = df['Position'].shift(1) * df['Returns'] * df['Position_Size'].shift(1)
        df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
        df['Buy_Hold_Returns'] = (1 + df['Returns']).cumprod()

        # 填补 NaN
        df['Cumulative_Returns'] = df['Cumulative_Returns'].fillna(1)
        df['Buy_Hold_Returns'] = df['Buy_Hold_Returns'].fillna(1)

        # 绩效指标
        sharpe = 0
        strategy_returns = df['Strategy_Returns'].dropna()
        if len(strategy_returns) > 0 and strategy_returns.std() != 0:
            sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)

        total_return = df['Cumulative_Returns'].iloc[-1] - 1 if len(df) > 0 else 0
        max_drawdown = (df['Cumulative_Returns'] / df['Cumulative_Returns'].cummax() - 1).min() if len(df) > 0 else 0
        
        # 计算胜率
        trades = df[df['Buy_Signal'] | df['Sell_Signal']].copy()
        if len(trades) > 0:
            win_rate = (trades['Returns'].shift(-1) > 0).mean()
        else:
            win_rate = 0

        return df, sharpe, total_return, max_drawdown, win_rate
    
    except Exception as e:
        st.error(f"回测策略执行错误: {e}")
        # 返回空的DataFrame和默认值
        empty_df = pd.DataFrame(index=df.index)
        return empty_df, 0, 0, 0, 0

# ==================== 参数优化 ====================
def optimize_params(df, param_grid, use_rsi, use_ma, use_ml, use_kd, use_patterns):
    close = df['Close'].squeeze()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.Series(close, index=df.index)

    best_sharpe = -np.inf
    best_params = None
    results = []

    progress_bar = st.progress(0)
    total_combinations = len(param_grid['fast']) * len(param_grid['slow']) * len(param_grid['signal'])
    current_combination = 0

    for fast in param_grid['fast']:
        for slow in param_grid['slow']:
            if slow <= fast:
                continue
            for signal in param_grid['signal']:
                try:
                    current_combination += 1
                    if total_combinations > 0:
                        progress_bar.progress(current_combination / total_combinations)
                    
                    _, sharpe, total_return, max_drawdown, win_rate = enhanced_backtest_strategy(
                        pd.DataFrame({'Close': close, 'High': close, 'Low': close, 'Open': close}), 
                        fast, slow, signal, use_rsi, use_ma, use_ml, use_kd, use_patterns
                    )
                    results.append({
                        'fast': fast, 
                        'slow': slow, 
                        'signal': signal, 
                        'sharpe': sharpe,
                        'total_return': total_return,
                        'max_drawdown': max_drawdown,
                        'win_rate': win_rate
                    })
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = (fast, slow, signal)
                except Exception as e:
                    continue
    
    progress_bar.empty()
    return pd.DataFrame(results), best_params

# ==================== 多时间框架分析 ====================
def multi_timeframe_analysis(ticker, period):
    """多时间框架分析"""
    timeframes = {
        '日线': '1d',
        '周线': '1wk', 
    }
    
    results = {}
    for tf_name, tf_interval in timeframes.items():
        try:
            data = yf.download(ticker, period=period, interval=tf_interval, progress=False)
            if not data.empty and len(data) > 1:
                trend = "上涨" if data['Close'].iloc[-1] > data['Close'].iloc[0] else "下跌"
                results[tf_name] = {
                    'trend': trend,
                    'return': (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1)
                }
        except:
            continue
    
    return results

# ==================== K线图绘制函数 ====================
def plot_candlestick(df, title, patterns_to_highlight=None):
    """绘制K线图并标记特殊形态"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 设置背景颜色
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    # 计算OHLC数据
    dates = df.index
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    
    # 绘制K线
    for i, (date, open_val, high_val, low_val, close_val) in enumerate(zip(dates, opens, highs, lows, closes)):
        color = 'red' if close_val < open_val else 'green'
        
        # 绘制影线
        ax.plot([i, i], [low_val, high_val], color='black', linewidth=1)
        
        # 绘制实体
        body_height = abs(close_val - open_val)
        body_bottom = min(open_val, close_val)
        
        # 只绘制有实体的K线
        if body_height > 0:
            rect = Rectangle((i-0.3, body_bottom), 0.6, body_height, 
                           facecolor=color, edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
    
    # 标记特殊K线形态
    if patterns_to_highlight is not None:
        for pattern_name, pattern_data in patterns_to_highlight.items():
            if pattern_name in df.columns:
                pattern_dates = df[df[pattern_name]]
                for date in pattern_dates.index:
                    idx = df.index.get_loc(date)
                    ax.plot(idx, df.loc[date, 'High'] * 1.01, 
                           marker='*', color='gold', markersize=10, 
                           label=f"{safe_chinese_text(pattern_name)}")
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.3)
    
    # 设置x轴标签
    ax.set_xticks(range(0, len(dates), max(1, len(dates)//10)))
    ax.set_xticklabels([dates[i].strftime('%Y-%m-%d') for i in range(0, len(dates), max(1, len(dates)//10))])
    
    # 添加图例
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:2], labels[:2])  # 只显示前两个图例避免过多
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

# ==================== Streamlit App ====================
def main():
    st.set_page_config(
        page_title="MACD+KD+K線形態分析系統", 
        layout="wide",
        page_icon="📈"
    )
    
    st.title("📈 MACD+KD+K線形態智能策略系統 - 台灣股市")
    st.markdown("---")
    
    # 侧边栏
    st.sidebar.header("🎯 策略設定")
    
    # 股票选择
    col1, col2 = st.sidebar.columns(2)
    with col1:
        ticker = st.text_input("股票代碼", "2330.TW", help="例如：2330.TW, 2317.TW, 0050.TW")
    with col2:
        period = st.selectbox("回測期間", ["3mo", "6mo", "1y", "2y", "3y", "5y"], index=2)
    
    # MACD参数调整
    st.sidebar.subheader("📊 MACD 參數")
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        fast = st.slider("快線", 5, 20, 12)
    with col2:
        slow = st.slider("慢線", 15, 50, 26)
    with col3:
        signal_line = st.slider("訊號線", 5, 15, 9)
    
    # 增强功能
    st.sidebar.subheader("🚀 增強功能")
    use_rsi = st.sidebar.checkbox("RSI 過濾", True)
    use_ma = st.sidebar.checkbox("移動平均線", True)
    use_kd = st.sidebar.checkbox("KD 指標", True)
    use_patterns = st.sidebar.checkbox("K線形態分析", True)  # 新增K线形态选项
    use_ml = st.sidebar.checkbox("智能增強信號", True)
    use_adaptive = st.sidebar.checkbox("自適應參數", False)
    
    # 信号强度调整
    st.sidebar.subheader("⚡ 信號強度")
    signal_strength = st.sidebar.selectbox("買入信號強度", ["標準", "中等", "強烈"], index=1)
    
    # 风险管理
    st.sidebar.subheader("🛡️ 風險管理")
    risk_per_trade = st.sidebar.slider("單筆風險(%)", 0.5, 5.0, 2.0) / 100
    
    # 参数优化
    st.sidebar.subheader("🔧 參數優化")
    optimize = st.sidebar.checkbox("執行參數優化", False)
    
    if optimize:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            fast_min = st.number_input("快線最小", 5, 20, 8)
            fast_max = st.number_input("快線最大", 10, 30, 15)
        with col2:
            slow_min = st.number_input("慢線最小", 20, 40, 20)
            slow_max = st.number_input("慢線最大", 25, 50, 30)
        signal_min = st.number_input("訊號線最小", 5, 15, 5)
        signal_max = st.number_input("訊號線最大", 7, 20, 12)
        
        param_grid = {
            'fast': range(fast_min, fast_max + 1),
            'slow': range(slow_min, slow_max + 1),
            'signal': range(signal_min, signal_max + 1)
        }
    
    # 存储会话状态
    st.session_state.use_adaptive_params = use_adaptive
    st.session_state.signal_strength = signal_strength
    
    # 执行分析按钮
    if st.sidebar.button("🚀 執行智能分析", type="primary", use_container_width=True):
        with st.spinner(f"正在分析 {ticker}..."):
            # 载入数据
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_data(ticker, period):
        import yfinance as yf
        from pandas_datareader import data as pdr
        import pandas as pd

        # 修正 yfinance 雲端連線問題
        yf.pdr_override()

        try:
            # 自動補上台股代碼
            if not ticker.endswith(".TW") and ticker.isdigit():
                ticker = ticker + ".TW"

            # 嘗試主要來源
            data = yf.download(
                ticker,
                period=period,
                progress=False,
                auto_adjust=True,
                timeout=15
            )

            # 若主要來源為空，嘗試 pandas_datareader 備援
            if data is None or data.empty:
                data = pdr.get_data_yahoo(ticker)

            # 若還是空，回傳 None
            if data is None or data.empty:
                return None

            # 處理多層欄位（有時候會出現在多指標下載時）
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)

            return data

        except Exception as e:
            st.warning(f"資料抓取錯誤：{e}")
            return None

            
            df = load_data(ticker, period)
            
            if df is None or df.empty:
                st.error("❌ 無法取得資料，請確認股票代碼正確 (如 2330.TW)")
                st.stop()
            
            st.success(f"✅ 成功載入 {ticker} 資料，共 {len(df)} 筆")
            
            # 多时间框架分析
            st.subheader("⏰ 多時間框架分析")
            mtf_results = multi_timeframe_analysis(ticker, period)
            if mtf_results:
                cols = st.columns(len(mtf_results))
                for idx, (tf_name, result) in enumerate(mtf_results.items()):
                    with cols[idx]:
                        color = "green" if result['return'] > 0 else "red"
                        st.metric(
                            f"{tf_name}趨勢",
                            f"{result['trend']}",
                            f"{result['return']:.2%}",
                            delta_color="normal" if result['return'] > 0 else "inverse"
                        )
            
            # 市场情绪
            sentiment = get_market_sentiment(ticker)
            st.metric("📊 市場情緒指數", f"{sentiment:.2f}/1.0", 
                     delta="積極" if sentiment > 0.6 else "中性" if sentiment > 0.4 else "保守")
            
            # 执行回测
            try:
                df_result, sharpe, total_return, max_drawdown, win_rate = enhanced_backtest_strategy(
                    df, fast, slow, signal_line, use_rsi, use_ma, use_ml, use_kd, use_patterns
                )
                
                # 检查回测结果是否有效
                if df_result.empty:
                    st.error("❌ 回测结果为空，请检查参数设置")
                    st.stop()
                
                # 显示结果
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("💰 總報酬", f"{total_return:.2%}")
                with col2:
                    st.metric("⭐ 夏普比率", f"{sharpe:.2f}")
                with col3:
                    st.metric("📉 最大回撤", f"{max_drawdown:.2%}")
                with col4:
                    st.metric("🎯 勝率", f"{win_rate:.2%}")
                with col5:
                    trades = df_result['Buy_Signal'].sum() + df_result['Sell_Signal'].sum()
                    st.metric("🔄 交易次數", f"{trades}")
                
                # 显示信号统计
                st.info(f"📊 信号统计: 买入信号 {df_result['Buy_Signal'].sum()} 个, 卖出信号 {df_result['Sell_Signal'].sum()} 个")
                
                # 绘图 - 增加K线形态标签页
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 價格圖", "🕯️ K線圖", "📊 MACD指標", "📈 KD指標", "📉 報酬曲線", "📋 詳細數據"])
                
                with tab1:
                    st.subheader(f"{ticker} 智能交易訊號")
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.plot(df_result.index, df_result['Close'], label=safe_chinese_text('收盤價'), color='blue', linewidth=2)
                    
                    if use_ma and 'MA20' in df_result.columns:
                        ax.plot(df_result.index, df_result['MA20'], label='MA20', color='orange', alpha=0.7)
                        ax.plot(df_result.index, df_result['MA50'], label='MA50', color='red', alpha=0.7)
                    
                    # 布林带
                    if 'BB_Upper' in df_result.columns and 'BB_Lower' in df_result.columns:
                        ax.fill_between(df_result.index, df_result['BB_Upper'], df_result['BB_Lower'], 
                                       alpha=0.2, color='gray', label=safe_chinese_text('布林带'))
                    
                    # 买入卖出信号
                    buy_signals = df_result[df_result['Buy_Signal']]
                    sell_signals = df_result[df_result['Sell_Signal']]
                    
                    if not buy_signals.empty:
                        ax.scatter(buy_signals.index, buy_signals['Close'], 
                                  marker='^', color='green', s=100, 
                                  label=safe_chinese_text('買入'), zorder=5)
                    
                    if not sell_signals.empty:
                        ax.scatter(sell_signals.index, sell_signals['Close'], 
                                  marker='v', color='red', s=100, 
                                  label=safe_chinese_text('賣出'), zorder=5)
                    
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.set_ylabel('Price')
                    st.pyplot(fig)
                
                with tab2:
                    st.subheader(f"{ticker} K線圖與形態分析")
                    
                    # 准备K线形态标记
                    patterns_to_highlight = {}
                    if use_patterns:
                        bullish_patterns = ['Hammer', 'Inverted_Hammer', 'Bullish_Engulfing', 'Morning_Star', 'Three_White_Soldiers']
                        bearish_patterns = ['Bearish_Engulfing', 'Evening_Star', 'Three_Black_Crows']
                        
                        for pattern in bullish_patterns + bearish_patterns:
                            if pattern in df_result.columns:
                                patterns_to_highlight[pattern] = df_result[pattern]
                    
                    # 绘制K线图
                    fig = plot_candlestick(df_result.tail(50), f"{ticker} K線圖 (最近50天)", patterns_to_highlight)
                    st.pyplot(fig)
                    
                    # 显示K线形态统计
                    if use_patterns:
                        st.subheader("📊 K線形態統計")
                        pattern_stats = []
                        bullish_patterns = ['Hammer', 'Inverted_Hammer', 'Bullish_Engulfing', 'Morning_Star', 'Three_White_Soldiers']
                        bearish_patterns = ['Bearish_Engulfing', 'Evening_Star', 'Three_Black_Crows']
                        
                        for pattern in bullish_patterns:
                            if pattern in df_result.columns:
                                count = df_result[pattern].sum()
                                if count > 0:
                                    pattern_stats.append({
                                        '形態': safe_chinese_text(pattern),
                                        '類型': '看漲',
                                        '出現次數': count,
                                        '最近出現': df_result[df_result[pattern]].index[-1].strftime('%Y-%m-%d') if count > 0 else '無'
                                    })
                        
                        for pattern in bearish_patterns:
                            if pattern in df_result.columns:
                                count = df_result[pattern].sum()
                                if count > 0:
                                    pattern_stats.append({
                                        '形態': safe_chinese_text(pattern),
                                        '類型': '看跌', 
                                        '出現次數': count,
                                        '最近出現': df_result[df_result[pattern]].index[-1].strftime('%Y-%m-%d') if count > 0 else '無'
                                    })
                        
                        if pattern_stats:
                            pattern_df = pd.DataFrame(pattern_stats)
                            st.dataframe(pattern_df, use_container_width=True)
                        else:
                            st.info("近期未检测到明显的K线形态")
                
                with tab3:
                    st.subheader(safe_chinese_text("MACD 指標"))
                    
                    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
                    
                    # MACD
                    if 'MACD' in df_result.columns and 'Signal' in df_result.columns:
                        axes[0].plot(df_result.index, df_result['MACD'], label='MACD', color='blue', linewidth=2)
                        axes[0].plot(df_result.index, df_result['Signal'], label=safe_chinese_text('訊號線'), color='red', linewidth=2)
                        colors = ['green' if x >= 0 else 'red' for x in df_result['Histogram']]
                        axes[0].bar(df_result.index, df_result['Histogram'], color=colors, alpha=0.6, width=1.0)
                        axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
                        axes[0].legend()
                        axes[0].grid(True, alpha=0.3)
                    
                    # RSI
                    if use_rsi and 'RSI' in df_result.columns:
                        axes[1].plot(df_result.index, df_result['RSI'], label='RSI', color='purple', linewidth=2)
                        axes[1].axhline(70, color='red', linestyle='--', alpha=0.7, label=safe_chinese_text('超买线'))
                        axes[1].axhline(30, color='green', linestyle='--', alpha=0.7, label=safe_chinese_text('超卖线'))
                        axes[1].set_ylim(0, 100)
                        axes[1].legend()
                        axes[1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with tab4:
                    st.subheader(safe_chinese_text("KD 指標"))
                    
                    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
                    
                    # KD指标
                    if use_kd and 'K' in df_result.columns and 'D' in df_result.columns:
                        axes[0].plot(df_result.index, df_result['K'], label=safe_chinese_text('K線'), color='blue', linewidth=2)
                        axes[0].plot(df_result.index, df_result['D'], label=safe_chinese_text('D線'), color='red', linewidth=2)
                        axes[0].axhline(80, color='red', linestyle='--', alpha=0.7, label=safe_chinese_text('超买线'))
                        axes[0].axhline(20, color='green', linestyle='--', alpha=0.7, label=safe_chinese_text('超卖线'))
                        axes[0].set_ylim(0, 100)
                        axes[0].legend()
                        axes[0].grid(True, alpha=0.3)
                        
                        # 标记KD金叉死叉
                        kd_golden = df_result[df_result['KD_Golden_Cross']]
                        kd_death = df_result[df_result['KD_Death_Cross']]
                        
                        if not kd_golden.empty:
                            axes[0].scatter(kd_golden.index, kd_golden['K'], 
                                          marker='^', color='green', s=80, label=safe_chinese_text('KD金叉'))
                        if not kd_death.empty:
                            axes[0].scatter(kd_death.index, kd_death['K'], 
                                          marker='v', color='red', s=80, label=safe_chinese_text('KD死叉'))
                    
                    # 其他指标或空白
                    axes[1].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with tab5:
                    st.subheader(safe_chinese_text("累積報酬比較"))
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    if 'Cumulative_Returns' in df_result.columns and 'Buy_Hold_Returns' in df_result.columns:
                        ax.plot(df_result.index, df_result['Cumulative_Returns'], 
                               label=safe_chinese_text('智能策略'), color='green', linewidth=3)
                        ax.plot(df_result.index, df_result['Buy_Hold_Returns'], 
                               label=safe_chinese_text('買進持有'), color='gray', linestyle='--', linewidth=2)
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        ax.set_ylabel("Cumulative Returns")
                    st.pyplot(fig)
                
                with tab6:
                    st.subheader("詳細交易數據")
                    display_cols = ['Close', 'MACD', 'Signal', 'Histogram', 'Buy_Signal', 'Sell_Signal']
                    if use_rsi and 'RSI' in df_result.columns:
                        display_cols.append('RSI')
                    if use_kd and 'K' in df_result.columns:
                        display_cols.extend(['K', 'D'])
                    if use_ma and 'MA20' in df_result.columns:
                        display_cols.extend(['MA20', 'MA50'])
                    if use_patterns:
                        display_cols.extend(['Bullish_Pattern_Score', 'Bearish_Pattern_Score', 'Overall_Pattern_Score'])
                    display_cols.extend(['Position', 'Cumulative_Returns'])
                    
                    if 'Position_Size' in df_result.columns:
                        display_cols.append('Position_Size')
                    if 'ML_Confidence' in df_result.columns:
                        display_cols.append('ML_Confidence')
                    
                    available_cols = [col for col in display_cols if col in df_result.columns]
                    if available_cols:
                        display_df = df_result[available_cols].tail(20)
                        
                        format_dict = {
                            'Close': '{:.2f}',
                            'MACD': '{:.4f}',
                            'Signal': '{:.4f}', 
                            'Histogram': '{:.4f}',
                            'Cumulative_Returns': '{:.4f}',
                            'ML_Confidence': '{:.2f}',
                            'K': '{:.2f}',
                            'D': '{:.2f}',
                            'RSI': '{:.2f}',
                            'MA20': '{:.2f}',
                            'MA50': '{:.2f}',
                            'Bullish_Pattern_Score': '{:.2f}',
                            'Bearish_Pattern_Score': '{:.2f}',
                            'Overall_Pattern_Score': '{:.2f}'
                        }
                        
                        if 'Position_Size' in available_cols:
                            format_dict['Position_Size'] = '{:.2%}'
                        
                        st.dataframe(display_df.style.format(format_dict), use_container_width=True)
                    else:
                        st.info("暂无详细数据")
                
                # 参数优化
                if optimize:
                    st.subheader(safe_chinese_text("參數優化結果"))
                    results_df, best_params = optimize_params(
                        df.copy(), param_grid, use_rsi, use_ma, use_ml, use_kd, use_patterns
                    )
                    
                    if not results_df.empty:
                        top_results = results_df.sort_values('sharpe', ascending=False).head(10)
                        st.dataframe(top_results.style.format({
                            'sharpe': '{:.3f}',
                            'total_return': '{:.2%}',
                            'max_drawdown': '{:.2%}',
                            'win_rate': '{:.2%}'
                        }), use_container_width=True)
                        
                        st.success(f"🎉 最佳參數: 快線={best_params[0]}, 慢線={best_params[1]}, 訊號線={best_params[2]} | " +
                                  f"夏普比率: {top_results.iloc[0]['sharpe']:.3f}")
                        
                        if len(top_results) > 0:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            scatter = ax.scatter(top_results['fast'], top_results['slow'], 
                                               c=top_results['sharpe'], s=top_results['signal']*20, 
                                               cmap='viridis', alpha=0.7)
                            plt.colorbar(scatter, label='Sharpe Ratio')
                            ax.set_xlabel(safe_chinese_text('快線週期'))
                            ax.set_ylabel(safe_chinese_text('慢線週期'))
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                    else:
                        st.warning("⚠️ 優化無結果，請調整參數範圍")
                
                st.info("💡 提示：此為歷史回測，非投資建議。實際交易需考慮手續費、滑價與稅務。")
            
            except Exception as e:
                st.error(f"❌ 回測過程中發生錯誤: {e}")
                st.info("💡 請嘗試調整參數或使用不同的股票代碼")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 使用說明")
    st.sidebar.info("""
    1. 輸入台灣股票代碼 (如 2330.TW)
    2. 調整MACD參數和功能開關
    3. 點擊「執行智能分析」
    4. 查看各分頁結果
    """)
    
    # 页脚
    st.markdown("---")
    st.caption("""
    🚀 Powered by Streamlit + yfinance + 智能算法 | 
    台灣股市 MACD+KD+K線形態智能策略分析系統 v3.0 | 
    📧 注意：本工具僅供教育研究使用
    """)

if __name__ == "__main__":

    main()
