#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock Investment Analyzer
วิเคราะห์หุ้นและให้คำแนะนำการลงทุน

Requirements:
pip install streamlit yfinance pandas numpy matplotlib plotly ta-lib
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

@dataclass
class InvestmentZone:
    name: str
    price_range: Tuple[float, float]
    color: str
    recommendation: str
    risk_level: str
    allocation: str

class StockAnalyzer:
    def __init__(self):
        self.data = None
        self.symbol = None
        
    def fetch_data(self, symbol: str, period: str = "3mo") -> pd.DataFrame:
        """ดึงข้อมูลหุ้นจาก Yahoo Finance"""
        try:
            self.symbol = symbol.upper()
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=period)
            
            if self.data.empty:
                raise ValueError(f"ไม่พบข้อมูลสำหรับ {symbol}")
                
            return self.data
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {str(e)}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self) -> Dict:
        """คำนวณตัวชี้วัดทางเทคนิค"""
        if self.data is None or self.data.empty:
            return {}
        
        # Moving Averages
        self.data['MA7'] = self.data['Close'].rolling(window=7).mean()
        self.data['MA21'] = self.data['Close'].rolling(window=21).mean()
        self.data['MA50'] = self.data['Close'].rolling(window=50).mean()
        
        # RSI (Relative Strength Index)
        self.data['RSI'] = self.calculate_rsi(self.data['Close'])
        
        # Bollinger Bands
        ma20 = self.data['Close'].rolling(window=20).mean()
        std20 = self.data['Close'].rolling(window=20).std()
        self.data['BB_Upper'] = ma20 + (2 * std20)
        self.data['BB_Lower'] = ma20 - (2 * std20)
        
        # Support และ Resistance
        high_period = 20
        low_period = 20
        
        recent_high = self.data['High'].rolling(window=high_period).max().iloc[-1]
        recent_low = self.data['Low'].rolling(window=low_period).min().iloc[-1]
        
        # คำนวณ Support และ Resistance จากข้อมูลย้อนหลัง
        resistance = recent_high * 0.98  # ลด 2% จาก high
        support = recent_low * 1.02      # เพิ่ม 2% จาก low
        
        current_price = self.data['Close'].iloc[-1]
        
        return {
            'current_price': current_price,
            'ma7': self.data['MA7'].iloc[-1],
            'ma21': self.data['MA21'].iloc[-1],
            'ma50': self.data['MA50'].iloc[-1],
            'rsi': self.data['RSI'].iloc[-1],
            'support': support,
            'resistance': resistance,
            'bb_upper': self.data['BB_Upper'].iloc[-1],
            'bb_lower': self.data['BB_Lower'].iloc[-1],
            'volume': self.data['Volume'].iloc[-1],
            'trend': self.determine_trend()
        }
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """คำนวณ RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def determine_trend(self) -> str:
        """กำหนดทิศทางเทรนด์"""
        if len(self.data) < 21:
            return "ไม่สามารถระบุได้"
        
        ma7 = self.data['MA7'].iloc[-1]
        ma21 = self.data['MA21'].iloc[-1]
        current_price = self.data['Close'].iloc[-1]
        
        if ma7 > ma21 and current_price > ma7:
            return "Bullish (ขาขึ้น)"
        elif ma7 < ma21 and current_price < ma7:
            return "Bearish (ขาลง)"
        else:
            return "Sideways (ไซด์เวย์)"
    
    def create_investment_zones(self, indicators: Dict) -> List[InvestmentZone]:
        """สร้างโซนการลงทุน"""
        current_price = indicators['current_price']
        support = indicators['support']
        resistance = indicators['resistance']
        
        zones = [
            InvestmentZone(
                name="🟢 Strong Buy Zone",
                price_range=(support, support * 1.05),
                color="#22c55e",
                recommendation="ลงทุนหนัก",
                risk_level="ต่ำ",
                allocation="40-50% ของพอร์ต"
            ),
            InvestmentZone(
                name="🔵 Buy Zone", 
                price_range=(support * 1.05, current_price * 0.95),
                color="#3b82f6",
                recommendation="ลงทุนปานกลาง",
                risk_level="ปานกลาง",
                allocation="20-30% ของพอร์ต"
            ),
            InvestmentZone(
                name="🟡 Hold Zone",
                price_range=(current_price * 0.95, current_price * 1.05),
                color="#f59e0b",
                recommendation="รอจังหวะ",
                risk_level="ปานกลาง",
                allocation="รอโอกาสที่ดีกว่า"
            ),
            InvestmentZone(
                name="🔴 Sell Zone",
                price_range=(resistance * 0.95, resistance),
                color="#ef4444",
                recommendation="พิจารณาขาย",
                risk_level="สูง",
                allocation="ตัดขาดทุน/ตรึงกำไร"
            )
        ]
        
        return zones
    
    def get_recommendation(self, indicators: Dict) -> Dict:
        """ให้คำแนะนำการลงทุน"""
        current_price = indicators['current_price']
        support = indicators['support']
        resistance = indicators['resistance']
        rsi = indicators['rsi']
        trend = indicators['trend']
        
        # คำนวณคำแนะนำตามราคาและตัวชี้วัด
        if current_price <= support * 1.05 and rsi < 30:
            action = "Strong Buy"
            reason = "ราคาอยู่ในโซน Support และ RSI แสดง Oversold"
        elif current_price <= support * 1.15 and rsi < 50:
            action = "Buy"
            reason = "ราคาใกล้ Support และ RSI ยังไม่สูงมาก"
        elif current_price >= resistance * 0.95 or rsi > 70:
            action = "Sell"
            reason = "ราคาใกล้ Resistance หรือ RSI แสดง Overbought"
        else:
            action = "Hold"
            reason = "ราคาอยู่ในช่วง Neutral"
        
        return {
            'action': action,
            'reason': reason,
            'target_price': resistance,
            'stop_loss': support * 0.95,
            'confidence': self.calculate_confidence(indicators)
        }
    
    def calculate_confidence(self, indicators: Dict) -> int:
        """คำนวณระดับความเชื่อมั่น (0-100)"""
        confidence = 50  # เริ่มต้นที่ 50%
        
        # ปรับตามเทรนด์
        if "Bullish" in indicators['trend']:
            confidence += 20
        elif "Bearish" in indicators['trend']:
            confidence -= 15
        
        # ปรับตาม RSI
        rsi = indicators['rsi']
        if 30 <= rsi <= 70:
            confidence += 10
        elif rsi < 20 or rsi > 80:
            confidence += 15  # RSI extreme มักให้สัญญาณดี
        
        return max(10, min(95, confidence))

def create_price_chart(analyzer: StockAnalyzer, indicators: Dict):
    """สร้างกราฟราคาหุ้น"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[f'{analyzer.symbol} - กราฟราคาและตัวชี้วัด', 'RSI'],
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05
    )
    
    # กราฟราคา
    fig.add_trace(
        go.Candlestick(
            x=analyzer.data.index,
            open=analyzer.data['Open'],
            high=analyzer.data['High'], 
            low=analyzer.data['Low'],
            close=analyzer.data['Close'],
            name='ราคา'
        ),
        row=1, col=1
    )
    
    # Moving Averages
    fig.add_trace(
        go.Scatter(
            x=analyzer.data.index,
            y=analyzer.data['MA7'],
            name='MA7',
            line=dict(color='orange', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=analyzer.data.index,
            y=analyzer.data['MA21'],
            name='MA21',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Support และ Resistance
    fig.add_hline(
        y=indicators['support'],
        line_dash="dash",
        line_color="green",
        annotation_text="Support",
        row=1, col=1
    )
    
    fig.add_hline(
        y=indicators['resistance'],
        line_dash="dash", 
        line_color="red",
        annotation_text="Resistance",
        row=1, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(
            x=analyzer.data.index,
            y=analyzer.data['RSI'],
            name='RSI',
            line=dict(color='purple', width=2)
        ),
        row=2, col=1
    )
    
    # RSI เส้นอ้างอิง
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
    
    fig.update_layout(
        title=f'{analyzer.symbol} - การวิเคราะห์ทางเทคนิค',
        height=700,
        showlegend=True
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="📈 Stock Investment Analyzer By Chantawat Wongasa",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Stock Investment Analyzer POP")
    st.markdown("### วิเคราะห์หุ้นและให้คำแนะนำการลงทุนแบบครบครัน")
    
    # Sidebar สำหรับการตั้งค่า
    with st.sidebar:
        st.header("🔧 การตั้งค่า")
        
        symbol = st.text_input(
            "รหัสหุ้น:",
            value="AAPL",
            help="ใส่รหัสหุ้น เช่น AAPL, GOOGL, MSFT"
        ).upper()
        
        period = st.selectbox(
            "ช่วงเวลา:",
            ["1mo", "3mo", "6mo", "1y", "2y"],
            index=1,
            help="เลือกช่วงเวลาที่ต้องการวิเคราะห์"
        )
        
        analyze_button = st.button("🔍 วิเคราะห์หุ้น", type="primary")
    
    if analyze_button and symbol:
        with st.spinner(f"กำลังดึงข้อมูล {symbol}..."):
            analyzer = StockAnalyzer()
            data = analyzer.fetch_data(symbol, period)
            
            if not data.empty:
                # คำนวณตัวชี้วัด
                indicators = analyzer.calculate_technical_indicators()
                
                # แสดงข้อมูลหลัก
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "ราคาปัจจุบัน",
                        f"${indicators['current_price']:.2f}",
                        delta=f"{((indicators['current_price'] - analyzer.data['Close'].iloc[-2]) / analyzer.data['Close'].iloc[-2] * 100):.2f}%"
                    )
                
                with col2:
                    trend_color = "🟢" if "Bullish" in indicators['trend'] else "🔴" if "Bearish" in indicators['trend'] else "🟡"
                    st.metric("เทรนด์", f"{trend_color} {indicators['trend']}")
                
                with col3:
                    rsi_color = "🟢" if indicators['rsi'] < 30 else "🔴" if indicators['rsi'] > 70 else "🟡"
                    st.metric("RSI", f"{rsi_color} {indicators['rsi']:.1f}")
                
                with col4:
                    recommendation = analyzer.get_recommendation(indicators)
                    action_color = {"Strong Buy": "🟢", "Buy": "🔵", "Hold": "🟡", "Sell": "🔴"}
                    st.metric("คำแนะนำ", f"{action_color.get(recommendation['action'], '⚪')} {recommendation['action']}")
                
                # กราฟราคา
                st.plotly_chart(
                    create_price_chart(analyzer, indicators),
                    use_container_width=True
                )
                
                # ข้อมูลการวิเคราะห์
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 ตัวชี้วัดทางเทคนิค")
                    
                    tech_data = {
                        "ตัวชี้วัด": ["MA7", "MA21", "MA50", "RSI", "Support", "Resistance"],
                        "ค่า": [
                            f"${indicators['ma7']:.2f}",
                            f"${indicators['ma21']:.2f}", 
                            f"${indicators['ma50']:.2f}",
                            f"{indicators['rsi']:.1f}",
                            f"${indicators['support']:.2f}",
                            f"${indicators['resistance']:.2f}"
                        ]
                    }
                    
                    st.dataframe(tech_data, hide_index=True)
                
                with col2:
                    st.subheader("💡 คำแนะนำการลงทุน")
                    
                    recommendation = analyzer.get_recommendation(indicators)
                    
                    st.markdown(f"""
                    **คำแนะนำ:** {recommendation['action']}  
                    **เหตุผล:** {recommendation['reason']}  
                    **Target Price:** ${recommendation['target_price']:.2f}  
                    **Stop Loss:** ${recommendation['stop_loss']:.2f}  
                    **ความเชื่อมั่น:** {recommendation['confidence']}%
                    """)
                
                # โซนการลงทุน
                st.subheader("🎯 โซนการลงทุน")
                
                zones = analyzer.create_investment_zones(indicators)
                
                for zone in zones:
                    with st.expander(f"{zone.name} (${zone.price_range[0]:.2f} - ${zone.price_range[1]:.2f})"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"**คำแนะนำ:** {zone.recommendation}")
                        
                        with col2:
                            st.write(f"**ความเสี่ยง:** {zone.risk_level}")
                        
                        with col3:
                            st.write(f"**การจัดสรร:** {zone.allocation}")
                
                # สรุปการวิเคราะห์
                st.subheader("📋 สรุปการวิเคราะห์")
                
                current_price = indicators['current_price']
                support = indicators['support']
                resistance = indicators['resistance']
                
                # กำหนดสถานะปัจจุบัน
                if current_price <= support * 1.05:
                    status = "🟢 อยู่ในโซนซื้อที่ดี"
                    advice = "แนะนำให้พิจารณาลงทุน"
                elif current_price >= resistance * 0.95:
                    status = "🔴 อยู่ในโซนราคาสูง" 
                    advice = "ควรระมัดระวัง อาจพิจารณาขาย"
                else:
                    status = "🟡 อยู่ในโซนกลาง"
                    advice = "ควรรอจังหวะที่ดีกว่า"
                
                st.info(f"""
                **สถานะปัจจุบัน:** {status}  
                **คำแนะนำ:** {advice}  
                **ช่วงราคาที่น่าสนใจ:** ${support:.2f} - ${support * 1.1:.2f}  
                **ควรขายที่:** ${resistance * 0.95:.2f} - ${resistance:.2f}
                """)
    
    # คำเตือน
    st.warning("""
    ⚠️ **คำเตือนความเสี่ยง**  
    ข้อมูลการวิเคราะห์นี้เป็นเพียงเครื่องมือช่วยในการตัดสินใจเท่านั้น ไม่ใช่คำแนะนำในการลงทุน  
    กรุณาศึกษาข้อมูลเพิ่มเติมและปรึกษาผู้เชี่ยวชาญก่อนตัดสินใจลงทุน  
    การลงทุนมีความเสี่ยง ควรลงทุนในสิ่งที่เข้าใจและสามารถรับความเสี่ยงได้
    """)

if __name__ == "__main__":
    main()