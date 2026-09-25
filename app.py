import sqlite3
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title='NOVA BTC TERMINAL', page_icon='₿', layout='wide')
BASE='https://www.okx.com/api/v5'; INST='BTC-USDT-SWAP'; DB='nova_terminal.db'

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:Inter,sans-serif}.stApp{background:radial-gradient(circle at 10% 0%,#09223a 0,transparent 28%),radial-gradient(circle at 90% 10%,#06251c 0,transparent 25%),linear-gradient(180deg,#04070b,#07111b 55%,#030509);color:#f5f7fa}.block-container{max-width:1450px;padding-top:1rem}.card,.hero,.signal{border:1px solid rgba(70,160,230,.2);border-radius:20px;background:linear-gradient(145deg,rgba(11,27,41,.96),rgba(4,10,16,.98));box-shadow:0 18px 55px rgba(0,0,0,.28),inset 0 1px rgba(255,255,255,.04);padding:16px;margin-bottom:14px}.hero{padding:22px;background:radial-gradient(circle at 20% 50%,rgba(0,245,139,.12),transparent 30%),linear-gradient(145deg,#0b2435,#040a10)}.brand{font-size:1.8rem;font-weight:800;letter-spacing:-.04em}.sub,.muted{color:#7f93a8;font-size:.8rem}.live{color:#00f58b;border:1px solid rgba(0,245,139,.3);background:rgba(0,245,139,.08);border-radius:999px;padding:5px 9px;font-size:.7rem;font-weight:800}.paper{color:#59b9ff;border:1px solid rgba(89,185,255,.25);background:rgba(89,185,255,.07);border-radius:999px;padding:5px 9px;font-size:.7rem;font-weight:800}.price{font-size:clamp(2.5rem,6vw,4.6rem);font-weight:800;letter-spacing:-.07em;line-height:1;margin:8px 0}.up{color:#00f58b}.down{color:#ff405b}.signal{min-height:220px;background:radial-gradient(circle at 80% 15%,rgba(0,245,139,.16),transparent 30%),linear-gradient(145deg,#092b25,#050d12)}.short{background:radial-gradient(circle at 80% 15%,rgba(255,64,91,.16),transparent 30%),linear-gradient(145deg,#32121b,#050d12)}.neutral{background:radial-gradient(circle at 80% 15%,rgba(70,170,255,.14),transparent 30%),linear-gradient(145deg,#0c2539,#050d12)}.badge{display:inline-block;padding:8px 14px;border-radius:12px;font-size:1.2rem;font-weight:800;color:#00f58b;background:rgba(0,245,139,.1);border:1px solid rgba(0,245,139,.3)}.short .badge{color:#ff405b;background:rgba(255,64,91,.1);border-color:rgba(255,64,91,.3)}.neutral .badge{color:#59b9ff;background:rgba(89,185,255,.1);border-color:rgba(89,185,255,.3)}.score{font-size:3.2rem;font-weight:800;margin-top:14px}.title{font-size:.9rem;font-weight:800;margin-bottom:8px}.stMetric{background:rgba(8,20,31,.8)!important}.stButton>button{border-radius:12px;font-weight:700;background:#102b40;border:1px solid rgba(70,160,230,.25);color:#fff}
</style>''', unsafe_allow_html=True)

s=requests.Session(); s.headers['User-Agent']='NOVA-BTC-Terminal/1.0'

def api(path,params=None):
    r=s.get(BASE+path,params=params or {},timeout=12); r.raise_for_status(); j=r.json()
    if str(j.get('code','0'))!='0': raise RuntimeError(j.get('msg','OKX error'))
    return j.get('data',[])

def candles(bar='1H',limit=300):
    rows=list(reversed(api('/market/candles',{'instId':INST,'bar':bar,'limit':str(limit)})))
    d=pd.DataFrame(rows,columns=['ts','o','h','l','c','vol','volCcy','volCcyQuote','confirm'])
    for c in ['o','h','l','c','vol']: d[c]=pd.to_numeric(d[c],errors='coerce')
    d['ts']=pd.to_datetime(d['ts'].astype('int64'),unit='ms',utc=True); return d.dropna()

def ticker():
    d=api('/market/ticker',{'instId':INST})[0]
    return {k:float(d[k]) for k in ['last','open24h','high24h','low24h','vol24h','bidPx','askPx']}

def funding(): return float(api('/public/funding-rate',{'instId':INST})[0]['fundingRate'])
def oi(): return float(api('/public/open-interest',{'instType':'SWAP','instId':INST})[0].get('oiCcy') or 0)
def book():
    d=api('/market/books',{'instId':INST,'sz':'30'})[0]

    # OKX orderbook levels can contain 4 fields depending on API response.
    # NOVA only needs price and size, so keep the first two fields safely.
    def parse_levels(levels):
        rows=[]
        for row in levels:
            if len(row) >= 2:
                rows.append([row[0], row[1]])
        frame=pd.DataFrame(rows,columns=['px','sz'])
        frame['px']=pd.to_numeric(frame['px'],errors='coerce')
        frame['sz']=pd.to_numeric(frame['sz'],errors='coerce')
        return frame.dropna().reset_index(drop=True)

    return parse_levels(d['bids']), parse_levels(d['asks'])
def trades():
    d=pd.DataFrame(api('/market/trades',{'instId':INST,'limit':'100'}))
    if d.empty:return d
    d['sz']=d['sz'].astype(float); d['ts']=pd.to_datetime(d['ts'].astype('int64'),unit='ms',utc=True); d['signed']=np.where(d.side.eq('buy'),d.sz,-d.sz); return d.sort_values('ts')

def ind(d):
    x=d.copy(); x['ema20']=x.c.ewm(span=20,adjust=False).mean(); x['ema50']=x.c.ewm(span=50,adjust=False).mean(); x['ema200']=x.c.ewm(span=200,adjust=False).mean()
    delta=x.c.diff(); g=delta.clip(lower=0).rolling(14).mean(); l=(-delta.clip(upper=0)).rolling(14).mean(); rs=g/l.replace(0,np.nan); x['rsi']=(100-100/(1+rs)).fillna(50)
    x['volma']=x.vol.rolling(20).mean(); x['vr']=x.vol/x.volma.replace(0,np.nan); return x

def signal(df,df1h,df4h,bids,asks,fr):
    a=ind(df).iloc[-1]; h=ind(df1h).iloc[-1]; q=ind(df4h).iloc[-1]; bull=bear=0; reasons=[]
    checks=[('Trend',a.ema20>a.ema50,22),('1H',h.c>h.ema50,18),('4H',q.c>q.ema50,15)]
    for n,up,w in checks:
        (lambda:None)(); bull+=w if up else 0; bear+=0 if up else w; reasons.append((n,'UP' if up else 'DOWN',w))
    r=a.rsi
    if 52<=r<=68: bull+=15; reasons.append(('RSI','HEALTHY',15))
    elif 32<=r<48: bear+=15; reasons.append(('RSI','WEAK',15))
    elif r>75: bear+=7; reasons.append(('RSI','OVERBOUGHT',7))
    else: reasons.append(('RSI','NEUTRAL',0))
    if a.vr>1.2:
        if a.c>=a.o: bull+=10; reasons.append(('Volume','BUY PRESSURE',10))
        else: bear+=10; reasons.append(('Volume','SELL PRESSURE',10))
    bi=(bids.head(15).sz.sum()-asks.head(15).sz.sum())/max(bids.head(15).sz.sum()+asks.head(15).sz.sum(),1)
    if bi>.08: bull+=12; reasons.append(('Orderbook','BID HEAVY',12))
    elif bi<-.08: bear+=12; reasons.append(('Orderbook','ASK HEAVY',12))
    else: reasons.append(('Orderbook','BALANCED',0))
    reasons.append(('Funding','CAUTION' if abs(fr)>.0003 else 'NORMAL',0))
    score=int(np.clip(50+abs(bull-bear)*.95,50,99)); sig='LONG' if bull>bear and bull>=48 else 'SHORT' if bear>bull and bear>=48 else 'NEUTRAL'
    return sig,score,r,bi,reasons

def layout(fig,h=320):
    fig.update_layout(height=h,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(3,12,20,.8)',font=dict(color='#c9d5e0',family='Inter'),margin=dict(l=8,r=8,t=28,b=8),xaxis=dict(gridcolor='rgba(130,170,205,.08)'),yaxis=dict(gridcolor='rgba(130,170,205,.08)'),legend=dict(bgcolor='rgba(0,0,0,0)',orientation='h'))
    return fig

def price_fig(d):
    x=ind(d); f=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.78,.22],vertical_spacing=.03)
    f.add_trace(go.Candlestick(x=x.ts,open=x.o,high=x.h,low=x.l,close=x.c,name='BTC',increasing_line_color='#00f58b',increasing_fillcolor='#00b96b',decreasing_line_color='#ff405b',decreasing_fillcolor='#b51f39'),row=1,col=1)
    for y,n,c in [('ema20','EMA 20','#ffad33'),('ema50','EMA 50','#22aaff'),('ema200','EMA 200','#b878ff')]: f.add_trace(go.Scatter(x=x.ts,y=x[y],name=n,line=dict(color=c,width=2)),row=1,col=1)
    f.add_trace(go.Bar(x=x.ts,y=x.vol,name='Volume',marker_color=np.where(x.c>=x.o,'#00c97b','#e83450'),opacity=.55),row=2,col=1); f.update_xaxes(rangeslider_visible=False); return layout(f,590)

def donut(labels,values,colors,text):
    f=go.Figure(go.Pie(labels=labels,values=values,hole=.68,textinfo='none',marker=dict(colors=colors,line=dict(color='#07101a',width=3))))
    f.update_layout(height=230,paper_bgcolor='rgba(0,0,0,0)',font=dict(color='#c9d5e0'),margin=dict(l=0,r=0,t=5,b=5),legend=dict(font=dict(size=10)),annotations=[dict(text=text,x=.5,y=.5,showarrow=False,font=dict(size=22,color='#fff'))]); return f

def book_fig(b,a):
    f=go.Figure(); b=b.head(15).sort_values('px'); a=a.head(15).sort_values('px'); f.add_trace(go.Bar(x=b.px,y=b.sz,name='Bids',marker_color='rgba(0,245,139,.7)')); f.add_trace(go.Bar(x=a.px,y=a.sz,name='Asks',marker_color='rgba(255,64,91,.7)')); return layout(f,300)

def cvd_fig(t):
    if t.empty:return None
    x=t.copy(); x['cvd']=x.signed.cumsum(); f=go.Figure(go.Scatter(x=x.ts,y=x.cvd,mode='lines',fill='tozeroy',fillcolor='rgba(0,217,255,.08)',line=dict(color='#00d9ff',width=2.5),name='CVD')); return layout(f,280)

def timing_fig(d):
    x=d.copy(); x['hour']=x.ts.dt.hour; x['ret']=x.c.pct_change(); g=x.groupby('hour').agg(ret=('ret','mean'),vol=('vol','mean')).reindex(range(24));
    f=go.Figure(go.Heatmap(z=[g.ret.fillna(0).values*100],x=list(range(24)),y=['Avg return %'],colorscale=[[0,'#ff405b'],[.5,'#07121d'],[1,'#00f58b']],colorbar=dict(title='%'))); return layout(f,220)

# Load
try:
    t=ticker(); d=candles('1H'); d15=candles('15m'); d4=candles('4H'); fr=funding(); oiv=oi(); bids,asks=book(); tr=trades()
except Exception as e:
    st.error(f'OKX data kon niet worden geladen: {e}'); st.stop()

chg=(t['last']-t['open24h'])/t['open24h']*100; spr=t['askPx']-t['bidPx']; bi=(bids.head(15).sz.sum()-asks.head(15).sz.sum())/max(bids.head(15).sz.sum()+asks.head(15).sz.sum(),1)
sig,score,rsi,bi,reasons=signal(d15,d,d4,bids,asks,fr)
reg='SHOCK' if ind(d15).iloc[-1].vr>2 else 'EXPANSION' if ind(d15).iloc[-1].vr>1.4 else 'TREND' if abs(ind(d15).iloc[-1].ema20-ind(d15).iloc[-1].ema50)/t['last']>.004 else 'RANGE'

st.markdown(f'<div><span class="brand">₿ NOVA BTC TERMINAL</span> <span class="live">● LIVE</span> <span class="paper">PAPER ONLY</span><div class="sub">Live Market Intelligence · OKX · BTC-USDT-SWAP · {datetime.now(timezone.utc).strftime("%H:%M:%S UTC")}</div></div>',unsafe_allow_html=True)

pc='up' if chg>=0 else 'down'; ar='▲' if chg>=0 else '▼'
st.markdown(f'<div class="hero"><div class="sub">BTC / USDT · OKX PERPETUAL</div><div class="price">${t["last"]:,.2f}</div><div class="{pc}">{ar} {chg:+.2f}% · 24H</div></div>',unsafe_allow_html=True)

cols=st.columns(5); vals=[('24H HIGH',f'${t["high24h"]:,.0f}'),('24H LOW',f'${t["low24h"]:,.0f}'),('24H VOLUME',f'{t["vol24h"]:,.0f}'),('FUNDING',f'{fr*100:+.4f}%'),('OPEN INTEREST',f'{oiv:,.1f} BTC')]
for c,(a,b) in zip(cols,vals): c.metric(a,b)

c1,c2,c3=st.columns([1.15,1,1])
with c1:
    cls='signal' if sig=='LONG' else 'signal short' if sig=='SHORT' else 'signal neutral'; badge='🚀 LONG' if sig=='LONG' else '🔻 SHORT' if sig=='SHORT' else '◆ NEUTRAL'
    st.markdown(f'<div class="{cls}"><div class="sub">NOVA SIGNAL</div><div style="margin-top:12px"><span class="badge">{badge}</span></div><div class="score">{score}<span style="font-size:1.1rem;color:#71869a"> / 100</span></div><div class="sub">MODEL CONVICTION · {reg}</div><div style="margin-top:15px" class="muted">RSI {rsi:.1f} · Book {bi*100:+.1f}% · Spread ${spr:.2f}</div></div>',unsafe_allow_html=True)
with c2:
    st.markdown('<div class="card"><div class="title">3D-STYLE CONVICTION</div>',unsafe_allow_html=True)
    st.plotly_chart(go.Figure(go.Indicator(mode='gauge+number',value=score,number={'font':{'size':42,'color':'#fff'}},gauge={'axis':{'range':[0,100],'tickcolor':'#5d7084'},'bar':{'color':'#00f58b' if sig=='LONG' else '#ff405b' if sig=='SHORT' else '#4daeff'},'bgcolor':'#0b1622','steps':[{'range':[0,35],'color':'#35131b'},{'range':[35,65],'color':'#152433'},{'range':[65,100],'color':'#092a20'}]})).update_layout(height=220,paper_bgcolor='rgba(0,0,0,0)',margin=dict(l=8,r=8,t=8,b=0)),use_container_width=True,config={'displayModeBar':False})
    st.markdown('</div>',unsafe_allow_html=True)
with c3:
    bull=sum(x[2] for x in reasons if x[1] in ['UP','HEALTHY','BUY PRESSURE','BID HEAVY']); bear=sum(x[2] for x in reasons if x[1] in ['DOWN','WEAK','OVERBOUGHT','SELL PRESSURE','ASK HEAVY']); neutral=max(10,100-bull-bear)
    st.markdown('<div class="card"><div class="title">SIGNAL COMPOSITION</div>',unsafe_allow_html=True); st.plotly_chart(donut(['Bullish','Neutral','Bearish'],[bull,neutral,bear],['#00f58b','#2699ff','#ff405b'],str(score)),use_container_width=True,config={'displayModeBar':False}); st.markdown('</div>',unsafe_allow_html=True)

st.markdown('<div class="card"><div class="title">BTC PRICE ACTION · EMA 20 / 50 / 200 · VOLUME</div>',unsafe_allow_html=True)
tf=st.selectbox('Timeframe',['5m','15m','1H','4H','1D'],index=2,label_visibility='collapsed'); st.plotly_chart(price_fig(candles(tf)),use_container_width=True,config={'displayModeBar':False,'responsive':True}); st.markdown('</div>',unsafe_allow_html=True)

a,b,c=st.columns(3)
with a:
    st.markdown('<div class="card"><div class="title">ORDERBOOK DEPTH</div>',unsafe_allow_html=True); st.plotly_chart(book_fig(bids,asks),use_container_width=True,config={'displayModeBar':False}); st.markdown('</div>',unsafe_allow_html=True)
with b:
    st.markdown('<div class="card"><div class="title">CVD · RECENT TAPE</div>',unsafe_allow_html=True); f=cvd_fig(tr); st.plotly_chart(f,use_container_width=True,config={'displayModeBar':False}) if f else st.caption('Geen tape-data'); st.markdown('</div>',unsafe_allow_html=True)
with c:
    st.markdown('<div class="card"><div class="title">MARKET BIAS</div>',unsafe_allow_html=True); lp=np.clip(50+bi*100,1,99); st.plotly_chart(donut(['Bid pressure','Ask pressure'],[lp,100-lp],['#00f58b','#ff405b'],f'{lp:.0f}%'),use_container_width=True,config={'displayModeBar':False}); st.markdown('</div>',unsafe_allow_html=True)

a,b,c=st.columns(3)
with a:
    st.markdown('<div class="card"><div class="title">BTC TIMING HEATMAP · UTC</div>',unsafe_allow_html=True); st.plotly_chart(timing_fig(d),use_container_width=True,config={'displayModeBar':False}); st.caption('Historische 1H sample; indicatief, geen voorspelling.'); st.markdown('</div>',unsafe_allow_html=True)
with b:
    st.markdown('<div class="card"><div class="title">TOP FACTORS</div>',unsafe_allow_html=True)
    for n,state,w in reasons:
        color='#00f58b' if state in ['UP','HEALTHY','BUY PRESSURE','BID HEAVY'] else '#ff405b' if state in ['DOWN','WEAK','OVERBOUGHT','SELL PRESSURE','ASK HEAVY'] else '#59b9ff'
        st.markdown(f'<div style="display:flex;justify-content:space-between;padding:9px 0;border-bottom:1px solid rgba(255,255,255,.05)"><span>{n}</span><b style="color:{color}">{state} {"+"+str(w) if w else ""}</b></div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)
with c:
    st.markdown('<div class="card"><div class="title">KEY MARKET DATA</div>',unsafe_allow_html=True)
    for n,v in [('BTC PRICE',f'${t["last"]:,.2f}'),('24H CHANGE',f'{chg:+.2f}%'),('BID',f'${t["bidPx"]:,.2f}'),('ASK',f'${t["askPx"]:,.2f}'),('SPREAD',f'${spr:.2f}'),('BOOK IMBALANCE',f'{bi*100:+.1f}%')]: st.markdown(f'<div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(255,255,255,.05)"><span class="muted">{n}</span><b>{v}</b></div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

st.caption('NOVA BTC TERMINAL · Paper trading analytics · Geen echte orders · OKX public market data')
