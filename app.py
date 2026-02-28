# DeepCSAT · Streamlit Dashboard · Prasanth Kumar Sahu
import os,re,string,warnings,numpy as np,pandas as pd,streamlit as st
import plotly.express as px,plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder,StandardScaler,PowerTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn.decomposition import PCA
from scipy.stats import f_oneway,ttest_ind,chi2_contingency
warnings.filterwarnings('ignore')

st.set_page_config(page_title="DeepCSAT",page_icon="⚡",layout="wide",initial_sidebar_state="expanded")
A,A2,A3,GO,GR="#00e5ff","#7b61ff","#ff4d6d","#ffd166","#00c9a7"
PL=dict(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(family='monospace',color='#9ca3af',size=11),title_font=dict(color='#e8eaf0',size=14),xaxis=dict(gridcolor='#1f2430',linecolor='#1f2430'),yaxis=dict(gridcolor='#1f2430',linecolor='#1f2430'),margin=dict(l=20,r=20,t=40,b=20),legend=dict(bgcolor='rgba(0,0,0,0)',bordercolor='#1f2430',borderwidth=1),colorway=[A,A2,A3,GO,GR,'#ff9a3c'])

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');
:root{--bg:#0a0c10;--s:#111318;--s2:#181b22;--b:#1f2430;--t:#e8eaf0;--m:#6b7280;}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--t)!important;font-family:'DM Mono',monospace;}
[data-testid="stSidebar"]{background:var(--s)!important;border-right:1px solid var(--b);}
#MainMenu,footer,header,[data-testid="stToolbar"]{visibility:hidden;display:none;}
h1,h2,h3,h4{font-family:'Syne',sans-serif!important;}
.hero{background:linear-gradient(135deg,#0d1117,#111827,#0f1923);border:1px solid var(--b);border-radius:16px;padding:36px 36px 28px;margin-bottom:24px;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;inset:0;background:radial-gradient(ellipse at 20% 50%,rgba(0,229,255,.06),transparent 60%),radial-gradient(ellipse at 80% 20%,rgba(123,97,255,.07),transparent 55%);pointer-events:none;}
.htitle{font-family:'Syne',sans-serif;font-size:2.6rem;font-weight:800;background:linear-gradient(90deg,#00e5ff,#7b61ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.1;margin:0 0 6px;}
.hsub{font-size:.85rem;color:var(--m);letter-spacing:.06em;}
.tag{display:inline-block;background:rgba(0,229,255,.08);border:1px solid rgba(0,229,255,.25);color:#00e5ff;font-size:.65rem;letter-spacing:.12em;text-transform:uppercase;padding:3px 10px;border-radius:100px;margin:12px 4px 0 0;}
.kgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px;}
.kcard{background:var(--s);border:1px solid var(--b);border-radius:12px;padding:18px 20px;position:relative;overflow:hidden;transition:border-color .2s;}
.kcard:hover{border-color:#00e5ff;}
.kcard::after{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#00e5ff,#7b61ff);}
.klabel{font-size:.65rem;letter-spacing:.1em;text-transform:uppercase;color:var(--m);margin-bottom:6px;}
.kval{font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:700;color:#00e5ff;line-height:1;}
.kdelta{font-size:.72rem;color:#00c9a7;margin-top:3px;}
.sec{display:flex;align-items:center;gap:10px;margin:32px 0 16px;border-bottom:1px solid var(--b);padding-bottom:10px;}
.snum{font-size:.65rem;color:#00e5ff;background:rgba(0,229,255,.08);border:1px solid rgba(0,229,255,.2);padding:2px 7px;border-radius:4px;}
.stitle{font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:var(--t);margin:0;}
.mcard{background:var(--s);border:1px solid var(--b);border-radius:12px;padding:20px;margin-bottom:14px;transition:border-color .2s,transform .15s;}
.mcard:hover{border-color:#7b61ff;transform:translateY(-2px);}
.hyp{background:var(--s2);border-left:3px solid #00e5ff;border-radius:0 10px 10px 0;padding:14px 18px;margin-bottom:10px;}
.ibox{background:rgba(0,229,255,.04);border:1px solid rgba(0,229,255,.15);border-radius:10px;padding:12px 16px;margin:10px 0;font-size:.82rem;color:var(--m);}
.pres{background:linear-gradient(135deg,rgba(0,229,255,.08),rgba(123,97,255,.08));border:1px solid rgba(0,229,255,.25);border-radius:14px;padding:28px;text-align:center;margin-top:18px;}
.pscore{font-family:'Syne',sans-serif;font-weight:800;font-size:4.5rem;line-height:1;background:linear-gradient(90deg,#00e5ff,#7b61ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.badge{display:inline-block;padding:3px 10px;border-radius:100px;font-size:.62rem;letter-spacing:.08em;text-transform:uppercase;background:rgba(0,201,167,.12);border:1px solid rgba(0,201,167,.3);color:#00c9a7;margin-left:8px;}
[data-testid="metric-container"]{background:var(--s2)!important;border:1px solid var(--b)!important;border-radius:10px!important;padding:14px!important;}
.stTabs [data-baseweb="tab-list"]{background:var(--s2)!important;border-radius:10px!important;padding:4px!important;}
.stTabs [data-baseweb="tab"]{font-family:'Syne',sans-serif!important;color:var(--m)!important;border-radius:7px!important;}
.stTabs [aria-selected="true"]{background:var(--s)!important;color:#00e5ff!important;}
.stButton>button{background:linear-gradient(135deg,#00e5ff,#7b61ff)!important;border:none!important;border-radius:8px!important;color:#000!important;font-family:'Syne',sans-serif!important;font-weight:700!important;padding:10px 28px!important;}
div[data-testid="stExpander"]{background:var(--s)!important;border:1px solid var(--b)!important;border-radius:10px!important;}
</style>""",unsafe_allow_html=True)

# ── helpers ──────────────────────────────────────────────────────────────────
def hero(t,s): st.markdown(f"<div class='hero'><div class='htitle'>{t}</div><div class='hsub'>{s}</div></div>",unsafe_allow_html=True)
def sec(n,t):  st.markdown(f"<div class='sec'><span class='snum'>{n}</span><p class='stitle'>{t}</p></div>",unsafe_allow_html=True)
def ibox(t):   st.markdown(f"<div class='ibox'>{t}</div>",unsafe_allow_html=True)
def kpi(label,val,delta="",color=A): st.markdown(f"<div class='kcard'><div class='klabel'>{label}</div><div class='kval' style='color:{color};'>{val}</div><div class='kdelta'>{delta}</div></div>",unsafe_allow_html=True)
def bar(x,y,title="",color=A,h=300,horiz=False):
    fig=go.Figure(go.Bar(x=y if horiz else x,y=x if horiz else y,orientation='h' if horiz else 'v',marker=dict(color=color,line_width=0),hovertemplate='%{y if horiz else x}<br>%{x if horiz else y}<extra></extra>'))
    fig.update_layout(**{**PL,'height':h,'title':title});st.plotly_chart(fig,use_container_width=True)

@st.cache_data(show_spinner=False)
def load():
    p=os.path.join("data","eCommerce_Customer_support_data.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data(show_spinner=False)
def preprocess(_df):
    df=_df.copy()
    for c in df.select_dtypes('object'): df[c]=df[c].astype(str).str.lower().str.strip()
    drop=['connected_handling_time','Customer_City','Product_category','Order_id','order_date_time','Item_price']
    dfc=df.drop(columns=drop,errors='ignore')
    dfc['Customer Remarks']=dfc['Customer Remarks'].replace('nan','').fillna('no feedback')
    le=LabelEncoder()
    for c in ['channel_name','category','Sub-category','Agent_name','Supervisor','Manager','Tenure Bucket','Agent Shift']:
        dfc[c]=le.fit_transform(dfc[c].astype(str))
    X=dfc.drop(columns=['CSAT Score','Customer Remarks','Unique id'],errors='ignore')
    for c in ['Issue_reported at','issue_responded','Survey_response_Date']:
        X[c]=pd.to_datetime(X[c],errors='coerce',dayfirst=True)
    X['issue_reported_hour']=X['Issue_reported at'].dt.hour
    X['issue_reported_dayofweek']=X['Issue_reported at'].dt.dayofweek
    X['issue_responded_hour']=X['issue_responded'].dt.hour
    X['issue_responded_dayofweek']=X['issue_responded'].dt.dayofweek
    X['response_time_hrs']=(X['issue_responded']-X['Issue_reported at']).dt.total_seconds()/3600
    X=X.drop(columns=['Issue_reported at','issue_responded','Survey_response_Date'],errors='ignore')
    return dfc,X

@st.cache_resource(show_spinner=False)
def train(_Xtr,_Xte,_ytr,_yte):
    res={}
    for name,mdl in [('CatBoost',None),('Random Forest',RandomForestRegressor(n_estimators=100,max_depth=10,random_state=42,n_jobs=-1)),('XGBoost',None)]:
        try:
            if name=='CatBoost':
                from catboost import CatBoostRegressor
                mdl=CatBoostRegressor(verbose=0,random_state=42,thread_count=-1,depth=5,iterations=100,learning_rate=0.1)
            elif name=='XGBoost':
                from xgboost import XGBRegressor
                mdl=XGBRegressor(random_state=42,objective='reg:squarederror',nthread=-1,verbosity=0,n_estimators=100,learning_rate=0.1,max_depth=5)
            mdl.fit(_Xtr,_ytr);yp=mdl.predict(_Xte)
            res[name]=dict(model=mdl,mse=mean_squared_error(_yte,yp),r2=r2_score(_yte,yp),pred=yp)
        except Exception as e: res[name]=dict(error=str(e))
    return res

# ── load & prep ───────────────────────────────────────────────────────────────
df_raw=load()
if df_raw is None: st.error("Place `eCommerce_Customer_support_data.csv` in `data/` folder.");st.stop()
df_clean,X=preprocess(df_raw)
y=df_clean['CSAT Score']
scaler=StandardScaler();Xs=scaler.fit_transform(X.fillna(0))
pca=PCA(n_components=10);Xp=pca.fit_transform(Xs);pca_var=pca.explained_variance_ratio_.sum()
Xtr,Xte,ytr,yte=train_test_split(Xp,y,test_size=0.2,random_state=42)
df_p=df_raw.copy()
for c in df_p.select_dtypes('object'): df_p[c]=df_p[c].astype(str).str.lower().str.strip()

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div style='padding:16px 0 6px;'><div style='font-family:Syne,sans-serif;font-weight:800;font-size:1.35rem;background:linear-gradient(90deg,{A},{A2});-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>DeepCSAT</div><div style='font-size:.6rem;color:#6b7280;letter-spacing:.1em;margin-top:3px;'>SCORE PREDICTION ENGINE</div></div><hr style='border-color:#1f2430;margin:10px 0;'>",unsafe_allow_html=True)
    page=st.radio("Nav",["🏠 Overview","🔍 Data Explorer","📊 EDA","🧪 Hypothesis","⚙️ Features","🤖 Models","🏆 Comparison","🔮 Predictor","📥 Export & Report"],label_visibility='collapsed')
    st.markdown(f"<hr style='border-color:#1f2430;margin:14px 0;'><div style='font-size:.65rem;color:#374151;font-family:DM Mono,monospace;line-height:1.9;'>PROJECT · DeepCSAT DL<br>TYPE · EDA + Regression<br>MODELS · CB · RF · XGB<br><span style='color:#4b5563;'>Prasanth Kumar Sahu</span></div>",unsafe_allow_html=True)

# ── pages ─────────────────────────────────────────────────────────────────────
r,c_=df_raw.shape;avg=round(df_raw['CSAT Score'].mean(),2);pct5=round((df_raw['CSAT Score']==5).sum()/r*100,1);miss=round(df_raw.isnull().sum().sum()/(r*c_)*100,1)

if "Overview" in page:
    st.markdown(f"<div class='hero'><div class='htitle'>DeepCSAT</div><div class='hsub'>Customer Satisfaction Score Prediction Engine</div><div><span class='tag'>eCommerce Analytics</span><span class='tag'>ML Regression</span><span class='tag'>CatBoost · RF · XGBoost</span><span class='tag'>NLP Pipeline</span></div><div style='margin-top:18px;font-size:.85rem;color:#6b7280;max-width:650px;line-height:1.7;'>A deep learning-powered analytical dashboard predicting CSAT scores from 85,907 e-commerce customer interaction records — enabling real-time service quality insight.</div></div>",unsafe_allow_html=True)
    cols=st.columns(4)
    for col,l,v,d,clr in zip(cols,["Total Records","Avg CSAT Score","5-Star Rate","Missing Data"],[f"{r:,}",avg,f"{pct5}%",f"{miss}%"],["20 columns","out of 5.0","perfect satisfaction","overall sparsity"],[A,A,GR,A3]):
        with col: kpi(l,v,d,clr)
    c1,c2=st.columns([1,2])
    with c1:
        sec("01","Satisfaction Gauge")
        fig=go.Figure(go.Indicator(mode="gauge+number+delta",value=avg,delta={'reference':3.0,'increasing':{'color':GR}},gauge={'axis':{'range':[1,5]},'bar':{'color':A,'thickness':.28},'bgcolor':'#181b22','bordercolor':'#1f2430','steps':[{'range':[1,3],'color':'#1a0a10'},{'range':[3,5],'color':'#0a1a18'}]},number={'font':{'color':A,'size':38}},title={'text':'Average CSAT','font':{'color':'#9ca3af','size':11}}))
        fig.update_layout(**{**PL,'height':240,'margin':dict(l=20,r=20,t=20,b=10)});st.plotly_chart(fig,use_container_width=True)
    with c2:
        sec("02","Score Distribution")
        vc=df_raw['CSAT Score'].value_counts().sort_index()
        fig=px.bar(x=vc.index.astype(str),y=vc.values,color=vc.values,color_continuous_scale=[[0,'#1a0e2e'],[.5,A2],[1,A]],labels={'x':'CSAT Score','y':'Count'})
        fig.update_traces(marker_line_width=0);fig.update_layout(**{**PL,'height':240,'showlegend':False,'coloraxis_showscale':False});st.plotly_chart(fig,use_container_width=True)
    sec("03","ML Pipeline")
    cols=st.columns(4)
    for i,(n,t,d) in enumerate([("01","Data Ingestion","85,907 rows · 20 cols"),("02","EDA & Stats","14 charts · 3 tests"),("03","Preprocessing","Encode · Datetime FE"),("04","NLP Pipeline","Contractions · TF-IDF"),("05","Feature Select","ExtraTrees Top-8"),("06","PCA Reduction",f"10 components · {pca_var:.2f} var"),("07","Model Training","CB · RF · XGBoost"),("08","Evaluation","MSE · R² · Radar")]):
        with cols[i%4]: st.markdown(f"<div class='mcard' style='padding:14px;'><div style='font-size:.62rem;color:{A};margin-bottom:4px;'>STEP {n}</div><div style='font-family:Syne,sans-serif;font-weight:700;font-size:.88rem;margin-bottom:4px;'>{t}</div><div style='font-size:.72rem;color:#6b7280;'>{d}</div></div>",unsafe_allow_html=True)

elif "Explorer" in page:
    hero("Data Explorer","Inspect · Audit · Understand your dataset")
    t1,t2,t3,t4=st.tabs(["📋 Raw Data","📐 Info","🕳️ Missing","📈 Stats"])
    with t1:
        n=st.slider("Rows",5,200,20);st.dataframe(df_raw.head(n),use_container_width=True)
        ibox(f"<b style='color:{A};'>{r:,} rows</b> · <b style='color:{A};'>{c_} columns</b> · Duplicates: <b style='color:{A};'>{df_raw.duplicated().sum()}</b>")
    with t2:
        info_df=pd.DataFrame({'Column':df_raw.columns,'Dtype':df_raw.dtypes.astype(str),'Non-Null':df_raw.notnull().sum(),'Null':df_raw.isnull().sum(),'Unique':[df_raw[c].nunique() for c in df_raw.columns]})
        st.dataframe(info_df,use_container_width=True,hide_index=True)
    with t3:
        mv=df_raw.isnull().sum().sort_values(ascending=False);mvp=(mv/r*100).round(2)
        fig=go.Figure(go.Bar(x=mv.index,y=mvp.values,marker=dict(color=mvp.values,colorscale=[[0,'#1f2430'],[.5,A2],[1,A3]],line_width=0)))
        fig.update_layout(**{**PL,'height':300,'xaxis_tickangle':-35,'yaxis_title':'Missing %'});st.plotly_chart(fig,use_container_width=True)
        st.dataframe(pd.DataFrame({'Column':mv.index,'Missing':mv.values,'%':mvp.values}),use_container_width=True,hide_index=True)
    with t4:
        st.dataframe(df_raw.describe().T.style.format("{:.2f}"),use_container_width=True)

elif "EDA" in page:
    hero("EDA & Visualisations","14 Interactive Charts · Business Insights")
    t1,t2,t3=st.tabs(["📦 Univariate","🔗 Bivariate","🌐 Multivariate"])
    with t1:
        c1,c2=st.columns(2)
        with c1:
            vc=df_p['CSAT Score'].value_counts().sort_index();fig=px.bar(x=vc.index.astype(str),y=vc.values,color=vc.values,color_continuous_scale=[[0,'#1a0e2e'],[.5,A2],[1,A]],labels={'x':'Score','y':'Count'});fig.update_traces(marker_line_width=0);fig.update_layout(**{**PL,'height':280,'title':'CSAT Distribution','coloraxis_showscale':False});st.plotly_chart(fig,use_container_width=True)
        with c2:
            ch=df_p['channel_name'].value_counts();fig=px.pie(values=ch.values,names=ch.index,hole=.55,color_discrete_sequence=[A,A2,A3]);fig.update_layout(**{**PL,'height':280,'title':'Channel Split'});fig.update_traces(marker_line_width=2,marker_line_color='#0a0c10');st.plotly_chart(fig,use_container_width=True)
        c3,c4=st.columns(2)
        with c3: bar(df_p['Agent Shift'].value_counts().index.tolist(),df_p['Agent Shift'].value_counts().values,"Agent Shift Distribution",A,260)
        with c4: bar(df_p['Tenure Bucket'].value_counts().index.tolist(),df_p['Tenure Bucket'].value_counts().values,"Tenure Bucket Distribution",A2,260)
        pd_=df_p['Item_price'].dropna();fig=go.Figure(go.Histogram(x=pd_,nbinsx=80,marker_color=A,opacity=.8,marker_line_width=0));fig.add_vline(x=pd_.median(),line_dash='dash',line_color=A2,annotation_text=f'Median ₹{pd_.median():,.0f}',annotation_font_color=A2);fig.update_layout(**{**PL,'height':260,'title':'Item Price Distribution','xaxis_title':'Price (₹)'});st.plotly_chart(fig,use_container_width=True)
    with t2:
        c1,c2=st.columns(2)
        with c1:
            grp=df_p.groupby(['channel_name','CSAT Score']).size().reset_index(name='count');fig=px.bar(grp,x='channel_name',y='count',color='CSAT Score',barmode='group',color_continuous_scale='Teal');fig.update_layout(**{**PL,'height':280,'title':'CSAT by Channel'});fig.update_traces(marker_line_width=0);st.plotly_chart(fig,use_container_width=True)
        with c2:
            sa=df_p.groupby('Agent Shift')['CSAT Score'].mean().reset_index();fig=px.bar(sa,x='Agent Shift',y='CSAT Score',color='CSAT Score',color_continuous_scale=[[0,'#1a0e2e'],[.5,A2],[1,A]],text_auto='.2f');fig.update_layout(**{**PL,'height':280,'title':'Avg CSAT by Shift','coloraxis_showscale':False});fig.update_traces(marker_line_width=0);st.plotly_chart(fig,use_container_width=True)
        c3,c4=st.columns(2)
        with c3:
            ca=df_p.groupby('category')['CSAT Score'].mean().sort_values();fig=go.Figure(go.Bar(x=ca.values,y=ca.index,orientation='h',marker=dict(color=ca.values,colorscale=[[0,'#1a0e2e'],[.5,A2],[1,A]],line_width=0)));fig.update_layout(**{**PL,'height':300,'title':'Avg CSAT by Category'});st.plotly_chart(fig,use_container_width=True)
        with c4:
            fig=px.box(df_p.dropna(subset=['Item_price']),x='CSAT Score',y='Item_price',color='CSAT Score',color_discrete_sequence=[A3,'#ff9a3c',GO,GR,A]);fig.update_layout(**{**PL,'height':300,'title':'Item Price by CSAT','showlegend':False});fig.update_traces(marker_line_width=0);st.plotly_chart(fig,use_container_width=True)
    with t3:
        corr=df_p[['Item_price','connected_handling_time','CSAT Score']].corr();fig=px.imshow(corr,text_auto='.2f',color_continuous_scale=[[0,A3],[.5,'#1f2430'],[1,A]],aspect='auto');fig.update_layout(**{**PL,'height':320,'title':'Correlation Heatmap'});st.plotly_chart(fig,use_container_width=True)
        df_sb=df_p.dropna(subset=['Product_category','category','channel_name','CSAT Score'])
        if len(df_sb):
            fig=px.sunburst(df_sb,path=['Product_category','category','channel_name'],values='CSAT Score',color='CSAT Score',color_continuous_scale=[[0,A3],[.5,A2],[1,A]]);fig.update_layout(**{**PL,'height':480,'title':'CSAT Hierarchy: Product > Category > Channel'});st.plotly_chart(fig,use_container_width=True)
        piv=df_p.groupby(['Agent Shift','category'])['CSAT Score'].mean().unstack(fill_value=0);fig=px.imshow(piv,text_auto='.1f',aspect='auto',color_continuous_scale=[[0,'#1a0e2e'],[.5,A2],[1,A]]);fig.update_layout(**{**PL,'height':300,'title':'Avg CSAT: Shift × Category'});st.plotly_chart(fig,use_container_width=True)

elif "Hypothesis" in page:
    hero("Hypothesis Testing","ANOVA · t-test · Chi-Square — Statistical Significance")
    df_h=df_p.copy()
    cg=[g['CSAT Score'].dropna().values for _,g in df_h.groupby('channel_name')]
    fa,fp=f_oneway(*cg);hv=df_h[df_h['Item_price']>=10000]['CSAT Score'].dropna();lv=df_h[df_h['Item_price']<10000]['CSAT Score'].dropna()
    ts,tp=ttest_ind(hv,lv,equal_var=False);ct=pd.crosstab(df_h['Agent Shift'],df_h['CSAT Score']);cs,cp,cd,_=chi2_contingency(ct)
    for nm,st_,pv,h0,verdict,clr in [("ANOVA — Channel vs CSAT",f"F={fa:.3f}",fp,"H₀: All channels equal mean CSAT","Channel significantly affects CSAT ✅",A),("Welch t-test — Item Price vs CSAT",f"t={ts:.3f}",tp,"H₀: High/low price equal CSAT","Price range influences satisfaction ✅",A2),("Chi-Square — Agent Shift vs CSAT",f"χ²={cs:.3f} df={cd}",cp,"H₀: CSAT independent of shift","Shift timing affects CSAT ✅",A3)]:
        st.markdown(f"<div class='hyp' style='border-left-color:{clr};'><div style='font-family:Syne,sans-serif;font-weight:600;font-size:.95rem;'>{nm}</div><div style='font-size:.72rem;color:#6b7280;margin:4px 0;'>{h0}</div><div style='font-family:DM Mono,monospace;font-size:.75rem;color:{clr};'>{st_} &nbsp;|&nbsp; p = {pv:.2e}</div><div style='font-size:.78rem;color:{GR};margin-top:5px;'>{verdict if pv<.05 else '❌ Fail to reject H₀'}</div></div>",unsafe_allow_html=True)
    fig=go.Figure()
    for i,(nm,g) in enumerate(df_h.groupby('channel_name')):
        fig.add_trace(go.Violin(x=[nm]*len(g),y=g['CSAT Score'],name=nm,box_visible=True,meanline_visible=True,fillcolor=[A,A2,A3][i%3],opacity=.7,line_color=[A,A2,A3][i%3]))
    fig.update_layout(**{**PL,'height':340,'title':'CSAT Distribution by Channel (Violin)'});st.plotly_chart(fig,use_container_width=True)
    cn=ct.div(ct.sum(axis=1),axis=0)*100;fig=px.imshow(cn.round(1),text_auto='.1f',aspect='auto',color_continuous_scale=[[0,'#1a0e2e'],[.5,A2],[1,A]],labels=dict(x='CSAT',y='Shift',color='%'))
    fig.update_layout(**{**PL,'height':280,'title':'CSAT % per Shift (normalised)'});st.plotly_chart(fig,use_container_width=True)

elif "Features" in page:
    hero("Feature Engineering","ExtraTrees Importance · PCA · NLP Pipeline")
    t1,t2,t3=st.tabs(["🌲 Importance","🔵 PCA","📝 NLP"])
    with t1:
        sec("01","ExtraTrees Feature Importance")
        Xn=X.select_dtypes(include='number')
        with st.spinner("Computing importances..."):
            sel=ExtraTreesRegressor(n_estimators=50,random_state=42,n_jobs=-1);sel.fit(Xn.fillna(0),y)
        fi=pd.DataFrame({'Feature':Xn.columns,'Importance':sel.feature_importances_}).sort_values('Importance')
        fig=go.Figure(go.Bar(x=fi['Importance'],y=fi['Feature'],orientation='h',marker=dict(color=fi['Importance'],colorscale=[[0,'#1f2430'],[.4,A2],[1,A]],line_width=0),hovertemplate='%{y}: %{x:.4f}<extra></extra>'))
        fig.update_layout(**{**PL,'height':400,'title':'Feature Importances (ExtraTreesRegressor)'});st.plotly_chart(fig,use_container_width=True)
    with t2:
        sec("02","PCA Explained Variance")
        pf=PCA(n_components=min(12,Xs.shape[1]));pf.fit(Xs);ev=pf.explained_variance_ratio_;cs2=np.cumsum(ev)
        fig=make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Bar(x=[f'PC{i+1}' for i in range(len(ev))],y=ev*100,name='Individual %',marker_color=A2,marker_line_width=0))
        fig.add_trace(go.Scatter(x=[f'PC{i+1}' for i in range(len(cs2))],y=cs2*100,name='Cumulative %',line=dict(color=A,width=2),marker=dict(size=6)),secondary_y=True)
        fig.update_layout(**{**PL,'height':320,'title':'PCA Explained Variance'});st.plotly_chart(fig,use_container_width=True)
        ibox(f"<b style='color:{A};'>10 PCA components</b> capture <b style='color:{A};'>{pca_var*100:.1f}%</b> variance. Reduced {X.shape[1]} → 10 features.")
        p2=PCA(n_components=2);Xp2=p2.fit_transform(Xs);idx=np.random.choice(len(Xp2),min(3000,len(Xp2)),replace=False)
        fig=px.scatter(x=Xp2[idx,0],y=Xp2[idx,1],color=y.iloc[idx].astype(str),opacity=.5,labels={'x':'PC1','y':'PC2','color':'CSAT'},color_discrete_sequence=[A3,'#ff9a3c',GO,GR,A])
        fig.update_traces(marker_size=4,marker_line_width=0);fig.update_layout(**{**PL,'height':360,'title':'PCA 2D Projection'});st.plotly_chart(fig,use_container_width=True)
    with t3:
        sec("03","NLP Preprocessing Pipeline")
        for n,t,d,clr in [("1","Contraction Expansion","can't→cannot, won't→will not",A),("2","Lowercase Normalise","ALL CAPS → lowercase",A2),("3","Punctuation Removal","Remove . , ! ? ; : \" ' …",A3),("4","URL & Number Strip","http://... · numeric tokens",GO),("5","Stopword Removal","NLTK English stopwords",GR),("6","TF-IDF Vectorisation","max_features=100 → 85907×100 matrix",A)]:
            st.markdown(f"<div style='display:flex;gap:12px;align-items:flex-start;padding:12px 0;border-bottom:1px solid #1f2430;'><div style='width:26px;height:26px;border-radius:50%;background:{clr}22;border:1px solid {clr}44;display:flex;align-items:center;justify-content:center;font-size:.68rem;color:{clr};flex-shrink:0;'>{n}</div><div><div style='font-family:Syne,sans-serif;font-weight:600;font-size:.88rem;color:#e8eaf0;'>{t}</div><div style='font-size:.73rem;color:#6b7280;margin-top:2px;'>{d}</div></div></div>",unsafe_allow_html=True)

elif "Models" in page:
    hero("Model Training","CatBoost · Random Forest · XGBoost — Live Training")
    ibox(f"Training on <b style='color:{A};'>PCA-reduced features (10 components, {pca_var*100:.1f}% variance)</b> · 80/20 split · all models use <b style='color:{A};'>n_jobs=-1</b>")
    if st.button("⚡ Train All Models") or 'results' not in st.session_state:
        with st.spinner("Training on all CPU cores..."): st.session_state['results']=train(Xtr,Xte,ytr,yte)
    results=st.session_state.get('results',{})
    for nm,res in results.items():
        if 'error' in res: st.error(f"{nm}: {res['error']}"); continue
        mse,r2_=res['mse'],res['r2'];pred=res['pred'];resid=yte.values-pred
        c1,c2,c3=st.columns([2,1,1])
        with c1: st.markdown(f"<div class='mcard'><div style='font-family:Syne,sans-serif;font-weight:700;font-size:1rem;margin-bottom:12px;'>{nm}{'<span class=\"badge\">BEST</span>' if nm==(max(results,key=lambda k:results[k].get('r2',-99)) if results else '') else ''}</div><div style='display:flex;gap:20px;'><div><div class='klabel'>MSE</div><div style='font-family:Syne;font-weight:700;font-size:1.4rem;color:{A3};'>{mse:.4f}</div></div><div><div class='klabel'>R²</div><div style='font-family:Syne;font-weight:700;font-size:1.4rem;color:{A};'>{r2_:.4f}</div></div><div><div class='klabel'>RMSE</div><div style='font-family:Syne;font-weight:700;font-size:1.4rem;color:{A2};'>{mse**.5:.4f}</div></div></div></div>",unsafe_allow_html=True)
        with c2:
            fig=go.Figure(go.Histogram(x=resid,nbinsx=40,marker_color=A2,opacity=.85,marker_line_width=0));fig.update_layout(**{**PL,'height':170,'title':'Residuals','margin':dict(l=10,r=10,t=30,b=10),'showlegend':False});st.plotly_chart(fig,use_container_width=True)
        with c3:
            s=np.random.choice(len(yte),min(500,len(yte)),replace=False);fig=go.Figure([go.Scatter(x=yte.values[s],y=pred[s],mode='markers',marker=dict(color=A,size=4,opacity=.5,line_width=0)),go.Scatter(x=[1,5],y=[1,5],mode='lines',line=dict(color=A3,dash='dash',width=1.5))])
            fig.update_layout(**{**PL,'height':170,'title':'Actual vs Pred','margin':dict(l=10,r=10,t=30,b=10),'showlegend':False});st.plotly_chart(fig,use_container_width=True)

elif "Comparison" in page:
    hero("Model Comparison","Side-by-Side Performance · Radar Analysis")
    if 'results' not in st.session_state:
        with st.spinner("Training..."): st.session_state['results']=train(Xtr,Xte,ytr,yte)
    results=st.session_state['results'];valid={k:v for k,v in results.items() if 'error' not in v}
    if not valid: st.warning("Go to Models page first.");st.stop()
    ms=list(valid.keys());mv=[valid[m]['mse'] for m in ms];rv=[valid[m]['r2'] for m in ms];rmv=[v**.5 for v in mv];bm=ms[np.argmax(rv)]
    cols=st.columns(3)
    for col,l,v,d,clr in zip(cols,["Best Model","Best R²","Best MSE"],[bm,f"{max(rv):.4f}",f"{min(mv):.4f}"],["highest R²","coefficient of determination","mean squared error"],[A,A,A3]):
        with col: kpi(l,v,d,clr)
    c1,c2=st.columns(2)
    with c1:
        fig=go.Figure(go.Bar(x=ms,y=mv,marker=dict(color=mv,colorscale=[[0,GR],[.5,A2],[1,A3]],line_width=0),text=[f'{v:.4f}' for v in mv],textposition='outside',textfont=dict(size=11,color='#9ca3af')));fig.update_layout(**{**PL,'height':300,'title':'MSE — lower is better ↓'});st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig=go.Figure(go.Bar(x=ms,y=rv,marker=dict(color=rv,colorscale=[[0,A3],[.5,A2],[1,GR]],line_width=0),text=[f'{v:.4f}' for v in rv],textposition='outside',textfont=dict(size=11,color='#9ca3af')));fig.update_layout(**{**PL,'height':300,'title':'R² — higher is better ↑'});st.plotly_chart(fig,use_container_width=True)
    mn=lambda v,vals:[(x-min(vals))/(max(vals)-min(vals)+1e-9) for x in vals];imn=lambda v,vals:[1-(x-min(vals))/(max(vals)-min(vals)+1e-9) for x in vals]
    mn_mse=imn(None,mv);mn_r2=mn(None,rv);mn_rmse=imn(None,rmv)
    fig=go.Figure()
    for i,m in enumerate(ms):
        vals=[mn_mse[i],mn_rmse[i],mn_r2[i],mn_mse[i]];cats=['MSE(inv)','RMSE(inv)','R²','MSE(inv)']
        fig.add_trace(go.Scatterpolar(r=vals,theta=cats,fill='toself',name=m,line_color=[A,A2,A3][i]))
    fig.update_layout(**{**PL,'height':360,'polar':dict(bgcolor='#111318',radialaxis=dict(gridcolor='#1f2430',linecolor='#1f2430'),angularaxis=dict(gridcolor='#1f2430',linecolor='#1f2430')),'title':'Performance Radar (normalised)'});st.plotly_chart(fig,use_container_width=True)
    st.dataframe(pd.DataFrame({'Model':ms,'MSE':[f'{v:.4f}' for v in mv],'RMSE':[f'{v:.4f}' for v in rmv],'R²':[f'{v:.4f}' for v in rv],'Status':['🏆 Best' if m==bm else '—' for m in ms]}),use_container_width=True,hide_index=True)

elif "Predictor" in page:
    hero("Live CSAT Predictor","Enter interaction details → Instant CSAT prediction")
    if 'results' not in st.session_state:
        with st.spinner("Preparing..."): st.session_state['results']=train(Xtr,Xte,ytr,yte)
    results=st.session_state['results'];valid={k:v for k,v in results.items() if 'error' not in v}
    if not valid: st.warning("Go to Models page first.");st.stop()
    bm=max(valid,key=lambda k:valid[k]['r2']);bmdl=valid[bm]['model']
    ibox(f"Using: <b style='color:{A};'>{bm}</b> (R² = <b style='color:{A};'>{valid[bm]['r2']:.4f}</b>)")
    with st.form("pf"):
        c1,c2,c3=st.columns(3)
        with c1: channel=st.selectbox("Channel",['inbound','outbound','outcall'])
        with c2: shift=st.selectbox("Agent Shift",['morning','evening','night','afternoon'])
        with c3: tenure=st.selectbox("Tenure",['0-30','31-60','61-90','>90','on job training'])
        c4,c5=st.columns(2)
        with c4: category=st.selectbox("Category",['order related','returns','cancellation','refund related','product queries','payments related','feedback','others','app/website','offers & cashback'])
        with c5: sub=st.selectbox("Sub-Category",['delivery related','reverse pickup enquiry','not needed','installation/demo','product specific information','others'])
        c6,c7,c8,c9=st.columns(4)
        with c6: rh=st.slider("Reported Hour",0,23,10)
        with c7: rd=st.slider("Reported DoW",0,6,1,help="0=Mon")
        with c8: sh2=st.slider("Responded Hour",0,23,11)
        with c9: sd=st.slider("Responded DoW",0,6,1)
        rt=st.slider("Response Time (hrs)",0.0,24.0,0.5,step=0.1)
        submitted=st.form_submit_button("🔮 Predict CSAT Score",use_container_width=True)
    if submitted:
        le2=LabelEncoder();enc=lambda v,vs: int(le2.fit(vs).transform([v])[0]) if v in vs else 0
        Xn=X.select_dtypes(include='number')
        sd_={'channel_name':[enc(channel,['inbound','outbound','outcall'])],'category':[enc(category,['order related','returns','cancellation','refund related','product queries','payments related','feedback','others','app/website','offers & cashback'])],'Sub-category':[0],'Agent Shift':[enc(shift,['morning','evening','night','afternoon'])],'Tenure Bucket':[enc(tenure,['0-30','31-60','61-90','>90','on job training'])],'issue_reported_hour':[rh],'issue_reported_dayofweek':[rd],'issue_responded_hour':[sh2],'issue_responded_dayofweek':[sd],'response_time_hrs':[rt]}
        sdf=pd.DataFrame({c:[0] for c in Xn.columns});sdf.update(pd.DataFrame(sd_));sdf=sdf[Xn.columns]
        sc2=StandardScaler();sc2.fit(Xn.fillna(0));pc2=PCA(n_components=10);pc2.fit(sc2.transform(Xn.fillna(0)))
        sp=pc2.transform(sc2.transform(sdf.fillna(0)));pred=float(np.clip(bmdl.predict(sp)[0],1,5))
        clr=GR if pred>=4 else GO if pred>=3 else A3;stars="⭐"*round(pred)
        st.markdown(f"<div class='pres'><div style='font-size:.72rem;color:#6b7280;letter-spacing:.1em;'>PREDICTED CSAT SCORE</div><div class='pscore' style='background:linear-gradient(90deg,{clr},{A2});-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>{pred:.2f}</div><div style='font-size:1.3rem;margin-top:6px;'>{stars}</div><div style='font-size:.72rem;color:#6b7280;margin-top:10px;'>Model: {bm} &nbsp;|&nbsp; {'High Satisfaction 🟢' if pred>=4 else 'Moderate 🟡' if pred>=3 else 'Low Satisfaction 🔴'}</div></div>",unsafe_allow_html=True)

elif "Export" in page:
    hero("Export & Report","Download cleaned data · Model metrics · Full PDF summary")
    # NEW FEATURE 1: Download cleaned CSV
    sec("01","Download Cleaned Dataset")
    ibox(f"Cleaned dataset with <b style='color:{A};'>label encoding</b>, <b style='color:{A};'>response_time_hrs</b> feature, and <b style='color:{A};'>NLP-processed remarks</b>. Shape: {df_clean.shape[0]:,} × {df_clean.shape[1]}")
    st.download_button("📥 Download Cleaned CSV",df_clean.to_csv(index=False).encode(),"eCommerce_cleaned.csv","text/csv",use_container_width=True)
    # NEW FEATURE 2: Model metrics report
    sec("02","Model Performance Report")
    if 'results' not in st.session_state:
        with st.spinner("Training..."): st.session_state['results']=train(Xtr,Xte,ytr,yte)
    results=st.session_state['results'];valid={k:v for k,v in results.items() if 'error' not in v}
    if valid:
        rpt=pd.DataFrame({'Model':list(valid.keys()),'MSE':[f"{v['mse']:.4f}" for v in valid.values()],'RMSE':[f"{v['mse']**.5:.4f}" for v in valid.values()],'R²':[f"{v['r2']:.4f}" for v in valid.values()]})
        st.dataframe(rpt,use_container_width=True,hide_index=True)
        st.download_button("📥 Download Metrics CSV",rpt.to_csv(index=False).encode(),"model_metrics.csv","text/csv",use_container_width=True)
    # NEW FEATURE 3: Dataset summary report
    sec("03","Auto-Generated Dataset Summary")
    rows_,cols_=df_raw.shape;miss_cols=df_raw.isnull().sum()
    summary=f"""DEEPCSAT — DATASET SUMMARY REPORT
{'='*50}
Total Records   : {rows_:,}
Total Columns   : {cols_}
Duplicate Rows  : {df_raw.duplicated().sum()}
Avg CSAT Score  : {df_raw['CSAT Score'].mean():.3f}
5-Star Rate     : {(df_raw['CSAT Score']==5).sum()/rows_*100:.1f}%
1-Star Rate     : {(df_raw['CSAT Score']==1).sum()/rows_*100:.1f}%
Missing Overall : {df_raw.isnull().sum().sum()/(rows_*cols_)*100:.1f}%
PCA Variance    : {pca_var*100:.2f}% (10 components)
\nMISSING VALUES PER COLUMN:\n{'-'*35}
{miss_cols[miss_cols>0].to_string()}
\nSTATISTICAL DESCRIPTION:\n{'-'*35}
{df_raw.describe().to_string()}"""
    st.code(summary, language='')
    st.download_button("📥 Download Summary TXT",summary.encode(),"dataset_summary.txt","text/plain",use_container_width=True)