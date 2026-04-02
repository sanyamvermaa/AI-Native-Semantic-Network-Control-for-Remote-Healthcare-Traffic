import pandas as pd, joblib
from pathlib import Path

BASE = Path('/mnt/c/Users/archi/OneDrive/Desktop/Sem 6/PE/AI-Native-Semantic-Network-Control-for-Remote-Healthcare-Traffic')

tel = pd.read_csv(BASE / 'outputs/evaluation/closedloop_natural_20260401_192923/network_telemetry.csv')
for col in ['packet_loss_rate','jitter','avg_delay','throughput_bps','bandwidth_usage_bps','active_devices','packets_per_window']:
    tel[col] = pd.to_numeric(tel.get(col, 0), errors='coerce').fillna(0.0)

def heuristic(r):
    if r['packet_loss_rate']>=0.08 or r['avg_delay']>=130.0 or r['jitter']>=35.0: return 'Critical'
    if r['packet_loss_rate']>=0.03 or r['avg_delay']>=35.0  or r['jitter']>=8.0:  return 'Unstable'
    return 'Stable'

heur_preds = tel.apply(heuristic, axis=1)
print('Heuristic on latest-run telemetry:')
h_vc = heur_preds.value_counts()
for s, n in h_vc.items():
    print(f'  {s:<12}: {n:>5}  ({n/len(tel)*100:.1f}%)')

model = joblib.load(str(BASE / 'models/best_network_model.pkl'))
feats = list(model.feature_names_in_)

def mean(v): return sum(v)/len(v) if v else 0.0
def std(v):
    if len(v)<2: return 0.0
    mu=mean(v); return (sum((x-mu)**2 for x in v)/len(v))**0.5
def slope(v):
    n=len(v)
    if n<2: return 0.0
    xm=(n-1)/2; ym=mean(v)
    num=sum((i-xm)*(y-ym) for i,y in enumerate(v))
    den=sum((i-xm)**2 for i in range(n))
    return num/den if den else 0.0

history, ml_preds = [], []
for _, row in tel.iterrows():
    curr = {k: float(row.get(k,0)) for k in ['bandwidth_usage_bps','throughput_bps','packet_loss_rate','jitter','avg_delay','active_devices','packets_per_window']}
    history.append(curr)
    if len(history)>64: history.pop(0)
    if len(history)<8:
        ml_preds.append(None); continue
    W,S=10,8
    lv=[h['packet_loss_rate'] for h in history]; jv=[h['jitter'] for h in history]
    dv=[h['avg_delay'] for h in history];        tv=[h['throughput_bps'] for h in history]
    def r(v,n): return v[-n:] if len(v)>=n else v
    ld=lv[-1]-lv[-2] if len(lv)>=2 else 0.0; jd=jv[-1]-jv[-2] if len(jv)>=2 else 0.0
    dd=dv[-1]-dv[-2] if len(dv)>=2 else 0.0; pld=lv[-2]-lv[-3] if len(lv)>=3 else 0.0
    l3=mean(r(lv,3)); l8=mean(r(lv,8)); t3=mean(r(tv,3)); t8=mean(r(tv,8)); d3=mean(r(dv,3)); d8=mean(r(dv,8))
    feat_row = {
        'bandwidth_usage_bps':curr['bandwidth_usage_bps'],'throughput_bps':curr['throughput_bps'],
        'packet_loss_rate':curr['packet_loss_rate'],'jitter':curr['jitter'],'avg_delay':curr['avg_delay'],
        'rolling_loss_mean':mean(r(lv,W)),'rolling_loss_std':std(r(lv,W)),
        'rolling_jitter_mean':mean(r(jv,W)),'rolling_delay_mean':mean(r(dv,W)),
        'rolling_throughput_mean':mean(r(tv,W)),'rolling_throughput_std':std(r(tv,W)),
        'loss_delta':ld,'jitter_delta':jd,'delay_delta':dd,'loss_accel':ld-pld,
        'loss_trend_3':l3-l8,'throughput_trend_3':t3-t8,'delay_trend_3':d3-d8,
        'delay_slope':slope(r(dv,S)),'loss_slope':slope(r(lv,S)),
        'active_devices':curr['active_devices'],'packets_per_window':curr['packets_per_window'],
        'loss_x_delay':curr['packet_loss_rate']*curr['avg_delay'],
        'loss_x_jitter':curr['packet_loss_rate']*curr['jitter'],
    }
    v = pd.DataFrame([feat_row], columns=feats)
    ml_preds.append(model.predict(v)[0])

ml_valid = [(h,m) for h,m in zip(heur_preds.tolist(), ml_preds) if m is not None]
ml_s = pd.Series([x[1] for x in ml_valid])
h_v  = pd.Series([x[0] for x in ml_valid])

print()
print(f'New model offline predictions ({len(ml_s)} windows):')
for s, n in ml_s.value_counts().items():
    print(f'  {s:<12}: {n:>5}  ({n/len(ml_s)*100:.1f}%)')
print()
print(f'ML vs Heuristic agreement: {(ml_s.values==h_v.values).mean()*100:.1f}%')
print('DONE')
