import pandas as pd
P = '/mnt/c/Users/archi/OneDrive/Desktop/Sem 6/PE/AI-Native-Semantic-Network-Control-for-Remote-Healthcare-Traffic'
df = pd.read_csv(P + '/data/datasets/realistic_network_dataset.csv')
df['jitter_ms'] = df['jitter'] * 1000.0
df['delay_ms']  = df['avg_delay'] * 1000.0

print('=== Raw CSV columns ===')
print(df.columns.tolist())
print()
print('=== Raw jitter/delay in CSV (seconds?) ===')
print(df[['jitter','avg_delay','packet_loss_rate']].describe().round(6))
print()
print('=== After *1000 (ms) ===')
print(df[['jitter_ms','delay_ms','packet_loss_rate']].describe().round(3))
print()
print('=== Label distribution ===')
print(df['network_condition'].value_counts())
print()
for cls in ['Stable','Unstable','Critical']:
    sub = df[df['network_condition']==cls]
    jms = sub['jitter_ms']
    dms = sub['delay_ms']
    lss = sub['packet_loss_rate']
    print(f'{cls}: jitter_ms [{jms.min():.1f} - {jms.max():.1f}]  delay_ms [{dms.min():.1f} - {dms.max():.1f}]  loss [{lss.min():.3f} - {lss.max():.3f}]')
