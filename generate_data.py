import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

# สร้างข้อมูล sensor ปกติ
data = {
    'timestamp': pd.date_range(start='2024-01-01', periods=n, freq='1min'),
    'temperature': np.random.normal(70, 5, n),
    'vibration': np.random.normal(0.5, 0.1, n),
    'pressure': np.random.normal(100, 10, n)
}

df = pd.DataFrame(data)

# ใส่ anomaly เข้าไป 20 จุด
anomaly_idx = np.random.choice(n, 20, replace=False)
df.loc[anomaly_idx, 'temperature'] += np.random.uniform(30, 50, 20)
df.loc[anomaly_idx, 'vibration'] += np.random.uniform(1, 2, 20)
df.loc[anomaly_idx, 'pressure'] += np.random.uniform(40, 60, 20)

df.to_csv('sensor_data.csv', index=False)
print("Dataset created!")