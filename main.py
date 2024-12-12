import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# Loading dataset
file_path = "airline7.csv"
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# applying Fourier Transform on passenger numbers
passenger_data = df['Number'].values
num_points = len(passenger_data)
freq = np.fft.fftfreq(num_points)
fft_result = fft(passenger_data)

# monthly averages
df['DayOfYear'] = df['Date'].dt.dayofyear
monthly_avg = df.groupby('Month')['Number'].mean()

# monthly trends with Fourier
plt.figure(figsize=(9, 5))
plt.bar(monthly_avg.index, monthly_avg, color='blue', alpha=0.7, label='Monthly Average')
plt.xlabel('Month')
plt.ylabel('Passengers (Thousands)')
plt.title('Monthly Passenger Trends - ID: 23072371')
plt.legend()
plt.savefig('fourier.png')

# power spectrum ans value of y
power_spectrum = np.abs(fft_result[:num_points // 2]) ** 2
frequencies = freq[:num_points // 2]
value_y = 1 / frequencies[np.argmax(power_spectrum[1:]) + 1]

plt.figure(figsize=(9, 5))
plt.plot(frequencies, power_spectrum, color='red', label='Power Spectrum')
plt.xlabel('Frequency (1/day)')
plt.ylabel('Power')
plt.title('Frequency Analysis - ID: 23072371')
plt.legend()
plt.savefig('powerSpectrum.png')

# value of X
revenue = df['Revenue'].sum()
passenger_count = df['Number'].sum()
value_x = revenue / passenger_count


print(f"X (Average Revenue per Passenger): {value_x:.2f}")
print(f"Y (Dominant Period): {value_y:.2f} days")