import numpy as np
import csv
from resample import resample_from_wv
import pdb


def read_csv(file):
    wv = []
    data = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            wv.append(float(row[0].replace(',', '.')))
            data.append(float(row[1].replace(',', '.')))
    wv = np.array(wv)
    data = np.array(data)
    return wv, data

def gas_transmittance(gas_concentration, gas_absorption, sensor_sfr, theta):
    transmittance = np.exp(- gas_concentration[:, np.newaxis, np.newaxis] * gas_absorption[np.newaxis, np.newaxis, :] * (1 + 1 / np.cos(theta))) * sensor_sfr[np.newaxis, :, :] 
    transmittance = np.sum(transmittance, axis=2) / np.sum(sensor_sfr, axis=1).reshape(1, -1)
    return transmittance