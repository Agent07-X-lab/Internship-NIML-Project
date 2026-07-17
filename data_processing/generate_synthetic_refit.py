import os
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_house_data(house_id, num_rows=3000):
    print(f"Generating synthetic data for House {house_id}...")
    np.random.seed(42 + house_id)
    random.seed(42 + house_id)
    
    start_time = datetime(2014, 4, 1, 0, 0, 0)
    timestamps = []
    current_time = start_time
    
    # Generate irregular timestamps (avg 8 seconds, sometimes gaps)
    for _ in range(num_rows):
        timestamps.append(current_time)
        # 2% chance of a longer gap (e.g., sensor dropout of 30-90s)
        if random.random() < 0.02:
            gap = random.randint(30, 90)
        else:
            gap = random.randint(6, 10)
        current_time += timedelta(seconds=gap)
        
    unix_timestamps = [int(t.timestamp()) for t in timestamps]
    time_strings = [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps]
    
    # Create the appliances
    # We will simulate 9 appliances + 1 aggregate
    data = {}
    
    # Simulating different types of appliances
    # Appliance 1: Fridge (cycles on/off)
    fridge_state = 0 # 0=off, 1=on
    fridge_ticks = 0
    fridge_power = []
    for i in range(num_rows):
        if fridge_ticks <= 0:
            fridge_state = 1 if fridge_state == 0 else 0
            fridge_ticks = random.randint(40, 80) # duration of cycle
        fridge_ticks -= 1
        if fridge_state == 1:
            power = 120.0 + np.random.normal(0, 5)
        else:
            power = 1.5 + np.random.normal(0, 0.2) # standby
        fridge_power.append(max(0.0, power))
        
    # Appliance 2: Washing Machine (complex profile)
    wm_active = False
    wm_ticks = 0
    wm_cycle_step = 0
    wm_power = []
    for i in range(num_rows):
        if not wm_active:
            if random.random() < 0.005: # starts occasionally
                wm_active = True
                wm_ticks = random.randint(150, 250)
                wm_cycle_step = 0
        if wm_active:
            wm_ticks -= 1
            if wm_ticks <= 0:
                wm_active = False
                power = 0.0
            else:
                # washing machine cycle phases: heating (high), spinning (medium), rinsing (low)
                if wm_ticks > 120: # heating phase
                    power = 1800.0 + np.random.normal(0, 50)
                elif wm_ticks > 40: # wash/rinse
                    power = 250.0 + np.random.normal(0, 20)
                else: # spin
                    power = 600.0 + np.random.normal(0, 40)
        else:
            power = 0.2 + np.random.normal(0, 0.05) # standby
        wm_power.append(max(0.0, power))
        
    # Appliance 3: Dishwasher (complex profile)
    dw_active = False
    dw_ticks = 0
    dw_power = []
    for i in range(num_rows):
        if not dw_active:
            if random.random() < 0.003:
                dw_active = True
                dw_ticks = random.randint(200, 300)
        if dw_active:
            dw_ticks -= 1
            if dw_ticks <= 0:
                dw_active = False
                power = 0.0
            else:
                if dw_ticks > 180 or (dw_ticks > 60 and dw_ticks < 100): # heating cycles
                    power = 1200.0 + np.random.normal(0, 30)
                else:
                    power = 150.0 + np.random.normal(0, 10)
        else:
            power = 0.5 + np.random.normal(0, 0.05)
        dw_power.append(max(0.0, power))
        
    # Appliance 4: Television (steady load)
    tv_active = False
    tv_ticks = 0
    tv_power = []
    for i in range(num_rows):
        if not tv_active:
            if random.random() < 0.01:
                tv_active = True
                tv_ticks = random.randint(100, 300)
        if tv_active:
            tv_ticks -= 1
            if tv_ticks <= 0:
                tv_active = False
                power = 0.0
            else:
                power = 80.0 + np.random.normal(0, 3)
        else:
            power = 0.5 + np.random.normal(0, 0.1)
        tv_power.append(max(0.0, power))
        
    # Appliance 5: Microwave (high load, short duration)
    mw_active = False
    mw_ticks = 0
    mw_power = []
    for i in range(num_rows):
        if not mw_active:
            if random.random() < 0.008:
                mw_active = True
                mw_ticks = random.randint(5, 20)
        if mw_active:
            mw_ticks -= 1
            if mw_ticks <= 0:
                mw_active = False
                power = 0.0
            else:
                power = 1100.0 + np.random.normal(0, 20)
        else:
            power = 1.0 + np.random.normal(0, 0.1)
        mw_power.append(max(0.0, power))
        
    # Appliance 6: Kettle (extreme load, very short duration)
    kt_active = False
    kt_ticks = 0
    kt_power = []
    for i in range(num_rows):
        if not kt_active:
            if random.random() < 0.005:
                kt_active = True
                kt_ticks = random.randint(8, 15)
        if kt_active:
            kt_ticks -= 1
            if kt_ticks <= 0:
                kt_active = False
                power = 0.0
            else:
                power = 2200.0 + np.random.normal(0, 50)
        else:
            power = 0.0
        kt_power.append(max(0.0, power))
        
    # Appliance 7, 8, 9: General low power/standby devices
    app7_power = []
    app8_power = []
    app9_power = []
    for i in range(num_rows):
        # random spikes for small items
        p7 = 15.0 + np.random.normal(0, 1) if random.random() < 0.2 else 0.0
        p8 = 45.0 + np.random.normal(0, 2) if random.random() < 0.1 else 0.5
        p9 = 10.0 + np.random.normal(0, 0.5) if random.random() < 0.05 else 0.0
        app7_power.append(max(0.0, p7))
        app8_power.append(max(0.0, p8))
        app9_power.append(max(0.0, p9))
        
    # Build dataframe columns
    data['Time'] = time_strings
    data['Unix'] = unix_timestamps
    
    # Calculate Aggregate: sum of all appliances + background load
    background_load = 50.0 + np.random.normal(0, 5, num_rows)
    appliance_sum = (
        np.array(fridge_power) + np.array(wm_power) + np.array(dw_power) +
        np.array(tv_power) + np.array(mw_power) + np.array(kt_power) +
        np.array(app7_power) + np.array(app8_power) + np.array(app9_power)
    )
    aggregate_power = appliance_sum + np.maximum(0, background_load)
    data['Aggregate'] = [int(p) for p in aggregate_power]
    
    # Add appliances as integers to mimic raw REFIT data
    data['Appliance1'] = [int(p) for p in fridge_power]
    data['Appliance2'] = [int(p) for p in wm_power]
    data['Appliance3'] = [int(p) for p in dw_power]
    data['Appliance4'] = [int(p) for p in tv_power]
    data['Appliance5'] = [int(p) for p in mw_power]
    data['Appliance6'] = [int(p) for p in kt_power]
    data['Appliance7'] = [int(p) for p in app7_power]
    data['Appliance8'] = [int(p) for p in app8_power]
    data['Appliance9'] = [int(p) for p in app9_power]
    
    # Introduce random missing sensor readings (NaNs) in about 1% of data
    df = pd.DataFrame(data)
    for col in ['Aggregate'] + [f'Appliance{i}' for i in range(1, 10)]:
        mask = np.random.rand(*df[col].shape) < 0.01
        df.loc[mask, col] = np.nan
        
    # Save to file
    out_dir = "1_raw_data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"CLEAN_House{house_id}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved synthetic house {house_id} to {out_path} with {len(df)} rows.")

def main():
    # Generate for Houses 1 to 21
    for i in range(1, 22):
        generate_house_data(i)
    print("All synthetic raw data generated successfully!")

if __name__ == "__main__":
    main()
