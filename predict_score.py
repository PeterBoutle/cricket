import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

ds = pd.read_csv('Batsman_Data.csv')
jr = ds[ds.Batsman=="Joe Root"]
jr_bat = jr[["Runs","Start Date"]]
jr_bat["Runs"] = pd.to_numeric(jr_bat["Runs"],errors='coerce')
jr_bat["Start Date"] = pd.to_datetime(jr_bat["Start Date"],errors='coerce')
jr_bat = jr_bat[jr_bat.Runs>0]

jr_runs = jr_bat["Runs"].to_numpy()
jr_date = jr_bat["Start Date"].to_numpy()

model = LinearRegression().fit(jr_date.reshape((-1,1)),jr_runs)
prediction = model.predict(jr_runs.reshape((-1,1)))

print(prediction*100)
print(jr_runs)

with plt.xkcd():
    fig = plt.figure()
    #ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar([-0.125, 1.0-0.125], [0, 100], 0.25)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xticks([])
    ax.set_yticks([])
   
    data = (jr_runs)

    ax.plot(data)
    #ax.plot(prediction*100)
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Runs")
plt.show()


