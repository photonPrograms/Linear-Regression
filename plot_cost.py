import json
import matplotlib.pyplot as plt

filename = "cost.json"
with open(filename) as f:
    cost_list = json.load(f)

plt.figure()
plt.plot(cost_list, "r")
plt.show()
