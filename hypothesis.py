import json

filename = "hyp_data.json"
with open(filename) as f:
    hyp_data = json.load(f)

x = [1]
print("Enter the input features, excluding the coventional x_0 = 1.")
for i in range(1, len(hyp_data["theta"])):
    x.append((float(input()) - hyp_data["mu_X"][i]) / hyp_data["sigma_X"][i])

y = sum([x_el * th_el for (x_el, th_el) in zip(x, hyp_data["theta"])])
print(f"With gradient descent: y = {y}")

y = sum([x_el * th_el for (x_el, th_el) in zip(x, hyp_data["theta_n"])])
print(f"With normal equation: y = {y}")
