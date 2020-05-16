import json

def hypothesis(xin):
    """the hypothesis function"""
    filename = "hyp_data.json"
    with open(filename) as f:
        hyp_data = json.load(f)

    x = [1]
    for i in range(1, len(hyp_data["theta"])):
        x.append((xin[i - 1] - hyp_data["mu_X"][i]) / hyp_data["sigma_X"][i])

    y = sum([x_el * th_el for (x_el, th_el) in zip(x, hyp_data["theta"])])
    return y
