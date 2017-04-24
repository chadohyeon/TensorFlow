import numpy as np
import scipy.stats as st

# rewrite the appropriate path to run
a = np.loadtxt("hcpIntel.csv", delimiter=",", dtype=np.int32)
uni, con = np.unique(a, return_counts=True)
dic = dict(zip(uni,con))

# probability of given x among the labels
def proba(x):
    return dic[x] / len(a)

# probability of given x in normal distribution
def probb(x):
    return st.norm(14.5, 1).pdf(x)

"""
totalprob = 0
for i in range(5, 25):
    total += proba(i)
    print(proba(i))

print(total)    -> this returns 1
"""

MSE = 0
for i in range(5,25):
    for j in range(5,25):
        tmp = (j-i) * (j-i)
        tmp *= proba(j) * probb(i)
        MSE += tmp

print(MSE)
# expected MSE becomes 28.4647743407 in case of random normal distribution
