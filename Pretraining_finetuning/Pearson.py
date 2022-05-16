from scipy.stats import pearsonr

x=[60.67,59.69,60,59.99,62.07]
y=[1,1.43e-10,0.0825,0.0721,4.85]

print(pearsonr(x,y))