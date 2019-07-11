from salesmap import *

n = 20
name = 'map'+str(n)+'.npy'
sales_map = SalesMap(name)
path = list(range(n))+[0]
print(sales_map.loss(path))
sales_map.show(path)
