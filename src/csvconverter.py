import csv

def converter(a):
  out = []
  
  for i in range(0,len(list(a.values())[0])):
    el = {}
    for k,e in a.items():
      el[k] = e[i]
    out.append(el)
  
  return out
  
def savecsv(trainHistory, filename='result.csv'):
  with open(filename, 'w') as f:
    w = csv.DictWriter(f, trainHistory.keys())
    w.writeheader()
    for data in converter(trainHistory):
      w.writerow(data)
