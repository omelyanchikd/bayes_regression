import pandas

datafile = "C:\Diana\Poland\Poland\poland_aggregate.csv"

poland_data = pandas.read_csv(datafile, sep = ";", decimal = ",")

print(poland_data)


from sklearn import linear_model

clf = linear_model.BayesianRidge(compute_score= True )
clf.fit(poland_data[['wages', 'workers', 'budget', 'investment']], poland_data['sold'])

ols = linear_model.LinearRegression()
ols.fit(poland_data[['wages', 'workers', 'budget', 'investment']], poland_data['sold'])


print(clf.predict([1000, 0, 1000, 0]))
print(ols.predict([1000, 0, 1000, 0]))

print(clf.score(poland_data[['wages', 'workers', 'budget', 'investment']], poland_data['sold']))
print(ols.score(poland_data[['wages', 'workers', 'budget', 'investment']], poland_data['sold']))
