import ipyparallel

print "Running ", __file__
c = ipyparallel.Client(profile="default")
print c.ids
