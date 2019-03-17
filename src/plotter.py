import csv
import matplotlib.pyplot as plt

with open("../report/perc_error_table.csv") as data:
    data_reader = csv.reader(data)
    per_err_table = list(data_reader)

with open("../report/cumm_perc_err_table.csv") as data:
    data_reader = csv.reader(data)
    cumm_per_err_table = list(data_reader)

plot_X = list(map(lambda x: x[0], per_err_table))[1:]
y_per_err = list(map(lambda x: x[1], per_err_table))[1:]
y_cumm_per_err = list(map(lambda x: x[1], cumm_per_err_table))[1:]

plot_X = list(map(lambda x: float(x), plot_X))
y_per_err = list(map(lambda x: float(x), y_per_err))
y_cumm_per_err = list(map(lambda x: float(x), y_cumm_per_err))


print(plot_X[:100])
print(y_per_err[:100])
print(y_cumm_per_err[:100])

# plt.scatter(plot_X, y_per_err, s=5)

# plt.plot(plot_X[:100], y_per_err[:100], 'g--', label='Percentage Error')
plt.bar(plot_X[:], y_per_err[:], label='Percentage Error')
plt.plot(plot_X[:], y_cumm_per_err[:], 'r--', label='Cumulative Percentage Error')
# plt.grid()
plt.xlabel("Index")
plt.ylabel("Percentage Error")
plt.legend()
plt.tight_layout()
plt.ylim(0,0.2)
# plt.annotate('%s' % y_cumm_per_err[-1], xy=(plot_X[-1], y_cumm_per_err[-1]), textcoords='data')
plt.show()
