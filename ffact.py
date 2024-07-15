import itertools
import csv

def generate_ranges(start, stop, step):
    # generates a list of values for a variable given starting and ending points and step size
    values = []
    current = start
    while current <= stop:
        values.append(current)
        current += step
    return values

def generate_factorial_table():
    # define ranges for each variable
    var1_range = generate_ranges(6, 10, 0.25)
    var2_range = generate_ranges(355, 455, 10)
    var3_range = generate_ranges(9, 14, 0.1)
    var4_range = generate_ranges(2, 4, 0.25)
    
    # generate combinations of all variables
    combinations = list(itertools.product(var1_range, var2_range, var3_range, var4_range))
    
    return combinations

# generate the factorial table
factorial_table = generate_factorial_table()

# save the full factorial table to a CSV file
csv_file = 'factorial_table.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['secondary headers', 'primary headers', 'plenum volume', 'intake runners'])  # Write header
    writer.writerows(factorial_table)  # Write rows of factorial_table
