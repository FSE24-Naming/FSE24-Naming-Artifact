import re
import json
from matplotlib import pyplot as plt

def camelCaseChecker(txt):
    return 1 if re.search("^[a-z][a-zA-Z0-9]*$", txt) is not None else 0

def PascalCaseChecker(txt):
    return 1 if re.search("^[A-Z][a-zA-Z0-9]*$", txt) is not None else 0

def snake_case_checker(txt):
    return 1 if re.search("^[a-zA-Z0-9]+(_[a-zA-Z0-9]+)*$", txt) is not None else 0

def kebab_case_checker(txt):
    return 1 if re.search("^[a-zA-Z0-9]+(-[a-zA-Z0-9]+)*$", txt) is not None else 0

def mixed_caseChecker(txt):
    return 1 if re.search("^(?=.*-)(?=.*_)[a-zA-Z0-9]+([-_][a-zA-Z0-9]+)*$", txt) is not None else 0

with open("final_cleaned_output.json") as f:
    in_data = json.load(f)

with open("external_model_name.json") as f:
    ex_data = json.load(f)


def process_data(data):

    names = ["snake_case", "kebab-case", "camelCase", "PascalCase", "mix_ed-Case", "Other"]

    ccnt, pcnt, scnt, kcnt, mcnt, other = 0, 0, 0, 0, 0, 0
    for model_name in data:
        model_name_tail = model_name.split('/')[-1]
        print(f"Checking: {model_name_tail}")  # Debugging line
        if kebab_case_checker(model_name_tail):
            print("Matched: kebab-case")  # Debugging line
            kcnt += 1
        elif snake_case_checker(model_name_tail):
            print("Matched: snake_case")  # Debugging line
            scnt += 1
        elif camelCaseChecker(model_name_tail):
            print("Matched: camelCase")  # Debugging line
            ccnt += 1
        elif PascalCaseChecker(model_name_tail):
            print("Matched: PascalCase")  # Debugging line
            pcnt += 1
        elif mixed_caseChecker(model_name_tail):
            print("Matched: mixed_case")  # Debugging line
            mcnt += 1
        else:
            print("Matched: Other")  # Debugging line
            other += 1


        
        
        
        # else:
        # # if camelCaseChecker(model_name_tail) + PascalCaseChecker(model_name_tail) + snake_case_checker(model_name_tail) + kebab_case_checker(model_name_tail) + mixed_caseChecker(model_name_tail) == 0:
        #     other += 1

    numbers = [scnt, kcnt, ccnt, pcnt, mcnt, other]

    # l = []
    # for i, j in zip(names, numbers):
    #     l.append((i, j))

    # l = sorted(l, key=lambda x: x[1])

    # names = [i[0] for i in l]
    # numbers = [i[1] for i in l]
    return names, numbers


def plot_data(names, numbers, color, label, offset):
    total = sum(numbers)
    if total == 0:
        print("Total is zero. Cannot calculate percentages.")
        return
    
    percentages = [(num / total) * 100 for num in numbers]  # Convert to percentages for plotting
    bars = plt.barh([y + offset for y in range(len(names))], percentages, color=color, height=0.4, label=label)
    for bar, percentage in zip(bars, percentages):
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height() / 2 - 0.1, f'{percentage:.2f}%', fontsize=25)
        

names = ["snake_case", "kebab-case", "camelCase", "PascalCase", "mix_ed-Case", "Other"]

# Process the data
names_in, numbers_in = process_data(in_data)
names_ex, numbers_ex = process_data(ex_data)

# Combine and sort both lists by value
combined_numbers = [(name, num_in, num_ex) for name, num_in, num_ex in zip(names, numbers_in, numbers_ex)]
combined_numbers.sort(key=lambda x: x[1] + x[2])
print(combined_numbers)
# Separate the sorted tuples back into individual lists
sorted_names = [x[0] for x in combined_numbers]
sorted_numbers_in = [x[1] for x in combined_numbers]
sorted_numbers_ex = [x[2] for x in combined_numbers]

# Create the plot
plt.figure(figsize=(15, 6))
print(names_in, numbers_in)
print(names_ex, numbers_ex)
# Make sure to use the sorted lists here
print(sorted_names, sorted_numbers_ex, sorted_numbers_in)
plot_data(sorted_names, sorted_numbers_ex, 'green', 'Gated PTM pkgs', 0.2)
plot_data(sorted_names, sorted_numbers_in, 'blue', 'Open PTM pkgs', -0.2)



plt.yticks(range(len(sorted_names)), sorted_names, fontsize=30)
plt.xlabel("Percentages", fontsize=40)  # Update this line to 'Percentages'
plt.ylabel("Naming styles", fontsize=40)
plt.legend(fontsize=30, loc='lower right')
plt.tight_layout()
plt.savefig('case_comparison_side_by_side.png')


