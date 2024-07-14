import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Tuition fees for each stage per term in 2023/2024
fees_per_term_2023 = {
    'Nursery': 0,  # We will calculate this separately
    'Reception': 2795,
    'Year 1': 3450,
    'Year 2': 3780,
    'Year 3': 4770,
    'Year 4': 5595,
    'Year 5': 6090,
    'Year 6': 6290,
    'Year 7': 6720,
    'Year 8': 6720,
    'Year 9': 7525,
    'Year 10': 7525,
    'Year 11': 7525,
    'Year 12': 7525,
    'Year 13': 7525
}

# Inflation rate
inflation_rate = 0.025

# Adjust fees for inflation from 2023 to 2025
initial_inflation_factor = (1 + inflation_rate) ** 2
fees_per_term_2025 = {k: v * initial_inflation_factor for k, v in fees_per_term_2023.items()}

# Function to adjust fees for inflation
def adjust_for_inflation(fees, start_year, end_year, rate):
    adjusted_fees = {}
    for year in range(start_year, end_year + 1):
        adjustment_factor = (1 + rate) ** (year - start_year)
        adjusted_fees[year] = {k: v * adjustment_factor for k, v in fees.items()}
    return adjusted_fees

# Children's ages on September 1, 2025, including the potential fourth child
children = {
    'Cleopatra': 6,
    'Octavia': 4,
    'Athena': 3,
    'Fourth Child': 0
}

# Determine the end year based on the youngest child finishing Year 13
start_year = 2025
youngest_age_2025 = min(children.values())
youngest_finish_year = start_year + (18 - youngest_age_2025)

# Adjust fees for inflation from 2025 to the calculated end year
fees_per_term = adjust_for_inflation(fees_per_term_2025, start_year, youngest_finish_year, inflation_rate)

# Nursery fee details (assuming nursery fees also increase with inflation)
nursery_fee_per_day_2023 = 56.70
nursery_days_per_week = 3
nursery_weeks_per_term = 12

# Calculate nursery term fee for each year
nursery_term_fee_2025 = nursery_fee_per_day_2023 * nursery_days_per_week * nursery_weeks_per_term * initial_inflation_factor
nursery_term_fee = {year: nursery_term_fee_2025 * (1 + inflation_rate) ** (year - start_year) for year in range(start_year, youngest_finish_year + 1)}

# Sibling discounts
discounts = [0, 0.05, 0.10, 0.10]  # No discount for the first child, 5% for the second, 10% for the third and fourth

# Term names
terms = ['Winter', 'Spring', 'Summer']

# Calculate your age at each year
birth_date = datetime(1987, 7, 30)

# Generate table of data
data = {'Your Age': []}
sorted_children = sorted(children.keys(), key=lambda x: children[x], reverse=True)
for child in sorted_children:
    data[child] = []
data['Total'] = []
data['Annual Total'] = []
data['Monthly Cost'] = []
data['Target Gross Income'] = []

for year in range(start_year, youngest_finish_year + 1):
    for term in terms:  # 3 terms per year
        data['Your Age'].append(year - birth_date.year)
        term_total = 0
        
        # Determine current discounts based on the number of children attending
        children_attending = sorted([(child, age) for child, age in children.items() if age + (year - start_year) <= 18], key=lambda item: item[1], reverse=True)
        discount_map = {0: 0, 1: 0.05, 2: 0.10, 3: 0.10}

        for i, (child, age) in enumerate(children_attending):
            discount = discount_map.get(i, 0.10)
            current_age = age + (year - start_year)
            if current_age < 2:
                stage = None  # Not yet in nursery
            elif current_age < 4:
                stage = 'Nursery'
            elif current_age == 4:
                stage = 'Reception'
            elif current_age == 5:
                stage = 'Year 1'
            elif current_age == 6:
                stage = 'Year 2'
            elif current_age == 7:
                stage = 'Year 3'
            elif current_age == 8:
                stage = 'Year 4'
            elif current_age == 9:
                stage = 'Year 5'
            elif current_age == 10:
                stage = 'Year 6'
            elif current_age == 11:
                stage = 'Year 7'
            elif current_age == 12:
                stage = 'Year 8'
            elif current_age == 13:
                stage = 'Year 9'
            elif current_age == 14:
                stage = 'Year 10'
            elif current_age == 15:
                stage = 'Year 11'
            elif current_age == 16:
                stage = 'Year 12'
            elif current_age == 17:
                stage = 'Year 13'
            else:
                stage = None

            if stage == 'Nursery':
                term_fee = nursery_term_fee[year]
            elif stage:
                term_fee = fees_per_term[year][stage]
                term_fee *= (1 - discount)  # Apply discount
            else:
                term_fee = 0

            data[child].append(term_fee)
            term_total += term_fee

        data['Total'].append(term_total)
        if term == 'Summer':
            annual_total = sum(data['Total'][-3:])
            data['Annual Total'].append(annual_total)
            data['Monthly Cost'].append(annual_total / 12)
            target_gross_income = (annual_total / 0.20) / 0.70  # Assuming a 30% effective tax rate
            data['Target Gross Income'].append(round(target_gross_income, -4))  # Nearest £10k
        else:
            data['Annual Total'].append(0)
            data['Monthly Cost'].append(0)
            data['Target Gross Income'].append(0)

# Ensure all lists in data are the same length
max_length = len(data['Your Age'])
for key in data.keys():
    if len(data[key]) < max_length:
        data[key].extend([0] * (max_length - len(data[key])))

# Create DataFrame
df = pd.DataFrame(data)

# Calculate total sums for each column
total_sums = {child: df[child].sum() for child in sorted_children}
total_sums['Total'] = df['Total'].sum()
total_sums['Annual Total'] = df['Annual Total'].sum()
total_sums['Monthly Cost'] = total_sums['Annual Total'] / 12
total_sums['Your Age'] = ''
total_sums['Target Gross Income'] = ''

# Append total row to the DataFrame
total_row = pd.DataFrame(total_sums, index=['Total'])
df = pd.concat([df, total_row])

# Replace zeros with empty strings for display
df.replace(0, '', inplace=True)

# Format the price values as GBP
for col in sorted_children + ['Total', 'Annual Total', 'Monthly Cost', 'Target Gross Income']:
    df[col] = df[col].map(lambda x: f"£{x:,.2f}" if x != '' else x)

# Adjust display settings to show the full DataFrame
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Display the DataFrame
print(df)

# Generate a line graph showing the costs for each child and the total cost
df_graph = df.copy()
df_graph.replace('[£,]', '', regex=True, inplace=True)
df_graph.replace('', 0, inplace=True)
df_graph = df_graph.apply(pd.to_numeric)

plt.figure(figsize=(14, 8))
for child in sorted_children:
    plt.plot(df_graph.index[:-1], df_graph[child][:-1], label=child)
plt.plot(df_graph.index[:-1], df_graph['Total'][:-1], label='Total', linewidth=2, linestyle='--')

# Set x-axis labels to terms and years
x_labels = [f"{year} {term}" for year in range(start_year, youngest_finish_year + 1) for term in terms]
x_labels.append('Total')

plt.xticks(ticks=range(len(x_labels)), labels=x_labels, rotation=90)

plt.xlabel('Term')
plt.ylabel('Cost (£)')
plt.title('Inflation Adjusted Education Costs per Term Projection')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
