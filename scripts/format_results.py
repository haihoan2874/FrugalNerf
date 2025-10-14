import sys
import csv

if len(sys.argv) < 2:
    print('Usage: python scripts/format_results.py <results.csv>')
    sys.exit(1)

csv_path = sys.argv[1]
rows = []
with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for r in reader:
        rows.append(r)

# Generate markdown
print('| ' + ' | '.join(header) + ' |')
print('| ' + ' | '.join(['---']*len(header)) + ' |')
for r in rows:
    print('| ' + ' | '.join(r) + ' |')
