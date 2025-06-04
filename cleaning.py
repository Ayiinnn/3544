import pandas as pd
from io import StringIO
import csv

def validate_row(raw_line):
    in_quotes = False
    comma_count = 0
    first_comma_pos = -1
    second_comma_pos = -1
    timestamp_valid = False
    for i, char in enumerate(raw_line.strip()):
        if char == '"':
            in_quotes = not in_quotes
        elif char == ',' and not in_quotes:
            comma_count += 1
            if comma_count == 1:
                first_comma_pos = i
                timestamp_part = raw_line[:i]
                timestamp_valid = (len(timestamp_part) >= 4 and
                                   timestamp_part[:4].isdigit())
            elif comma_count == 2:
                second_comma_pos = i
                break
    return (
            comma_count >= 2 and
            timestamp_valid and
            first_comma_pos < second_comma_pos
    )


def split_columns(raw_line):
    in_quotes = False
    comma_count = 0
    positions = []

    for i, char in enumerate(raw_line.strip()):
        if char == '"':
            in_quotes = not in_quotes
        elif char == ',' and not in_quotes:
            comma_count += 1
            if comma_count <= 2:
                positions.append(i)
            if comma_count == 2:
                break

    col1 = raw_line[:positions[0]].strip()
    col2 = raw_line[positions[0] + 1:positions[1]].strip()
    col3 = raw_line[positions[1] + 1:].strip()

    return (
        col1.strip('"'),
        col2.strip('"'),
        col3.strip('"')
    )

def clean_csv(input_path, output_path=None):

    valid_rows = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:

            if not line.strip():
                continue

            if validate_row(line):
                try:
                    cols = split_columns(line)
                    valid_rows.append(cols)
                except:
                    continue
    df = pd.DataFrame(valid_rows,
                      columns=['timestamp', 'username', 'content'])
    if output_path:
        df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    return df


cleaned_df = clean_csv("bitcoin-tweets-2021.csv", "cleaned_2021.csv")
cleaned_df2 = clean_csv("bitcoin-tweets-2022.csv", "cleaned_2022.csv")
