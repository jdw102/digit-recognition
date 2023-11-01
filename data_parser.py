
def parse(name, num_blocks):
    data = {}
    # Initialize an empty list to store the separated blocks of text
    text_blocks = []

    # Initialize an empty string to accumulate lines between blank lines
    current_block = ""

    # Open the text file for reading
    with open(name, "r") as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if line:
                current_block += line + "\n"  # Add the line to the current block
            else:
                # Encountered a blank line, add the current block to the list
                if current_block:
                    text_blocks.append(current_block.strip())  # Remove trailing newline
                    current_block = ""  # Reset the current block

    # Add the last block if it's not empty
    if current_block:
        text_blocks.append(current_block.strip())
    for index, block in enumerate(text_blocks):
        digit = index // num_blocks
        gender = "male" if (index % num_blocks) < num_blocks // 2 else "female"
        if digit not in data:
            data[digit] = {}
        if gender not in data [digit]:
            data[digit][gender] = []
        data[digit][gender].append([line.split(" ") for line in block.split("\n")])

    return data
