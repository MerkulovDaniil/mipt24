import re

def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Replace frontmatter
    content = re.sub(r'---.*?format:\s*beamer.*?---', 
                     r'---\nformat:\n    pdf:\n        pdf-engine: xelatex\n        include-in-header: ../files/docheader.tex  # Custom LaTeX commands and preamble\n---',
                     content, flags=re.DOTALL)

    # Remove new slide delimiter: ". . ."
    content = re.sub(r'. . .\n', '', content)

    # Replace step-by-step formulas with the desired format
    content = re.sub(r'\$\$(.*?)\$\$', lambda m: re.sub(r'\\uncover<.*?>{', '', m.group(1).replace('}', '')), content)

    # Write the modified content to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(content)

if __name__ == "__main__":
    input_filename = '1.md'
    output_filename = '1_doc.md'
    process_file(input_filename, output_filename)
    print(f"Processed file saved as {output_filename}")
