import os

def get_python_module_names(directory):
    """
    Returns a list of Python module names (without .py extension) in the given directory.
    """
    return [
        os.path.splitext(f)[0]
        for f in os.listdir(directory)
        if f.endswith('.py') and os.path.isfile(os.path.join(directory, f))
    ]

# Example usage:
# module_names = get_python_module_names('/path/to/your/directory')

def create_md_files_from_list(strings, directory):
    """
    For each string in the list, creates a .md file named after the string,
    and writes the string as the content of the file.
    """
    os.makedirs(directory, exist_ok=True)
    strings.sort()
    for s in strings:
        if s != '__init__':
            filename = f"{s}.md"
            filepath = os.path.join(directory, filename)
            with open(filepath, 'w') as f:
                f.write(f"# {s}\n\n")
                f.write("::: " + s)
            f.close()

            print(f'- "{s}": {s}.md')

# Example usage:
# names = ['foo', 'bar',

def create_md_file_from_list(strings, directory):
    """
    For each string in the list, creates a .md file named after the string,
    and writes the string as the content of the file.
    """
    os.makedirs(directory, exist_ok=True)
    strings.sort()
    filename = f"stencils.md"
    filepath = os.path.join(directory, filename)
    f = open(filepath, 'w')
    for s in strings:
        if s != '__init__':
            f.write(f"# {s}\n\n")
            f.write(f"::: stencils.{s}\n\n")

            print(f'- "{s}": stencils/stencils/#{s}')
    f.close()

file_names = get_python_module_names('/Users/ckung/Documents/Code/MkDocs_playground/PyFV3_CK_fork/pyFV3/stencils')
print(file_names)

create_md_file_from_list(file_names, '/Users/ckung/Documents/Code/MkDocs_playground/PyFV3_CK_fork/pyFV3/pyFV3_docs/docs/stencils')