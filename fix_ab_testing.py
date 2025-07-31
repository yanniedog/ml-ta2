"""
Script to fix syntax errors in ab_testing.py
"""

def fix_ab_testing_file():
    # Read the file content
    with open('src/ab_testing.py', 'r') as f:
        content = f.read()
    
    # Find and fix duplicate _get_metric_values method
    # We'll keep only one implementation
    lines = content.split('\n')
    fixed_lines = []
    in_first_get_metric_values = False
    skip_until_next_def = False
    
    for line in lines:
        if skip_until_next_def:
            if line.strip().startswith('def ') and not line.strip().startswith('def _get_metric_values'):
                skip_until_next_def = False
            else:
                continue
                
        if line.strip().startswith('def _get_metric_values'):
            if in_first_get_metric_values:
                # This is the second occurrence, skip it
                skip_until_next_def = True
                continue
            else:
                # This is the first occurrence
                in_first_get_metric_values = True
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    # Join the lines back together
    fixed_content = '\n'.join(fixed_lines)
    
    # Make sure all triple quotes are properly terminated
    # This is a simple check - if there's an odd number of triple quotes, we need to fix it
    triple_quotes_count = fixed_content.count('"""')
    if triple_quotes_count % 2 != 0:
        print(f"Found {triple_quotes_count} triple quotes (odd number)")
        # Find the position of the last triple quote
        last_pos = fixed_content.rfind('"""')
        # Add a closing triple quote at the end of that line
        lines = fixed_content.split('\n')
        for i, line in enumerate(lines):
            if '"""' in line and i < len(lines) - 1:
                # Check if this line has an opening quote without a closing quote
                if line.count('"""') % 2 != 0:
                    print(f"Fixing line {i}: {line}")
                    # Add closing quote at the end of the line
                    lines[i] = line + ' """'
        
        fixed_content = '\n'.join(lines)
    
    # Write the fixed content back to the file
    with open('src/ab_testing.py', 'w') as f:
        f.write(fixed_content)
    
    print("File has been fixed.")

if __name__ == "__main__":
    fix_ab_testing_file()
