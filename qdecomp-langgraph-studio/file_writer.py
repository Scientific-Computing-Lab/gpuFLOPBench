#!/usr/bin/env python3
"""
Python script demonstrating how to create a new file and write to it.
This script shows various methods for file creation and writing operations.
"""

import os
from datetime import datetime


def create_and_write_basic():
    """Basic file creation and writing example."""
    filename = "output_basic.txt"
    
    # Create and write to a new file
    with open(filename, 'w') as file:
        file.write("Hello, World!\n")
        file.write("This is a basic file writing example.\n")
        file.write(f"Created on: {datetime.now()}\n")
    
    print(f"✅ Created and wrote to '{filename}'")


def create_and_write_with_content():
    """Create a file with more structured content."""
    filename = "output_structured.txt"
    
    content = [
        "=== Python File Writing Demo ===",
        "",
        "This file was created programmatically using Python.",
        "",
        "Features demonstrated:",
        "- File creation",
        "- Writing multiple lines",
        "- String formatting",
        "- Error handling",
        "",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "End of file."
    ]
    
    try:
        with open(filename, 'w') as file:
            for line in content:
                file.write(line + '\n')
        
        print(f"✅ Created structured file '{filename}'")
        
        # Read back and display the content
        with open(filename, 'r') as file:
            print(f"\n📄 Content of '{filename}':")
            print("-" * 40)
            print(file.read())
            print("-" * 40)
            
    except IOError as e:
        print(f"❌ Error creating file: {e}")


def create_csv_file():
    """Create a CSV file with sample data."""
    filename = "sample_data.csv"
    
    # Sample data
    headers = ["Name", "Age", "City", "Profession"]
    data = [
        ["Alice Johnson", "28", "New York", "Engineer"],
        ["Bob Smith", "34", "San Francisco", "Designer"],
        ["Carol Davis", "42", "Chicago", "Manager"],
        ["David Brown", "29", "Seattle", "Developer"]
    ]
    
    try:
        with open(filename, 'w') as file:
            # Write headers
            file.write(','.join(headers) + '\n')
            
            # Write data rows
            for row in data:
                file.write(','.join(row) + '\n')
        
        print(f"✅ Created CSV file '{filename}'")
        
    except IOError as e:
        print(f"❌ Error creating CSV file: {e}")


def create_json_file():
    """Create a JSON file with sample data."""
    import json
    
    filename = "sample_data.json"
    
    data = {
        "project": "File Writing Demo",
        "version": "1.0",
        "created_by": "Python Script",
        "timestamp": datetime.now().isoformat(),
        "users": [
            {
                "id": 1,
                "name": "Alice Johnson",
                "email": "alice@example.com",
                "active": True
            },
            {
                "id": 2,
                "name": "Bob Smith",
                "email": "bob@example.com",
                "active": False
            }
        ],
        "settings": {
            "debug": False,
            "max_users": 100,
            "features": ["logging", "authentication", "notifications"]
        }
    }
    
    try:
        with open(filename, 'w') as file:
            json.dump(data, file, indent=2)
        
        print(f"✅ Created JSON file '{filename}'")
        
    except IOError as e:
        print(f"❌ Error creating JSON file: {e}")


def create_in_subdirectory():
    """Create files in a subdirectory."""
    directory = "output_files"
    
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"📁 Created directory '{directory}'")
    
    filename = os.path.join(directory, "subdirectory_file.txt")
    
    try:
        with open(filename, 'w') as file:
            file.write("This file was created in a subdirectory.\n")
            file.write(f"Full path: {os.path.abspath(filename)}\n")
            file.write(f"Directory: {directory}\n")
        
        print(f"✅ Created file in subdirectory: '{filename}'")
        
    except IOError as e:
        print(f"❌ Error creating file in subdirectory: {e}")


def append_to_existing_file():
    """Demonstrate appending to an existing file."""
    filename = "append_example.txt"
    
    # Create initial file
    with open(filename, 'w') as file:
        file.write("Initial content\n")
        file.write("This is the first write operation\n")
    
    print(f"✅ Created initial file '{filename}'")
    
    # Append to the file
    with open(filename, 'a') as file:
        file.write("\n--- Appended Content ---\n")
        file.write("This content was appended later\n")
        file.write(f"Appended at: {datetime.now()}\n")
    
    print(f"✅ Appended content to '{filename}'")
    
    # Read and display the complete file
    with open(filename, 'r') as file:
        print(f"\n📄 Final content of '{filename}':")
        print("-" * 40)
        print(file.read())
        print("-" * 40)


def main():
    """Main function to run all examples."""
    print("🐍 Python File Creation and Writing Examples")
    print("=" * 50)
    
    # Run all examples
    create_and_write_basic()
    print()
    
    create_and_write_with_content()
    print()
    
    create_csv_file()
    print()
    
    create_json_file()
    print()
    
    create_in_subdirectory()
    print()
    
    append_to_existing_file()
    print()
    
    print("🎉 All file operations completed successfully!")
    print("\nFiles created:")
    files_created = [
        "output_basic.txt",
        "output_structured.txt", 
        "sample_data.csv",
        "sample_data.json",
        "output_files/subdirectory_file.txt",
        "append_example.txt"
    ]
    
    for file in files_created:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"  📄 {file} ({size} bytes)")


if __name__ == "__main__":
    main()
