import os
import time

MODULES = {
    # Shared Modules
    '1': 'src.shared.preprocessing.latex_tokenizer',
    '2': 'src.shared.preprocessing.lmdb_writer',
    '3': 'src.shared.preprocessing.inkml_loader',
    '4': 'src.shared.preprocessing.main',
    
    # Symbol Detection Module
    # '3': 'src.symbol_detection.scripts.trainer',

    # # Mathwriting modules
    # '1': 'src.mathwriting.scripts.trainer',
    # '2': 'src.mathwriting.scripts.detector',
    # '3': 'src.mathwriting.preprocessing.main',
    # '4': 'src.mathwriting.preprocessing.inkml_dataset',
    # '5': 'src.mathwriting.preprocessing.math_expression_dataset',
    
    # # Digit recognition modules
    # '5': 'src.digit_recognition.scripts.main',
    # '6': 'src.digit_recognition.scripts.preprocessor',
    # '7': 'src.digit_recognition.scripts.detector',
}

def print_modules():
    print("\nAvailable modules:")
    print("-" * 50)
    for num, module in MODULES.items():
        name = module.split('.')[-1].replace('_', ' ').title()
        print(f"{num}. {name} ({module})")
    print("-" * 50)

def run_module(choice):
    if choice in MODULES:
        command = f"python -m {MODULES[choice]}"
        print(f"\nExecuting: {command}")
        os.system(command)
    else:
        print("Invalid choice! Please try again.")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')
    time.sleep(0.1)

def main():
    while True:
        print_modules()
        choice = input("\nEnter the number of the module to run (0 to exit): ")
        
        if choice == '0':
            print("Exiting...")
            break
            
        run_module(choice)
        
        print("\nPress any key to exit or Enter to continue...")
        key = input()
        if key:
            break
        clear_screen()

if __name__ == "__main__":
    main() 