import os
import time

MODULES = {
    # Build
    '1': 'build.api',
    '2': 'build.solver',
    '3': 'build.pre_render_dataset',
    '4': 'build.visualize_batch',
    '5': 'build.visualize_model',

    # Shared Modules
    '6': 'src.shared.preprocessing.latex_tokenizer',
    '7': 'src.shared.preprocessing.inkml_loader',

    # Mathwriting modules
    '8': 'src.mathwriting.scripts.trainer',
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