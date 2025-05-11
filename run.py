import os
import time

MODULES = {
    # Build
    '1': 'build.api',
    '2': 'build.pre_render_dataset',
    '3': 'build.resize_image',
    '4': 'build.visualize_batch',

    # Shared Modules
    '7': 'src.shared.preprocessing.inkml_loader',

    # Mathwriting modules
    '10': 'src.mathwriting.preprocessing.tokenizer',
    '11': 'src.mathwriting.scripts.trainer',

    # Mathsolver modules
    '15': 'src.mathsolver.datamodule.create_dataset',
    '16': 'src.mathsolver.preprocessing.tokenizer',
    '17': 'src.mathsolver.datamodule.dataloader',
    '18': 'src.mathsolver.scripts.trainer',

    # Image2Latex modules
    '20': 'src.image2latex.preprocessing.generate_latex_file',
    '21': 'src.image2latex.datamodule.dataloader',
    '22': 'src.image2latex.scripts.trainer',
    '23': 'src.image2latex.scripts.mw_trainer',}

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