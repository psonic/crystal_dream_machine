#!/usr/bin/env python3
"""
Script per rimuovere le funzioni SVG/PDF dal file principale
"""

def remove_svg_pdf_functions():
    # Leggi il file originale
    with open('natisone_trip_generator.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Trova le funzioni SVG/PDF da rimuovere
    functions_to_remove = [
        'def get_svg_dimensions(',
        'def extract_contours_from_svg(',
        'def extract_contours_from_svg_fallback(',
        'def extract_contours_from_pdf('
    ]
    
    new_lines = []
    skip_until_next_def = False
    current_function = None
    
    for i, line in enumerate(lines):
        # Controlla se questa riga inizia una funzione da rimuovere
        should_remove = False
        for func in functions_to_remove:
            if line.strip().startswith(func):
                should_remove = True
                current_function = func
                print(f"Rimuovendo funzione: {func} alla riga {i+1}")
                break
        
        if should_remove:
            skip_until_next_def = True
            continue
        
        # Se stiamo saltando, controlla se inizia una nuova funzione o classe
        if skip_until_next_def:
            # Se troviamo una nuova def o class che non è una funzione da rimuovere
            if (line.strip().startswith('def ') or line.strip().startswith('class ')) and line.strip() != '':
                # Controlla se è una delle funzioni da rimuovere
                is_function_to_remove = False
                for func in functions_to_remove:
                    if line.strip().startswith(func):
                        is_function_to_remove = True
                        current_function = func
                        print(f"Rimuovendo anche funzione: {func} alla riga {i+1}")
                        break
                
                if not is_function_to_remove:
                    skip_until_next_def = False
                    new_lines.append(line)
            continue
        
        # Mantieni la riga
        new_lines.append(line)
    
    # Scrivi il nuovo file
    with open('natisone_trip_generator.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"✅ File aggiornato: {len(lines)} -> {len(new_lines)} righe (-{len(lines) - len(new_lines)})")

if __name__ == "__main__":
    remove_svg_pdf_functions()
