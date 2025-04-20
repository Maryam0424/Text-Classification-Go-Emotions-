import os

def split_file(file_path, chunk_size=20 * 1024 * 1024, chunk_prefix="model_chunk_"):
    # Get the folder where the file is located
    parent_folder = os.path.dirname(file_path)
    chunk_folder = os.path.join(parent_folder, "chunks")
    
    # Create the chunks folder if it doesn't exist
    os.makedirs(chunk_folder, exist_ok=True)
    
    # Open the original file for reading
    with open(file_path, 'rb') as f:
        i = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            # Generate chunk filename (e.g., model_chunk_aa, model_chunk_ab, ...)
            chunk_filename = os.path.join(chunk_folder, f"{chunk_prefix}{chr(97 + i // 26)}{chr(97 + i % 26)}")
            with open(chunk_filename, 'wb') as chunk_file:
                chunk_file.write(chunk)
            print(f"✅ Created chunk: {chunk_filename}")
            i += 1

# Usage
split_file('./goemotions_model/model.safetensors')
