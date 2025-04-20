import os

def reassemble_model(chunk_prefix="model_chunk_", folder="goemotions_model/chunks", output_path="../model.safetensors"):
    # Ensure the folder exists
    if not os.path.exists(folder):
        print(f"⚠️ The folder {folder} does not exist.")
        return

    # Ensure the output file is within the same folder
    output_path = os.path.join(folder, output_path)
    
    # Check if the output file already exists
    if not os.path.exists(output_path):
        print("🔧 Reassembling model from chunks...")
        with open(output_path, 'wb') as out_file:
            i = 0
            while True:
                # Construct the chunk filename (e.g., model_chunk_aa, model_chunk_ab, ...)
                chunk_file = os.path.join(folder, f"{chunk_prefix}{chr(97 + i // 26)}{chr(97 + i % 26)}")
                if not os.path.exists(chunk_file):
                    print(f"⚠️ Missing chunk: {chunk_file}")
                    break
                else:
                    print(f"Reading chunk: {chunk_file}")
                    with open(chunk_file, 'rb') as f:
                        chunk_data = f.read()
                        if len(chunk_data) > 0:
                            out_file.write(chunk_data)
                        else:
                            print(f"⚠️ Empty chunk detected: {chunk_file}")
                i += 1
        
        # Check if the file size matches the expected size
        reassembled_size = os.path.getsize(output_path)
        print(f"Reassembled model size: {reassembled_size} bytes")
        
        if reassembled_size > 0:
            print("✅ Model reassembled successfully.")
        else:
            print("⚠️ Model reassembly failed, size is 0 bytes!")
    else:
        print(f"⚠️ {output_path} already exists. Skipping reassembly.")

# Usage: Reassemble the model chunks into the same folder (goemotions_model)

