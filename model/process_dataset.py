import os
import glob
import librosa
import shutil


def count_total_duration(dataset_path, extension: str = ".wav"):
    print(f"Searching for {extension} files in {dataset_path}...")
    # glob recursive search
    files = glob.glob(os.path.join(dataset_path, "**", "*" + extension), recursive=True)
    print(f"Found {len(files)} {extension} files.")

    total_duration = 0
    for i, file_path in enumerate(files):
        try:
            # librosa.get_duration is faster than loading the whole audio
            duration = librosa.get_duration(path=file_path)
            total_duration += duration
            if (i + 1) % 100 == 0:
                print(
                    f"Processed {i + 1}/{len(files)} files. Current total: {total_duration:.2f}s"
                )
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print("-" * 30)
    print("Total Duration:")
    print(f"Seconds: {total_duration:.2f}")
    print(f"Minutes: {total_duration / 60:.2f}")
    print(f"Hours:   {total_duration / 3600:.2f}")


def flatten_dataset(dataset_path):
    print(f"Flattening dataset at {dataset_path}...")
    # Walk through the directory
    for root, dirs, files in os.walk(dataset_path, topdown=False):
        # Skip the root directory itself for source files
        if root == dataset_path:
            continue

        for file in files:
            if not file.endswith(".wav"):
                continue

            source_path = os.path.join(root, file)
            dest_path = os.path.join(dataset_path, file)

            # Handle name collisions
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(file)
                counter = 1
                while os.path.exists(dest_path):
                    # Try to include parent dir name to make it unique and meaningful
                    parent_dir = os.path.basename(root)
                    new_name = f"{base}_{parent_dir}_{counter}{ext}"
                    dest_path = os.path.join(dataset_path, new_name)
                    counter += 1

            try:
                shutil.move(source_path, dest_path)
                # print(f"Moved {source_path} -> {dest_path}")
            except Exception as e:
                print(f"Error moving {source_path}: {e}")

        # Remove empty directories
        if not os.listdir(root):
            try:
                os.rmdir(root)
                print(f"Removed empty directory: {root}")
            except Exception as e:
                print(f"Error removing directory {root}: {e}")

    print("Flattening complete.")


if __name__ == "__main__":
    # Assuming the script is run from the root of the repo or model dir,
    # but let's make it relative to this file to be safe.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Based on previous exploration, dataset is at model/data/VocalSet
    # If this script is in model/, then data/VocalSet is correct relative path
    dataset_path = os.path.join(current_dir, "data_instruments_small")
    instruments = ["piano", "voice", "guitar", "bass"]

    for instrument in instruments:
        instrument_path = os.path.join(dataset_path, instrument)
        # flatten_dataset(instrument_path)
        if instrument in ["piano", "bass"]:
            count_total_duration(instrument_path, extension=".mp3")
        else:
            count_total_duration(instrument_path, extension=".wav")
