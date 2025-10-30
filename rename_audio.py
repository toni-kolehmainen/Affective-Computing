import os
import csv
import shutil

# === CONFIGURATION ===
csv_path = "video_labels.csv"   # path to your CSV file
folder_path = "audio"    # folder containing the files to rename
has_header = True                  # set to False if your CSV has no header

# CSV format:
# original_filename,new_filename
# example:
# old_photo1.jpg,new_photo1.jpg
# old_photo2.png,new_photo2.png

# === SCRIPT ===
def rename_videos(csv_path, folder_path, use_emotion_subfolders=False):
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder '{folder_path}' does not exist.")
        return

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        renamed_count = 0
        missing_count = 0

        for row in reader:
            original_name = row["original_name"].strip() + ".wav"
            new_name = row["video_name"].strip().removesuffix(".mp4") + ".wav"
            emotion = row.get("emotion", "").strip()

            original_path = os.path.join(folder_path, original_name)

            # Optionally organize by emotion
            if use_emotion_subfolders and emotion:
                target_folder = os.path.join(folder_path, emotion)
                os.makedirs(target_folder, exist_ok=True)
                new_path = os.path.join(target_folder, new_name)
            else:
                new_path = os.path.join(folder_path, new_name)

            if not os.path.exists(original_path):
                print(f"❌ File not found: {original_name}")
                missing_count += 1
                continue

            try:
                shutil.move(original_path, new_path)
                print(f"✅ Renamed: {original_name} → {new_name}")
                renamed_count += 1
            except Exception as e:
                print(f"⚠️ Error renaming '{original_name}': {e}")

        print(f"\n✅ Done! Renamed {renamed_count} files. Missing: {missing_count}.")

if __name__ == "__main__":
    rename_videos(csv_path, folder_path, has_header)
