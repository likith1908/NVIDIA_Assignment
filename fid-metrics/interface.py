import gradio as gr
import subprocess
import os
import shutil

# Define a function to process the videos and return FID and FVD scores
def calculate_scores(video1, video2):
    # Save the uploaded videos locally
    if not video1 or not video2:
        return "Error: Both video files are required.", "Error"

    input_dir = "uploaded_videos"
    os.makedirs(input_dir, exist_ok=True)

    video1_path = os.path.join(input_dir, "video1.mp4")
    video2_path = os.path.join(input_dir, "video2.mp4")

    shutil.copy(video1, video1_path)
    shutil.copy(video2, video2_path)

    # Run the existing pipeline with the videos
    cmd = f"python main.py +paths=['{video1_path}','{video2_path}']"
    try:
        result = subprocess.check_output(cmd, shell=True, text=True)
        # Parse the output to extract FID and FVD scores
        fid_score = None
        fvd_score = None

        for line in result.splitlines():
            if "FID:" in line:
                fid_score = line.split(":")[1].strip()
            elif "FVD:" in line:
                fvd_score = line.split(":")[1].strip()

        if fid_score and fvd_score:
            return fid_score, fvd_score
        else:
            return "Error: Could not calculate scores.", "Error"
    except subprocess.CalledProcessError as e:
        return f"Error: {e.output}", "Error"
    finally:
        # Cleanup uploaded files
        shutil.rmtree(input_dir)

# Create the Gradio interface
gr.Interface(
    fn=calculate_scores,
    inputs=[
        gr.Video(label="Upload Video 1"),
        gr.Video(label="Upload Video 2")
    ],
    outputs=[
        gr.Textbox(label="FID Score"),
        gr.Textbox(label="FVD Score")
    ],
    title="FID and FVD Score Calculator",
    description="Upload two video files to calculate FID and FVD scores."
).launch(share=True)
