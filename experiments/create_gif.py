import os
import glob
from PIL import Image

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, help='Directory with frames', required=True)
    parser.add_argument('--duration', '-dur', type=int, default=30, help="Duration of the gif or video in seconds.")
    args = parser.parse_args()

    cwd = os.path.realpath(args.dir)
    print(f"Reading from {cwd}")
    # parent_dir, dir_name = os.path.split(cwd)
    gif_save_path = cwd

    print(f"Creating gif and saving it to {args.dir}")
    frames = [Image.open(image) for image in sorted(glob.glob(f"{cwd}/*.jpg"))]
    assert len(frames) > 0, f"No frames in frame directory {cwd}."
    frames[0].save(f"{gif_save_path}.gif", format="GIF", append_images=frames, save_all=True,
                   duration=args.duration * 1000 / len(frames), loop=0)
    if False:
        os.system(
            f"ffmpeg -f image2 -r {args.duration} -i {cwd}/* "
            f"-vcodec mpeg4 -y {gif_save_path}.mp4")
