import os
import glob
from PIL import Image

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, nargs='+', help='Directory with frames', required=True)
    parser.add_argument('--duration', '-dur', type=int, default=30, help="Duration of the gif or video in seconds.")
    args = parser.parse_args()

    for cwd in args.dir:
        cwd = cwd.replace('\'', '')
        cwd = os.path.realpath(cwd)
        assert os.path.exists(cwd), f"Path {cwd} doesn't exist!"
        print(f"Reading from {cwd}")
        gif_save_path = cwd

        print(f"Creating gif and saving it to {cwd}")
        frames = [Image.open(image) for image in sorted(glob.glob(f"{cwd}/*.jpg"))]
        assert len(frames) > 0, f"No frames in frame directory {cwd}."
        frames[0].save(f"{gif_save_path}_{args.duration}s.gif", format="GIF", append_images=frames, save_all=True,
                       duration=args.duration * 1000 / len(frames), loop=0)
        if False:
            os.system(
                f"ffmpeg -f image2 -r {args.duration} -i {cwd}/* "
                f"-vcodec mpeg4 -y {gif_save_path}.mp4")
