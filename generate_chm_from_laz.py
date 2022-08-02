import subprocess
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Visualization and analysis of LAS/LAZ point cloud files')

    parser.add_argument('filename', type=str, help='Name of LAS/LAZ file')
    parser.add_argument(
        '--root', type=str, default=r"C:\Users\aisl\Documents\dataset\\", help='Name of LAS/LAZ file')

    return parser.parse_args()


def generate_chm(args):
    print(args.filename)
    path = args.root + args.filename
    temp_path = args.root + "temp.laz"
    # subprocess.run(["/mnt/c/LAStools/bin/lasview.exe", "-i", path, "-points", "5000000"])

    # 1. Remove noise
    # /mnt/c/LAStools/bin/lasnoise.exe -cpu64 -i "C:\Users\aisl\Documents\dataset\Evo_HeliALS-TW_2021_euroSDR\1005.laz" \
    #  -isolated 10 -step_xy 0.1 -step_z 0.1 -classify_as 7 -o "C:/Users/aisl/Documents/dataset/Evo_HeliALS-TW_2021_euroSDR/1005_tmp.laz" \
    #  -remove_noise
    subprocess.run(["/mnt/c/LAStools/bin/lasnoise.exe", "-cpu64", "-i", path, "-isolated", "10",
        "-step_xy", "0.1", "-step_z", "0.05",
       "-classify_as", "7", "-o", "tmp/temp_denoise.laz", "-remove_noise"])

    # 2. Ground normalization
    # /mnt/c/LAStools/bin/lasground_new.exe -cpu64 -i "C:\Users\aisl\Documents\dataset\Evo_HeliALS-TW_2021_euroSDR\1005_noise.laz"
    # -ignore_class 4 -wilderness -fine -o "C:/Users/aisl/Documents/dataset/Evo_HeliALS-TW_2021_euroSDR/1005_norm.laz" -replace_z
    subprocess.run(["/mnt/c/LAStools/bin/lasground_new.exe", "-cpu64", "-i", "tmp/temp_denoise.laz", "-ignore_class", "4",
        "-wilderness", "-fine", "-o", "tmp/temp_norm.laz", "-replace_z"])

    # 3. Generate CHM
    # /mnt/c/LAStools/bin/lasgrid -i "C:\Users\aisl\Documents\dataset\Evo_HeliALS-TW_2021_euroSDR\1005_norm.laz"
    # -step 0.2 -highest -o "C:/Users/aisl/Documents/dataset/Evo_HeliALS-TW_2021_euroSDR/1005_CHM.png" -subcircle 0.1 -set_min_max 0 50
    chm_filename = args.filename.rsplit('\\', 1)[1].split('.')[0] + "_CHM.png"
    print(chm_filename)
    subprocess.run(["/mnt/c/LAStools/bin/lasgrid.exe", "-cpu64", "-i", "tmp/temp_norm.laz",
        "-step", "0.2", "-highest", "-o", "CHM/" + chm_filename, "-subcircle", "0.1", "-set_min_max", "0", "50"])


def main():
    args = get_arguments()
    generate_chm(args)


if __name__=='__main__':
    main()
