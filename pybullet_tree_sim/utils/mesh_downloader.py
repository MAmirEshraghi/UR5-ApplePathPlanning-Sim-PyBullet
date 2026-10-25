#!/usr/bin/env python3
from pybullet_tree_sim import MESHES_PATH

import os
import requests
from tqdm import tqdm
import zipfile


meshes_path = MESHES_PATH  # follows PYBULLET_TREE_SIM_DATA; was derived from a site-packages-only folder depth


def get_filename_from_url(url: str) -> str:
    local_filename = url.split("/")[-1]
    return os.path.join(meshes_path, local_filename)


def is_download_needed(target_file_path: str) -> bool:
    if not os.path.exists(meshes_path):
        print(f"Creating path {meshes_path}")
        os.mkdir(meshes_path)
        return True

    if not os.path.exists(os.path.join(meshes_path, "trees")):
        if not os.path.isfile(os.path.join(meshes_path, "pybullet-tree-sim-meshes.zip")):  # TODO: pass name into func
            return True
        else:
            print(f"File {os.path.join(meshes_path, 'pybullet-tree-sim-meshes.zip')} already exists")
            return False
    else:
        print(f"Path {os.path.join(meshes_path, 'trees')} already exists.")
        return False


def is_unzip_needed(target_file_path: str) -> bool:
    if not os.path.exists(os.path.join(meshes_path, "trees")):
        return True
    else:
        return False


def download_file(url: str, target_file_path: str) -> bool:
    try:
        # NOTE the stream=True parameter below
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
                with open(target_file_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        # If you have chunk encoded response uncomment if
                        # and set chunk_size parameter to None.
                        # if chunk:
                        progress_bar.update(len(chunk))
                        f.write(chunk)

                    return True

    except Exception as e:
        print(f"{e}")
        return False


def unzip(zip_file: str):
    print(f"Extracting file {zip_file}")
    with zipfile.ZipFile(zip_file, "r") as zipper:
        zipper.extractall(os.path.dirname(zip_file))
    print(f"Files extracted")

    if not os.path.exists(meshes_path):
        print(f"Cannot find path {meshes_path}, error with process. Try running the file directly.")
    return


def main():
    url = "https://zenodo.org/records/14991250/files/pybullet-tree-sim-meshes.zip"

    file_abs_path = get_filename_from_url(url=url)

    download_needed = is_download_needed(target_file_path=file_abs_path)
    if download_needed:
        download_file(url=url, target_file_path=file_abs_path)

    unzip_needed = is_unzip_needed(target_file_path=file_abs_path)
    if unzip_needed:
        unzip(zip_file=file_abs_path)
    return


if __name__ == "__main__":
    main()
