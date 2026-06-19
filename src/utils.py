import os
import io
import glob
import struct
from multiprocessing import Pool, cpu_count

from PIL import Image
from tqdm import tqdm

from tensorflow.core.example.example_pb2 import Example


def read_tfrecord(file_path: str):
    with open(file_path, "rb") as f:
        while True:
            length_bytes = f.read(8)
            if not length_bytes:
                break

            length = struct.unpack("<Q", length_bytes)[0]
            f.read(4)
            data = f.read(length)
            f.read(4)

            yield data


def parse_example(serialized: bytes):
    example = Example()
    example.ParseFromString(serialized)

    features = example.features.feature

    image = features["image"].bytes_list.value[0]
    label = features["class"].int64_list.value[0]
    img_id = features["id"].bytes_list.value[0].decode()

    return image, label, img_id


def parse_test_example(serialized: bytes):
    example = Example()
    example.ParseFromString(serialized)

    features = example.features.feature

    image = features["image"].bytes_list.value[0]
    img_id = features["id"].bytes_list.value[0].decode()

    return image, img_id


def process_tfrecord_file(file_path: str, output_path: str):
    local_count = 0

    for record in read_tfrecord(file_path):
        img_bytes, label, img_id = parse_example(record)

        class_dir = os.path.join(output_path, str(label))
        os.makedirs(class_dir, exist_ok=True)

        img = Image.open(io.BytesIO(img_bytes))
        img.save(os.path.join(class_dir, f"{img_id}.jpg"))

        local_count += 1

    return local_count


def process_test_tfrecord_file(file_path: str, output_path: str):
    import cv2
    import numpy as np

    count = 0

    for record in read_tfrecord(file_path):
        img_bytes, img_id = parse_test_example(record)

        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        cv2.imwrite(os.path.join(output_path, f"{img_id}.jpg"), img)

        count += 1

    return count


def _worker(args):
    file_path, output_path = args
    return process_tfrecord_file(file_path, output_path)


def _test_worker(args):
    file_path, output_path = args
    return process_test_tfrecord_file(file_path, output_path)


def extract_tfrecords(
    input_pattern: str,
    output_path: str,
    is_test: bool = False,
    max_workers: int | None = None,
):
    os.makedirs(output_path, exist_ok=True)
    files = glob.glob(input_pattern)

    if not files:
        raise FileNotFoundError(f"No TFRecord files found matching: {input_pattern}")

    num_workers = max_workers or min(8, cpu_count())
    worker_fn = _test_worker if is_test else _worker
    args_list = [(f, output_path) for f in files]

    with Pool(num_workers) as p:
        results = list(
            tqdm(p.imap(worker_fn, args_list), total=len(files))
        )

    total = sum(results)
    print(f"Total images processed: {total}")
    return total
