import os
import cv2
import argparse
import numpy as np
from common import *
from tqdm import tqdm
from itertools import permutations


def binarize(x: np.ndarray, t=0.5) -> np.ndarray:
    """배열을 이진화한다.
    모든 요소의 값이 [0, 1]에 속하는 배열을 입력으로 받아, 문턱값보다 크면 True, 작으면 False를 리턴한다.

    Parameters
    ----------
    x : np.ndarray
        이미지, 혹은 임의의 numpy.ndarray
    t : float, optional
        문턱값, by default 0.5

    Returns
    -------
    binary_image : np.ndarray
        이진화된 배열
    """

    y = x.copy()
    return (y > t).astype(np.bool_)


def binarize_image(x: np.ndarray, threshold: float):
    """이미지를 이진 이미지로 변환한다.

    Parameters
    ----------
    x : np.ndarray
        BGR 3채널의 이미지
    t : float, optional
        문턱값, by default 0.5

    Returns
    -------
    binary_image : np.ndarray
        이진화된 이미지
    """
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)  # 이미지 흑백화(채널 단일화: 3 -> 1)
    x = x.astype(np.float64) / 255  # 0 ~ 1 사이의 실수값으로 스케일링
    return binarize(x, threshold)  # 이미지 이진화


def binarize_video(video_path: str, threshold=0.5) -> np.ndarray:
    """이진화된 영상을 계산한다.

    Parameters
    ----------
    video_path : str
        영상 파일 경로
    threshold : float, optional
        문턱값, by default 0.5

    Returns
    -------
    binary_image : np.ndarray
        이진화된 이미지
    """

    text = "Binarizing Images"

    binary_images = video_apply(video_path, binarize_image, text, threshold=threshold)
    binary_images = np.stack(binary_images)
    return binary_images


def calculate_background(images: np.ndarray, threshold: float) -> np.ndarray:
    """동영상의 배경 이미지를 계산한다.

    Parameters
    ----------
    images : np.ndarray
        (n_frames, height, width) 형태의 연속된 이미지 배열

    Returns
    -------
    background : np.ndarray
        배경 이미지(흑백)
    """

    # 평균으로 단일 이미지를 계산한다.
    background = images.mean(axis=0)

    # 필터를 적용해 제거 범위를 더 넓힌다.
    k = 5
    kernel = np.ones((k, k)) / k**2
    background = cv2.filter2D(background, -1, kernel)

    # 배경 이미지를 이진화
    background = binarize(background, threshold)
    return background


def remove_background(x: np.ndarray, background: np.ndarray) -> np.ndarray:
    """이미지에서 배경을 제거한다."""

    assert x.dtype == background.dtype == np.bool_
    if len(x.shape) > len(background.shape):
        background = background[np.newaxis, ...]

    return x > background  # 어떤 픽셀에 대해, 배경에는 신호가 없고 이미지에만 신호가 있으면 배경이 아닌 마커이다.


def find_markers(image: np.ndarray, area_min_size: int = 1) -> np.ndarray:
    """이미지에서 마커를 찾는다.
    신호가 있는 픽셀들이 연속되어 하나의 영역을 이룰 경우 이를 묶어서 하나의 점으로 계산한다.

    Parameters
    ----------
    image : np.ndarray
        대상 이미지
    area_min_size : int, optional
        영역의 최소 넓이. 이 값보다 넓이가 적은 영역은 무시한다. 기본값은 1.

    Returns
    -------
    markers : np.ndarray
        마커 좌표
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    markers = []
    for c in contours:
        if len(c) >= area_min_size:
            markers.append(c.reshape(-1, 2).mean(axis=0).astype(np.int32))

    return np.array(markers)


def main(video_path, output_path, threshold, background_threshold, area_min_size):
    # 이진화된 영상을 로드
    images = binarize_video(video_path, threshold)

    # 이진 영상의 배경을 계산한다.
    background = calculate_background(images, background_threshold)

    # 배경을 제거한 이미지를 계산한다.
    print("Removing Background...", end=" ")
    images = remove_background(images, background)
    print("Done!")

    # 영상으로 저장하기 위해 형식 변경
    images = images.astype(np.uint8) * 255

    # 원본 영상의 정보 가져오기
    cap = cv2.VideoCapture(video_path)
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    delay = round(1000 / fps)
    cap.release()

    os.makedirs(os.path.split(output_path)[0], exist_ok=True)

    # 영상 내보내기
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h), 0)
    progress = tqdm(images)
    for image in progress:
        out.write(image)
        progress.set_description("Writing Video")

    out.release()

    # 완벽히 이진화된 데이터가 아닌 인코딩된 영상에서 데이터를 가져오므로 다시 이진화한다.(잡음 제거)
    func = lambda x: find_markers(
        ((cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) > 200) * 255).astype(np.uint8),
        area_min_size=area_min_size,
    )

    markers = []
    frame_numbers = []

    found_points = video_apply(output_path, func, "Detecting Markers...")

    print(
        "The number of frames that captured markers' count is valid:",
        len(list(filter(lambda x: len(x) > 0, found_points))),
    )

    for i, points in enumerate(found_points):
        if points is None or len(points) == 0:
            continue

        frame_numbers.append(i)
        markers.append(points)

    print("Detected Frame Count:", len(frame_numbers))

    os.makedirs(os.path.split(output_path)[0], exist_ok=True)
    np.save(output_path, {"frame_numbers": frame_numbers, "markers": markers})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("-t", "--threshold", type=float, default=0.5)
    parser.add_argument("-bt", "--background-threshold", type=float, default=0.3)
    parser.add_argument("--area-min-size", type=int, default=3)
    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        raise InvalidArgumentError("입력 영상 경로가 잘못되었습니다.")
    if not (0 <= args.threshold <= 1):
        raise InvalidArgumentError("이미지 이진화 문턱값은 구간 [0, 1]에 존재해야 합니다.")
    if args.area_min_size < 1:
        raise InvalidArgumentError("최소 영역 넓이는 1 이상이어야 합니다.")

    main(args.video_path, args.output_path, args.threshold, args.background_threshold, args.area_min_size)
