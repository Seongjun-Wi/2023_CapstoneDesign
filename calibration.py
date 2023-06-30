import os
import argparse
import numpy as np
from common import *
from tqdm import tqdm
from scipy.optimize import least_squares
from utils.camera import (
    estimate_homography,
    calculate_relative_extr,
    calculate_reprojection_error,
    NotPerpendicularError,
)
from utils.linalg import exp_map, log_map


def estimate_intr_params(H_mats):
    N_image = H_mats.shape[0]

    h = lambda k, i, j: H_mats[k, j - 1, i - 1]
    v = lambda k, i, j: np.array(
        [
            [
                h(k, i, 1) * h(k, j, 1),
                h(k, i, 1) * h(k, j, 2) + h(k, i, 2) * h(k, j, 1),
                h(k, i, 2) * h(k, j, 2),
                h(k, i, 3) * h(k, j, 1) + h(k, i, 1) * h(k, j, 3),
                h(k, i, 3) * h(k, j, 2) + h(k, i, 2) * h(k, j, 3),
                h(k, i, 3) * h(k, j, 3),
            ]
        ]
    ).T

    V = np.concatenate(
        [
            np.concatenate([v(i, 1, 2).T, (v(i, 1, 1) - v(i, 2, 2)).T], axis=0)
            for i in range(H_mats.shape[0])
        ]
    )

    if N_image == 2:
        V = np.concatenate([V, np.array([[0, 1, 0, 0, 0, 0]])], axis=0)

    w, v = np.linalg.eig(V.T @ V)
    b = v[:, np.argmin(w)]
    b11, b12, b22, b13, b23, b33 = b
    v0 = (b12 * b13 - b11 * b23) / (b11 * b22 - b12**2)
    lambda_ = b33 - (b13**2 + v0 * (b12 * b13 - b11 * b23)) / b11
    alpha = np.sqrt(lambda_ / b11)
    beta = np.sqrt(lambda_ * b11 / (b11 * b22 - b12**2))
    gamma = -b12 * alpha**2 * beta / lambda_
    u0 = gamma * v0 / beta - b13 * alpha**2 / lambda_
    # gamma = 0
    # u0 = gamma * v0 / beta - b13 * alpha**2 / lambda_
    # K = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])
    K = np.array([[alpha, 0, u0], [0, beta, v0], [0, 0, 1]])

    return K


def calibrate_intr(
    points_2d: np.ndarray,
    points_3d: np.ndarray,
    H_mats: np.ndarray,
    n_iter: int = 100,
    n_samples: int = -1,
    inlier_error: float = 10,
) -> np.ndarray:
    """RANSAC으로 카메라 내부 파라미터 행렬 K를 계산한다.

    Parameters
    ----------
    points_2d : np.ndarray
        매 프레임마다 영상에서 촬영된 점. (T, 2, N)
    points_3d : np.ndarray
        실제 점 좌표. (3, N)
    H_mats : np.ndarray
        매 프레임마다 계산된 호모그라피 행렬. (T, 3, 3)
    n_iter : int
        RANSAC 반복 횟수
    n_samples : int
        RANSAC 반복마다 추출할 샘플의 수

    Returns
    -------
    np.ndarray
        계산된 내부 파라미터 행렬 K. (3, 3)
    """

    assert points_2d.shape[0] == H_mats.shape[0]
    assert points_2d.shape[2] == points_3d.shape[1]
    T = points_2d.shape[0]
    N = points_2d.shape[2]

    if n_samples < 1:
        n_samples = T // 10

    def proj_cam(m, M, K, R_mats, T_vecs):
        """3차원 점들을 카메라로 사영해 실제 관측된 점과의 오차를 계산한다.

        Parameters
        ----------
        m : np.ndarray
            2차원에서 관측된 점
        M : np.ndarray
            실제 3차원 점 좌표
        K : np.ndarray
            카메라 내부 파라미터 행렬
        R_mats : np.ndarray
            카메라 외부 파라미터. Rotation Matrix
        T_vecs : np.ndarray
            카메라 외부 파라미터. Translation Vector

        Returns
        -------
        float
            계산된 Reprojection Error(RMSE)
        """

        X = np.einsum("nij, jk -> nik", R_mats, M) + T_vecs
        X /= X[:, -1:]
        X = np.einsum("ij, njk -> nik", K, X)
        error = np.sqrt(((X[:, :-1] - m) ** 2).sum(axis=1)).mean(axis=1)

        return error

    max_inlier_count = 0
    best_K = None
    minimum_error = None
    for i in tqdm(range(n_iter), leave=False):
        args = np.random.choice(np.arange(T), n_samples)
        sample_H_mats = H_mats[args]

        K = estimate_intr_params(sample_H_mats)
        RT_mats = []
        valid_args = []
        for i, H in enumerate(H_mats):
            try:
                RT = np.concatenate(calculate_relative_extr(K, H), axis=1)
                RT_mats.append(RT)
                valid_args.append(i)
            except NotPerpendicularError as e:
                continue

        RT_mats = np.stack(RT_mats)
        error = proj_cam(
            points_2d[valid_args],
            points_3d,
            K,
            RT_mats[..., :3],
            RT_mats[..., 3:],
        )
        inlier_count = (error < inlier_error).sum()

        if inlier_count > max_inlier_count:
            max_inlier_count = inlier_count
            best_K = K
            minimum_error = error

    assert best_K is not None

    return best_K, minimum_error


def find_initial_extr_params(points_2d, K1, K2, H_mats):
    T = points_2d.shape[1]
    N = points_2d.shape[3]

    progress = tqdm(range(T))

    minimum_error = None
    best_RT = None
    valid_args = list(range(T))

    for i in progress:
        H1 = H_mats[0, i]
        H2 = H_mats[1, i]
        try:
            R1, T1 = calculate_relative_extr(K1, H1)
            R2, T2 = calculate_relative_extr(K2, H2)
        except NotPerpendicularError:
            valid_args.remove(i)
            continue

        P1 = K1 @ np.concatenate([R1, T1], axis=1)
        P2 = K2 @ np.concatenate([R2, T2], axis=1)
        project_matrices = np.stack([P1, P2])

        U = len(valid_args)
        X = points_2d[:, valid_args]  # (2, U, 2, N)
        X = X.transpose(0, 1, 3, 2)  # (2, U, N, 2)
        X = X.reshape(2, -1, 2)  # (2, UN, 2)
        error, m, M = calculate_reprojection_error(
            X, project_matrices, with_points=True
        )

        if minimum_error is None or minimum_error > error:
            R_rel = R2 @ np.linalg.inv(R1)
            T_rel = -R2 @ np.linalg.inv(R1) @ T1 + T2

            minimum_error = error
            best_RT = np.concatenate([R_rel, T_rel], axis=1)

        progress.set_description("Finding Initial Extrinsic Parameters...")

    assert best_RT is not None

    return best_RT, minimum_error


def calibrate_extr(points_2d, K1, K2, init_RT):
    from scipy.optimize import least_squares

    T = points_2d.shape[1]
    N = points_2d.shape[3]

    T = init_RT[:, 3:]
    w = log_map(init_RT[:, :3])

    x0 = np.concatenate([w, T]).reshape(-1)

    def fun(x):
        R1, T1 = np.eye(3), np.zeros((3, 1))
        R2, T2 = exp_map(x[:3, np.newaxis]), x[3:, np.newaxis]
        P1 = K1 @ np.concatenate([R1, T1], axis=1)
        P2 = K2 @ np.concatenate([R2, T2], axis=1)
        project_matrices = np.stack([P1, P2])

        U = points_2d.shape[1]
        X = points_2d  # (2, U, 2, N)
        X = X.transpose(0, 1, 3, 2)  # (2, U, N, 2)
        X = X.reshape(2, -1, 2)  # (2, UN, 2)
        error = calculate_reprojection_error(X, project_matrices, to_scalar=None)
        error = error.reshape(2, U, N).mean(axis=(0, 2))

        return error

    res = least_squares(fun, x0, method="lm", verbose=0, xtol=1e-8, ftol=1e-8)

    R = exp_map(res.x[:3, np.newaxis])
    T = res.x[3:, np.newaxis]
    RT = np.concatenate([R, T], axis=1)
    error = fun(res.x)

    return RT, error


def main(input_path1, input_path2, intr_path1, intr_path2, output_path):
    data = np.load(args.input_path, allow_pickle=True).tolist()
    points_2d = data["markers"].transpose(0, 2, 1)

    # 피사체에 알맞게 POSITIONS 수정 필요.
    points_3d = np.concatenate([POSITIONS, np.ones((len(POSITIONS), 1))], axis=1).T

    # 영상의 매 프레임 사진마다 호모그라피 행렬을 따로 계산한다.
    progress = tqdm(range(points_2d.shape[0]))
    homography_matrices = [None] * points_2d.shape[0]

    for i in progress:
        homography_matrices[i] = estimate_homography(points_2d[i], points_3d)
        progress.set_description("Caculating Homography Matrices...")

    homography_matrices = np.stack(homography_matrices)

    K, error = calibrate_intr(points_2d, points_3d, homography_matrices)

    text = f"""캘리브레이션 오차
    mean: {error.mean()}
    std: {error.std()}
    min: {error.min()}
    max: {error.max()}
    """
    print(text)
    print(K)

    os.makedirs(os.path.split(output_path)[0], exist_ok=True)
    np.save(output_path, K)

    data = np.load(input_path1, allow_pickle=True).tolist()
    K1 = np.load(intr_path1)
    frame_numbers1 = list(data["frame_numbers"])
    m1 = data["markers"].transpose(0, 2, 1)

    data = np.load(input_path2, allow_pickle=True).tolist()
    K2 = np.load(intr_path2)
    frame_numbers2 = list(data["frame_numbers"])
    m2 = data["markers"].transpose(0, 2, 1)

    frame_numbers = list(set(frame_numbers1) & set(frame_numbers2))

    indices1 = [frame_numbers1.index(n) for n in frame_numbers]
    indices2 = [frame_numbers2.index(n) for n in frame_numbers]

    m1 = m1[indices1]
    m2 = m2[indices2]

    points_2d = np.stack([m1, m2])
    points_3d = np.concatenate([POSITIONS, np.ones((len(POSITIONS), 1))], axis=1).T

    T = points_2d.shape[1]
    N = points_2d.shape[3]

    H_mats = [[], []]
    progress = tqdm(range(T))

    for i in progress:
        H_mats[0].append(estimate_homography(points_2d[0, i], points_3d))
        H_mats[1].append(estimate_homography(points_2d[1, i], points_3d))
        progress.set_description("Caculating Homography Matrices...")

    H_mats = np.array(H_mats)

    init_RT, error = find_initial_extr_params(points_2d, K1, K2, H_mats)
    print("Optimizing Parameters...", end="")
    RT, error = calibrate_extr(points_2d, K1, K2, init_RT)
    print("Done!")

    text = f"""캘리브레이션 오차: {error.mean()}"""
    print(text)

    os.makedirs(os.path.split(output_path)[0], exist_ok=True)
    np.save(output_path, RT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path1", type=str)
    parser.add_argument("input_path2", type=str)
    parser.add_argument("intr_path1", type=str)
    parser.add_argument("intr_path2", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    main(args.input_path1, args.input_path2, args.intr_path1, args.intr_path2, args.output_path)
