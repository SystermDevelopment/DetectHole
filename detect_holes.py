import cv2
import numpy as np
import random
import math
import os
import glob

GAUS_BLUR = 15        # ガウシアンブラーのぼかし範囲
CANNY_MIN = 30        # エッジ検出の最小閾値
CANNY_MAX = 80        # エッジ検出の最大閾値
INNER_RINGSIZE = -35  # リングの内枠（緑線）
OUTER_RINGSIZE = 5    # リングの外枠（緑線）
MIN_RINGSIZE = 436.0  # 抽出するリングの最小サイズ
MAX_RINGSIZE = 437.0  # 抽出するリングの最大サイズ
MIN_HOLEDIST = 450    # ペアの穴の最小距離
MAX_HOLEDIST = 490    # ペアの穴の最大距離
MIN_HOLESIZE = 3      # 抽出する穴の最小サイズ
MAX_HOLESIZE = 7      # 抽出する穴の最大サイズ


def fit_circle_3points(p1, p2, p3):
    """
    3点 (p1, p2, p3) から円の中心 (cx, cy) と半径 r を求める。
    3点が同一直線状にある場合は None を返す。
    p1, p2, p3 = (x, y) のタプル
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # 行列を使った計算手順
    # 参考: https://mathworld.wolfram.com/Circle.html
    #       または 幾何学的アプローチで計算
    denom = 2 * (
        x1*(y2 - y3) +
        x2*(y3 - y1) +
        x3*(y1 - y2)
    )
    if abs(denom) < 1e-7:
        return None  # ほぼ同一直線

    # 中心 (cx, cy)
    cx = (
        (x1**2 + y1**2)*(y2 - y3) +
        (x2**2 + y2**2)*(y3 - y1) +
        (x3**2 + y3**2)*(y1 - y2)
    ) / denom

    cy = (
        (x1**2 + y1**2)*(x3 - x2) +
        (x2**2 + y2**2)*(x1 - x3) +
        (x3**2 + y3**2)*(x2 - x1)
    ) / denom

    # 半径 r
    r = math.sqrt((x1 - cx)**2 + (y1 - cy)**2)
    return (cx, cy, r)

def ransac_circle_detection(
    edge_points, 
    max_iterations=100000, 
    distance_threshold=2.0, 
    min_inliers=1000,
    min_radius=MIN_RINGSIZE,
    max_radius=MAX_RINGSIZE
):
    best_cx, best_cy, best_r = 0, 0, 0
    best_inliers = 0

    n_points = len(edge_points)
    if n_points < 3:
        return None

    edge_points_arr = np.array(edge_points, dtype=np.float32)

    for _ in range(max_iterations):
        # 1) エッジ点の中からランダムに3点選ぶ
        idx = random.sample(range(n_points), 3)
        p1 = tuple(edge_points_arr[idx[0]])
        p2 = tuple(edge_points_arr[idx[1]])
        p3 = tuple(edge_points_arr[idx[2]])

        # 2) 3点から円を求める
        circle_params = fit_circle_3points(p1, p2, p3)
        if circle_params is None:
            continue
        cx, cy, r = circle_params

        if r < min_radius or r > max_radius:
            continue

        # 3) すべてのエッジ点について、円周との距離をチェック
        #    distance = |sqrt((x - cx)^2 + (y - cy)^2) - r|
        dist_arr = np.sqrt((edge_points_arr[:, 0] - cx)**2 + 
                           (edge_points_arr[:, 1] - cy)**2)
        dist_from_circle = np.abs(dist_arr - r)

        inliers = np.count_nonzero(dist_from_circle < distance_threshold)

        # 4) インライアが多ければ更新
        if inliers > best_inliers:
            best_inliers = inliers
            best_cx, best_cy, best_r = cx, cy, r

        # min_inliers を超えたら早期終了
        if best_inliers > min_inliers:
            break

    if best_inliers < min_inliers:
        return None

    return (best_cx, best_cy, best_r)

def circle_ransac(img):
    # 1) 前処理: グレースケール + ブラー + Canny
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (GAUS_BLUR, GAUS_BLUR), 0)
    edges = cv2.Canny(blur, CANNY_MIN, CANNY_MAX)
 
    # エッジを出力
    # #edge_path = f"image/test/edge_{base_path}.jpg"
    # #cv2.imwrite(edge_path, edges)

    # 2) エッジ点をリスト化
    edge_points = np.column_stack(np.where(edges > 0))  # (row, col)
    # OpenCVは画像を [y, x] = [row, col] で扱うので注意
    # fit_circle_3points の計算は (x, y) が前提なので入れ替える
    edge_points = [(float(xc[1]), float(xc[0])) for xc in edge_points]

    # 4) RANSACで円を推定
    circle_result = ransac_circle_detection(edge_points)

    return circle_result, edges, edge_points

def detect_holes(img, edge_points, best_cx, best_cy, best_r, edges, base_path):
    # 5) 緑線以下、青線以上のエッジ点を抽出
    edge_points_arr = np.array(edge_points, dtype=np.float32)
    distances = np.sqrt((edge_points_arr[:, 0] - best_cx)**2 +
                        (edge_points_arr[:, 1] - best_cy)**2)

    mask = (distances >= best_r + INNER_RINGSIZE) & (distances <= best_r + OUTER_RINGSIZE)
    filtered_points = edge_points_arr[mask]

    # 6) 領域をクラスタリングして円を計算
    # 二値画像を作成
    mask_img = np.zeros_like(edges)
    for point in filtered_points:
       mask_img[int(point[1]), int(point[0])] = 255

    filename = os.path.join('img_detect', f"edge_{base_path}.jpg")
    cv2.imwrite(filename, mask_img)

    # 輪郭を抽出
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6) 小さな円を検出
    small_circles = []
    for contour in contours:
        (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
        if MIN_HOLESIZE <= radius <= MAX_HOLESIZE:
            small_circles.append((circle_x, circle_y, radius))
            cv2.circle(img, (int(circle_x), int(circle_y)), int(radius), (255, 0, 0), 2)

    valid_circle_pairs = []
    for i in range(len(small_circles)):
        for j in range(i + 1, len(small_circles)):
            circle1 = small_circles[i]
            circle2 = small_circles[j] 
            distance = math.sqrt((circle1[0] - circle2[0])**2 + (circle1[1] - circle2[1])**2)
            if MIN_HOLEDIST <= distance <= MAX_HOLEDIST:
                valid_circle_pairs.append((circle1, circle2))

    ret = "ng"
    # 条件を満たしたペアを赤色で描画
    for circle1, circle2 in valid_circle_pairs:
        cv2.circle(img, (int(circle1[0]), int(circle1[1])), int(circle1[2]), (0, 0, 255), 2)
        cv2.circle(img, (int(circle2[0]), int(circle2[1])), int(circle2[2]), (0, 0, 255), 2)
        ret = "ok"

    # 結果を保存
    filename = os.path.join('img_detect', f"hole_{base_path}.jpg")
    cv2.imwrite(filename, img)

    return ret