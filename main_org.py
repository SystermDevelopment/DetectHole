# coding: utf-8
import glob
import os
import time
import sys
import math
from datetime import datetime as dt
import serial
import cv2 as cv
import numpy as np
from serial.tools import list_ports
import PySimpleGUI as sg
from datetime import datetime

from networks.detect_network import DetectNetwork as detect
from networks.getpos_network import GetposNetwork as getpos
from networks.rotate_network import RotateNetwork as rotate
from networks.gethole_network import GetHoleNetwork as gethole
from networks.anomaly_det_network import AnomalyDetNetwork as anomaly
from networks.checkhole_network import CheckHoleNetwork as checkhole
from networks.holeexist_network import HoleExistNetwork as holeexist
from networks.angle_network import AngleNetwork as angle
import gpgpu
import visca

np.set_printoptions(suppress=True)

#CUDA使用有無
USE_GPU = False

# 対象物の中心X座標
CX_MIN_LIMIT = 720
CX_MAX_LIMIT = 1430

# 対象物位置が範囲外に入った場合の
# フレームスキップ数
TARGET_AREAOUT_FRAMESKIP = 20   # 要調整

# 穴有無チェックの閾値
# 数値以上であれば穴有判定
HOLE1_EXIST_THRESH = 0.7
HOLE2_EXIST_THRESH = 0.7

# 1個体の穴有無判定のOK/NG判定用
# 数値以上であれば穴有判定
HOLE1_EXIST_OK_RATIO = 0.55
HOLE2_EXIST_OK_RATIO = 0.50

# 間引きフラグ（開発用）
do_frame_skip = True

def get_current_timestamp():
    """現在の日時をyyyyMMdd_hhmmss形式で返す"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def imgconv(imgHD, withGray=False):
    """
    opencv画像(BGRフォーマット)をNNABLA用フォーマットへ変換
    """
    # resize to 480x270
    img = cv.resize(imgHD, None, fx=0.25, fy=0.25)
    if withGray:
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)

    if withGray:
        return img, img_gray
    else:
        return img

def imgconv_gray(imgHD):
    """
    opencv画像(BGRフォーマット)をNNABLA用フォーマットへ変換
    """
    # resize to 480x270
    img = cv.resize(imgHD, None, fx=0.25, fy=0.25)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    return img

def get_angle(x, y, img):
    """
     穴有無の検出
    """
    x.d = img.reshape(x.shape)
    s = time.time()
    y.forward()
    elapse = time.time() -s

    angle =int(np.argmax(y.d))  # 0-48の値
    # 0 : -48度, 右下向き
    # 24: 0度　右向き
    # 48: 48度, 右上向き
    angle = (angle * 2) - 48

    print("get_angle %d ms, %d" % (int(elapse*1000), angle))

    return angle

def get_holeexist(x, y, img):
    """
     穴有無の検出
    """
    #img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    x.d = img.reshape(x.shape)
    #s = time.time()
    y.forward()
    #elapse = time.time() -s
    #print("get_holeexist %f" % (elapse))

    hole0 = round(float(y.d[0, 0]), 3)
    hole1 = round(float(y.d[0, 1]), 3)

    return hole0, hole1

def rotate_img(img, angle):
    """
    angle角度で回転させた画像を返す
    """
    h = img.shape[0]
    w = img.shape[1]
    scale = 1.0
    center = (int(w/2), int(h/2))
    trans = cv.getRotationMatrix2D(center, angle, scale)
    return cv.warpAffine(img, trans, (w, h))

def get_capture_device():
    for idx in range(0,10):
        capture = cv.VideoCapture(idx, cv.CAP_DSHOW)
        #capture = cv.VideoCapture(idx, cv.CAP_MSMF)
        if capture.isOpened():
            break
    if not capture.isOpened():
        print('capture device open failed')
        sg.popup("HDMIキャプチャデバイスorカメラが見つかりません\n接続を確認してください", title="エラー")
        exit(0)

    print("capture deviceinfo(before)")
    print(f'\tCAP_PROP_FOURCC:{capture.get(cv.CAP_PROP_FOURCC)}')
    print(f'\tCAP_PROP_FPS:{capture.get(cv.CAP_PROP_FPS)}')
    print(f'\tCAP_PROP_WIDTH:{capture.get(cv.CAP_PROP_FRAME_WIDTH)}')
    print(f'\tCAP_PROP_HEIGHT:{capture.get(cv.CAP_PROP_FRAME_HEIGHT)}')

    capture.set(cv.CAP_PROP_FPS, 60)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    time.sleep(7)

    print("capture deviceinfo(after)")
    print(f'\tCAP_PROP_FOURCC:{capture.get(cv.CAP_PROP_FOURCC)}')
    print(f'\tCAP_PROP_FPS:{capture.get(cv.CAP_PROP_FPS)}')
    print(f'\tCAP_PROP_WIDTH:{capture.get(cv.CAP_PROP_FRAME_WIDTH)}')
    print(f'\tCAP_PROP_HEIGHT:{capture.get(cv.CAP_PROP_FRAME_HEIGHT)}')

    # YUY2 60fpsでキャプチャ
    return capture

def get_log_path():
    # 年-月別にフォルダ分け
    path = './log/' + dt.now().strftime('%Y-%m')
    os.makedirs(path, exist_ok=True)

    path += '/' + dt.now().strftime('%Y-%m-%d.txt')
    return path

def write_log(logstr, logfile):
    logfile.write('[' + dt.now().strftime('%H:%M:%S.%f') + '] ' + logstr + '\n')

def get_mask_existhole():
    mask_size = 960, 490, 1 # hight:960, width:490
    img_mask = np.zeros(mask_size, dtype=np.uint8)
    cv.circle(img_mask, (15, mask_size[1]), 472, color=(255), thickness=30)
    cv.circle(img_mask, (10, mask_size[1]), 455, color=(255), thickness=30)
    cv.circle(img_mask, (10, mask_size[1]), 430, color=(255), thickness=30)
    cv.line(img_mask, (mask_size[1], 0), (mask_size[1],mask_size[0]), 0, 10)
    return img_mask

def adjust_image_zoom(img, scale=0.92):

    resized = cv.resize(img, None, None, scale, scale, cv.INTER_CUBIC)
    h, w, _ = resized.shape

    return cv.copyMakeBorder(resized, 0, 1080-h, 0, 1920-w, cv.BORDER_CONSTANT, (0,0,0))

def capture_from_cam(video_path, start_frame=0):
    """
    UVCからのキャプチャ(メイン)
    """

    real_camera = True

    if video_path:
        capture = cv.VideoCapture(video_path)
        capture.set(cv.CAP_PROP_POS_FRAMES, start_frame)
        real_camera = False
    else:
        capture = get_capture_device()
        # カメラ初期化
        camera_init()
    
    layout = [
        [sg.Submit(button_text='終了', size=(8, 1), font=('', 20))]
    ]

    # 終了ボタンウインドウの表示と位置指定
    window = sg.Window('検査画面', layout, no_titlebar=True, keep_on_top=True, finalize=True)
    disp_w, disp_h = window.get_screen_size()
    win_w, win_h = window.Size
    window.Move(disp_w-win_w, disp_h-win_h)

    # カメラ画ウインドウの表示と位置指定
    ret, imgHD = capture.read()

    while imgHD is None:
        imgHD = capture.read()

    cv.imshow("camera", imgHD)
    cv.moveWindow("camera", 0, 0)

    # NG画像書き込み用の日付別フォルダ作成
    datedir = dt.now().strftime('%Y-%m-%d')
    os.makedirs("./img_ng/" + datedir, exist_ok=True)

    is_discance_confirm = False
    is_ng = False
    wait_remove = False
    dist_list = []

    mask_size = 960, 490, 1 # hight:960, width:490
    img_mask = get_mask_existhole()
    hole1_exist = False
    hole2_exist = False
    #last_cx = -1
    check_count = 0
    hole1_ng_count = 0
    hole2_ng_count = 0
    angle_old = 0

    with open(get_log_path(), 'a') as log:
        while True:
            ret, imgHD = capture.read()
            # 動的なファイル名で画像を保存
            timestamp = get_current_timestamp()
            filename = f"{timestamp}.jpg"
            cv.imwrite(filename, imgHD)
            if not ret:
                print('capture read is error')
                write_log('capture read is error', log)
                break

            if do_frame_skip:
                frame = capture.get(cv.CAP_PROP_POS_FRAMES)
                capture.set(cv.CAP_PROP_POS_FRAMES, frame + 5)
            
            if video_path and "WIN_20230726_15_05" in video_path:
                imgHD = adjust_image_zoom(imgHD)
            elif video_path and "WIN_20230726_" in video_path:
                imgHD = adjust_image_zoom(imgHD, 0.945)
               
            # 終了ボタン押下チェック
            # 毎フレームチェックによる処理時間への影響を要確認
            event, values = window.read(timeout=1)
            if event == '終了':
                capture.release()
                cv.destroyAllWindows()
                window.close()
                break


            # 穴有無推論用画像に変換
            img_gray = cv.cvtColor(imgHD, cv.COLOR_BGR2GRAY)
            img_gauss = cv.GaussianBlur(img_gray, (5, 3), 3)
            min_val = 100
            max_val = 270

            #print("med:", med_val, "min:", min_val, " max:", max_val)
            img_canny = cv.Canny(img_gauss, min_val, max_val)
            circles = cv.HoughCircles(img_canny, cv.HOUGH_GRADIENT,
                dp=0.9, minDist=500, param1=50, param2=15, minRadius=460, maxRadius=480)
                #dp=0.5, minDist=200, param1=100, param2=55, minRadius=400, maxRadius=490)
            
            if is_ng:
                # NG判定後は対象物の取り外し(円検出なし)待ち状態へ
                wait_remove = True
                is_ng = False

                cv.setWindowTitle("camera", "Please remove the target")
                write_log("[State] wait for Remove", log)
                print("[State] wait for Remove")
                log.flush()
                cv.waitKey(1)
                continue
            elif wait_remove:
                if circles is None:
                    # 対象物取り外しを検出
                    wait_remove = False
                    cv.setWindowTitle("camera", "in Testing")
                    print("[State] Restart")
                else:
                    # 対象物取り外し待ち
                    #time.sleep(0.5)
                    pass
                
                cv.imshow("camera", imgHD)
                cv.waitKey(1)
                continue
            else:
                cv.setWindowTitle("camera", "in Testing")

            if circles is not None:
                #print("Frame:", cap.get(cv.CAP_PROP_POS_FRAMES))

                for cx, cy, r in circles.squeeze(axis=0).astype(int):

                    # 最終cxよりも大きくなった場合の判定が必要かも
                    #print("CX:", cx, " CY:", cy, " R:", r)

                    if cx < CX_MIN_LIMIT or cx > CX_MAX_LIMIT:
                        #print("CX:", cx, " CY:", cy, " R:", r)
                        # 検出範囲外
                        if check_count > 0:

                            # NG判定の割合で穴有無を判定
                            hole1_exist = hole2_exist = False
                            if (hole1_ng_count / check_count) < HOLE1_EXIST_OK_RATIO:
                                hole1_exist = True
                            if (hole2_ng_count / check_count) < HOLE2_EXIST_OK_RATIO:
                                hole2_exist = True

                            # 穴存在結果を確定
                            if not (hole1_exist and hole2_exist):
                                is_ng = True
                                write_log("[detect] ------ detNG ----------", log)
                                write_log("[detect] check_count:%d" % (check_count), log)
                                write_log("[detect] hole1:%s" % (hole1_exist), log)
                                write_log("[detect] hole2:%s" % (hole2_exist), log)

                                print("detNG")
                                print("[detect] hole1:%s" % (hole1_exist))
                                print("[detect] hole2:%s" % (hole2_exist))

                            print("hole1:%d/%d => %.3f" % (hole1_ng_count, check_count, hole1_ng_count / check_count))
                            print("hole2:%d/%d => %.3f" % (hole2_ng_count, check_count, hole2_ng_count / check_count))
                            print("--------------------------")
                            write_log("hole1:%d/%d => %.3f" % (hole1_ng_count, check_count, hole1_ng_count / check_count), log)
                            write_log("hole2:%d/%d => %.3f" % (hole2_ng_count, check_count, hole2_ng_count / check_count), log)
                            write_log("--------------------------", log)
                            log.flush()

                            # 初期化
                            hole1_exist = False
                            hole2_exist = False
                            check_count = 0
                            hole1_ng_count = 0
                            hole2_ng_count = 0
                            angle_old = 0

                            # フレームskip
                            for num in range(TARGET_AREAOUT_FRAMESKIP):
                                capture.read()

                        cv.imshow("camera", imgHD)
                        cv.waitKey(1)
                        continue

                    # 円周を描画する
                    cv.circle(imgHD, (cx, cy), r, (0, 165, 255), 5) 
                    # 中心点を描画する
                    cv.circle(imgHD, (cx, cy), 2, (0, 0, 255), 3)

                    # 穴検出用画像
                    #img_cannyから穴検出用の画像生成(y,x)

                    try:
                        frame = capture.get(cv.CAP_PROP_POS_FRAMES)
                        fname = "./canny/WIN_20230413_10_34_19_Pro_%06d.png" % (frame)

                        # リング全体画像
                        img_ring = img_canny[cy-int(mask_size[0]/2):int(cy+mask_size[0]/2), cx - mask_size[1]: cx+mask_size[1]]
                        cv.imwrite(fname, img_ring)

                        # リング右半分画像および不要部分のマスク処理
                        img_hole = img_canny[cy-int(mask_size[0]/2):int(cy+mask_size[0]/2), cx: cx+mask_size[1]]
                        img_hole_masked = cv.bitwise_and(img_hole, img_mask)

                        # 推論用画像(角度検出)
                        img_hole_dilated_mini = cv.morphologyEx(img_hole_masked, cv.MORPH_CROSS, np.ones((9,9),np.uint8))
                        img_hole_dilated_mini = cv.resize(img_hole_dilated_mini, None, None, 0.1, 0.1, cv.INTER_CUBIC)
                        #cv.imshow("dilated-mini", img_hole_dilated_mini)
                        #print("dilated-mini ", img_hole_dilated_mini.shape[1], ", ", img_hole_dilated_mini.shape[0])
                        det_angle = get_angle(angle_x, angle_y, img_hole_dilated_mini)
                        if det_angle == 0:
                            det_angle = angle_old
                            cv.imwrite(fname, img_ring)
                        else:
                            angle_old = det_angle

                        # 推論用画像(角度検出) 結果出力
                        if False:
                            dir = "./angle/%03d/" % (det_angle + 48)
                            os.makedirs(dir, exist_ok=True)
                            # トレーニング用画像
                            fname = dir + "20230413_10_34_19_Pro_%06d_mini.png" % (frame)
                            cv.imwrite(fname, img_hole_dilated_mini)
                            # リング画像
                            fname = dir + "20230413_10_34_19_Pro_%06d.png" % (frame)
                            img_hole = img_canny[cy-int(mask_size[0]/2):int(cy+mask_size[0]/2), cx - mask_size[1]: cx+mask_size[1]]
                            cv.imwrite(fname, img_hole)

                        # 検出した角度分、リング画像を回転
                        ring_w = int(img_ring.shape[1])
                        ring_h = int(img_ring.shape[0])
                        trans = cv.getRotationMatrix2D((int(ring_w/2),int(ring_h/2)), det_angle * -1 , 1.0)
                        img_ring_rotated = cv.warpAffine(img_ring, trans, (ring_w, ring_h))
                        cv.imshow("ring rotated", img_ring_rotated)

                        # リング右半分画像および不要部分のマスク処理(回転補正済み)
                        img_hole = img_ring_rotated[0:ring_h, int(ring_w/2):ring_w]
                        img_hole_masked = cv.bitwise_and(img_hole, img_mask)

                        # 推論用画像(穴検出)
                        img_hole_dilated = cv.morphologyEx(img_hole_masked, cv.MORPH_DILATE, np.ones((3,3),np.uint8))
                        img_hole_dilated = cv.resize(img_hole_dilated, None, None, 0.5, 0.5, cv.INTER_CUBIC)

                        #cv.imshow("ring rotated", img_hole_dilated)

                        # 推論実行
                        s = time.time()
                        exist_hole1_prob, exist_hole2_prob = get_holeexist(holeexist_x, holeexist_y, img_hole_dilated)
                        elapse = time.time() - s

                        b_hole1 = False
                        if HOLE1_EXIST_THRESH < exist_hole1_prob:
                            hole1_exist = True
                            b_hole1 = True
                        else:
                            hole1_ng_count += 1

                        b_hole2 = False
                        if HOLE2_EXIST_THRESH < exist_hole2_prob:
                            hole2_exist = True
                            b_hole2 = True
                        else:
                            hole2_ng_count += 1

                        if not (b_hole1 and b_hole2):
                            # NG画像を保存
                            datestr = dt.now().strftime('%Y%m%d_%H%M%S_%f')
                            #cv.imwrite("./img_ng/" + datedir + "/" + datestr + ".png", imgHD)
                            cv.imwrite("./img_ng/" + datedir + "/" + datestr + ".png", img_hole_dilated)
                            pass

                        # 穴検出推論画像の出力
                        if False:
                            dir_no = 0
                            if b_hole1 and b_hole2:
                                dir_no = 3
                            elif b_hole1:
                                dir_no = 1
                            elif b_hole2:
                                dir_no = 2
                                
                            dir = "F://nnc_datasets/ring_hole_type3_4/raw/%d/" % (dir_no)
                            os.makedirs(dir, exist_ok=True)
                            fname = "WIN_20230413_10_34_19_Pro_%06d.png" % (frame)
                            img_w = img_hole_dilated.shape[1]
                            img_h = img_hole_dilated.shape[0]
                            img_hole_dilated_half = img_hole_dilated[0:img_h, int(img_w/2):img_w]
                            cv.imwrite(dir + fname, img_hole_dilated_half)

                        hole1_str = "ok" if b_hole1 else "ng"
                        hole2_str = "ok" if b_hole2 else "ng"
                        # Frame数, ok/ngおよび推論値を出力
                        output = "%08d %s(%.3f), %s(%.3f), %d, %dms" % (frame, hole1_str, exist_hole1_prob, hole2_str, exist_hole2_prob, cx, elapse*1000)
                        print(output)
                        write_log(output, log)
                        cv.imshow("dilated", img_hole_dilated)

                        check_count += 1
                    except:
                        check_count = check_count
                    
                    cv.imshow("camera", imgHD)

                    if cv.waitKey(1) > 0:
                        break
            else:
                cv.imshow("camera", imgHD)
                if cv.waitKey(1) > 0:
                        break
                        
            datestr = dt.now().strftime('%Y%m%d_%H%M%S')

            if is_ng:
                #NG
                print("NG")
                # NG信号送信
                if com_port:
                    print("send NG")
                    com_port.write(b"NGGG")
                # NG判定画像を保存
                if True:
                    print("NG det")
                    #cv.imwrite("./img_ng/" + datedir + "/" + datestr + ".png", imgHD)
                    #cv.imwrite("./img_ng/" + datedir + "/" + datestr + "_fill.png", imgHD_fill)
                    #cv.imwrite("./img_ng/" + datedir + "/" + datestr + "_sq.png", imgHD_sq)
                    #cv.imwrite("./img_ng/" + datedir + "/" + datestr + "_mark.png", imgHD2)
                    #if imgHD_holearea_valid:
                    #    cv.imwrite("./img_ng/" + datestr + "_hole.png", imgHD_holearea)
            else:
                #print("OK")
                if False:
                    cv.imwrite("./img_ok/" + datestr + "_fill.png", imgHD_fill)
                    cv.imwrite("./img_ok/" + datestr + "_sq.png", imgHD_sq)
                    cv.imwrite("./img_ok/" + datestr + "_mark.png", imgHD_sq2)
                    if imgHD_holearea:
                        cv.imwrite("./img_ok/" + datestr + "_hole.png", imgHD_holearea)

repeat_ison = False
repeat_start = 0
repeat_count = 0

def get_path_label(path):
    idx = path.rfind('/')
    idx2 = path.rfind('\\')
    if idx is -1 and idx2 is -1:
        return 'hoge'
    if idx2 > idx:
        idx = idx2
    return path[idx+1:].replace('.MP4', '').replace('.png', '')

def get_com_port():
    """
    COMポート(1番目)を取得
    """
    ports = list_ports.comports()
    no_visca_ports = [v for v in ports if not "Harrier" in v.description]

    try:
        ser = serial.Serial()
        ser.port = no_visca_ports[0].device
        ser.baudrate = 19200

        ser.open()

        return ser
    except:
        print("[Error] can't open a serial port")
        return None

def get_target():
    """
    検査対象リストを選択するためのウインドウを表示
    開始ボタン押下時に選択されていた文字列を返す
    """
    sg.theme('Dark Blue 3')

    itm = ['83-915202-09', '設置位置調整']

    layout = [
        [sg.Text('検査対象リスト')],
        [sg.Listbox(itm, font=('', 50), size=(24, 4), key='lboxTarget')],
        [sg.Submit(button_text='開始', size=(15, 2), font=('', 40)), sg.Submit(button_text='終了', size=(15, 2), font=('', 40))]
    ]

    window = sg.Window('検査対象を選択してください', layout)

    while True:
        event, values = window.read()
        if event is None or event is '終了':
            print('exit')
            sys.exit(0)
        elif event == '開始':
            print('start')
            if values['lboxTarget']:
                break

    ret = values['lboxTarget'][0]

    window.close()

    # 検査対象リストの選択内容を保存
    return ret

def get_standard_marker_img(w, h):

    fg = np.zeros((h, w, 3), np.uint8)

    line_tickness = 5
    circle_tickness = 6
    color = (0, 0, 255, 128)

    line_big_tl = (0.098, 0.1424)
    line_big_tr = (0.6745, 0.1424)
    line_big_bl = (0.1052, 0.8305)
    line_big_br = (0.6745, 0.8203)
    line_sml_tl = (0.1307, 0.201)
    line_sml_tr = (0.6422, 0.201)
    line_sml_bl = (0.1370, 0.7735)
    line_sml_br = (0.6422, 0.7685)

    lines = [
        # big
        line_big_tl + line_big_tr,
        line_big_tl + line_big_bl,
        line_big_tr + line_big_br,
        line_big_bl + line_big_br,
        #small
        line_sml_tl + line_sml_tr,
        line_sml_tl + line_sml_bl,
        line_sml_tr + line_sml_br,
        line_sml_bl + line_sml_br,
    ]

    circles = [
        (0.168, 0.269),      # top-left
        (0.594, 0.260),    # top-right
        (0.168, 0.670),    # btm-left
        (0.596, 0.624),    # btm-right
        (0.391, 0.456)    # center
    ]

    for c in circles:
        cv.circle(fg, (int(w * c[0]), int(h * c[1])), 30, color, circle_tickness)

    #for l in lines:
    #    cv.line(fg, (int(w * l[0]), int(h * l[1])), (int(w * l[2]), int(h * l[3])), color, line_tickness)

    return fg

def camera_init():
    visca.wait_power_on()
    visca.call_preset(0)
    visca.picture_flip(True)

def start_adjust_setting():
    """
    UVCからのキャプチャ
    """
    capture = get_capture_device()
    #capture = cv.VideoCapture('F://nnc_datasets/movie/0602/M1060002.MP4')
    
    # カメラ初期化
    camera_init()

    layout = [
        [sg.Listbox([], size=(30, 40), key='-LIST-'),
         sg.Image(filename='', key='image')],
        [sg.Submit(button_text='開始/停止', size=(15, 2), font=('', 40)), sg.Submit(button_text='終了', size=(15, 2), font=('', 40))]
    ]

    window = sg.Window('設置位置調整画面', layout, location=(0, 0))

    isStarted = True
    loglist = []
    count = 0

    #marker_fg = get_standard_marker_img(1920, 1080)
    marker_fg = cv.imread("ref.jpg")

    while True:
        event, values = window.read(timeout=30)
        if event in (None, 'Exit'):
            try:
                capture.release()
                cv.destroyAllWindows()
            except:
                pass
            break

        elif event == '開始/停止':
            isStarted = not isStarted

        elif event == '終了':
            capture.release()
            window.close()
            break

        if isStarted:
            ret, imgHD = capture.read()
            if ret is True:
                # 認識対象外の領域を塗りつぶし
                #imgHD_fill = fill_outside(imgHD)

                # 推論用に画像を変換
                #img_fill_gray = imgconv_gray(imgHD_fill)

                # 検査対象の存在チェック
                #exist, prob = is_exsist(detect_x, detect_y, img_fill_gray)
                exist = False

                strlog = '[' + format(count, '08') + '] '
                count += 1

                if not exist:
                    strlog += 'none'
                else:
                    strlog += 'exist prob=' + str(prob)

                # 存在チェック結果のログを追加
                loglist.append(strlog)
                #window['-LIST-'].Update(loglist, scroll_to_index=len(loglist)-1)

                # windowに画像を表示
                imgHD_fill = cv.addWeighted(imgHD, 0.7, marker_fg, 0.5, 2.2)
                img_fill = cv.resize(imgHD_fill, (1280, 720))
                imgbytes = cv.imencode('.png', img_fill)[1].tobytes()
                window['image'].update(data=imgbytes)
                continue

if __name__ == "__main__":

    # VISCAカメラのプリセット0呼出し
    #visca.call_preset(0)

    # COMポート準備
    com_port = get_com_port()

    # GPU準備
    if USE_GPU:
        gpgpu.enable_gpu()

    # 推論用ネットワーク構築
    #detect_x, detect_y = detect.create(13)
    #getpos_x, getpos_y = getpos.create()
    #rotate_x, rotate_y = rotate.create()
    #gethole_x, gethole_y = gethole.create(16)
    #checkhole_x, checkhole_y = checkhole.create(4)
    #anomaly_x, anomaly_y = anomaly.create(8)
    #anomaly2_x, anomaly2_y = anomaly.create(2)
    #anomaly3_x, anomaly3_y = anomaly.create(3)
    holeexist_x, holeexist_y = holeexist.create(10)
    angle_x, angle_y = angle.create(2)

    dist_min = 0xffffff
    dist_max = 2

    # 検査対象選択画面表示し検査対象を選択
    TargetCode = None

    TargetCode = get_target()
    while TargetCode:
        if TargetCode == '設置位置調整':
            start_adjust_setting()
        elif TargetCode == '83-915202-09':
            capture_from_cam(None)
        TargetCode = get_target()
    
    capture_from_cam(None)
    #capture_from_cam("F://nnc_datasets/movie/230406/WIN_20230406_15_51_12_Pro.mp4", 120) # NGのみ, 学習済み
    #capture_from_cam("F://nnc_datasets/movie/221129/WIN_20221129_14_08_50_Pro.mp4", 2460)   # OK品のみ
    #capture_from_cam("F://nnc_datasets/movie/230406/WIN_20230406_15_42_11_Pro.mp4", 480)   # NGのみ, 未学習
    #capture_from_cam("F://nnc_datasets/movie/230406/WIN_20230406_15_48_45_Pro.mp4", 0)   # NGのみ, 未学習
    #capture_from_cam("F://nnc_datasets/movie/230413/WIN_20230413_10_34_19_Pro.mp4", 0)   # OKのみ, NG判定多数
    #capture_from_cam("F://nnc_datasets/movie/230413/WIN_20230413_10_34_19_Pro.mp4", 3700)   # OKのみ, NG判定多数, 初回NG品
    #capture_from_cam("F://nnc_datasets/movie/230413/WIN_20230413_10_34_19_Pro.mp4", 26600)   # OKのみ, NG判定多数, 明るさ調整後のNG品
    #capture_from_cam("F://nnc_datasets/movie/230413/WIN_20230413_10_34_19_Pro.mp4", 20500)   # OKのみ, NG判定多数, 明るさ調整後のNG品
    #capture_from_cam("F://nnc_datasets/movie/230726/WIN_20230726_15_05_32_Pro.mp4", 0)   # 
    #capture_from_cam("F://nnc_datasets/movie/230726/WIN_20230726_15_05_46_Pro.mp4", 0)   # 
    #capture_from_video()
    #capture_from_images()
