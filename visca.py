# -*- coding: utf-8 -*-
import time
import serial
from serial.tools import list_ports

def get_visca_port():
    """
    HarrierVirtualSerialのCOMポートを探す
    """
    ports = list_ports.comports()
    visca_ports = [v for v in ports if "Harrier" in v.description]

    try:
        ser = serial.Serial()
        ser.port = visca_ports[0].device
        ser.baudrate = 9600

        return ser
    except:
        print("[Error] can't open a VISCA port")
        return None

def call_preset(n):
    """
    プリセットn番目を呼出し
    """
    data = b'\x81\x01\x04\x3F\x02' + n.to_bytes(1, 'big')  + b'\xFF'
    send_recv_cmd(data)

# CAM_PowerInq
def inq_power():
    resp = send_recv_cmd(b"\x81\x09\x04\x00\xff")
    
    if resp != -1 and resp[2] == 2:
        return 1
    else:
        return 0

def picture_flip(onoff):
    if onoff:
        send_recv_cmd(b'\x81\x01\x04\x66\x02\xFF')
    else:
        send_recv_cmd(b'\x81\x01\x04\x66\x03\xFF')

# CAM_PictureFlipModeInq
def inq_picture_flip():
    resp = send_recv_cmd(b"\x81\x09\x04\x66\xff")
    if resp == -1:
        return -1
    else:
        return resp[2]

# コマンド送受信
def send_recv_cmd(cmd):
    port = get_visca_port()
    if port is None:
        return -1

    try:
        port.open()
    except:
        print("port open error")
        return -1
    
    try:
        port.write(cmd)
        time.sleep(0.1)
        rep = port.read_all()
    except:
        return -1
    finally:
        port.close()

    if len(rep)==-1:
        return -1

    return rep

# カメラ電源ON待ち
def wait_power_on():
    start = time.time()
    while inq_power() == 0:
        time.sleep(1)
        if (time.time() - start) > 15:
            print("wait_power_on : timeout")
            return
    print("wait_power_on : success %d" % (time.time() - start))

if __name__ == "__main__":
    wait_power_on()
    rep = inq_picture_flip()
    print(rep)
    call_preset(0)
    rep = inq_picture_flip()
    print(rep)
