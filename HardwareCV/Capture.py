import serial

ser = serial.Serial("/dev/cu.usbmodem1101", 115200, timeout=5)
buffer = bytearray()
recording = False
frame_num = 0

while True:
    line = ser.readline()
    if b"===FRAME_START===" in line:
        buffer = bytearray()
        recording = True
        continue
    if b"===FRAME_END===" in line and recording:
        with open(f"frame_{frame_num}.jpg", "wb") as f:
            f.write(buffer)
        print(f"Saved frame_{frame_num}.jpg ({len(buffer)} bytes)")
        frame_num += 1
        recording = False
        continue
    if recording:
        buffer.extend(line)
