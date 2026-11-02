"""Host-side UART reader for the falling_glucose_guard hybrid VIO+UART
hardware test. Independently bypasses the JTAG/VIO debug-probe readback
path (found intermittently unreliable) by reading block/secded_status
straight off the Arty S7's onboard USB-UART bridge instead.

Frame format (1 byte, free-running, ~11.5k frames/sec at 115200 baud):
  bit0 = block, bits[2:1] = secded_status, bits[7:3] = 0

Usage: python uart_reader.py <COM_PORT> <DURATION_SECONDS> <OUT_LOG_PATH>
"""
import sys
import time
import serial

port, duration_s, out_path = sys.argv[1], float(sys.argv[2]), sys.argv[3]

ser = serial.Serial(port, 115200, timeout=0.5)
ser.reset_input_buffer()

t_end = time.time() + duration_s
with open(out_path, "w") as f:
    f.write("# t_ms,byte_hex,block,status\n")
    while time.time() < t_end:
        chunk = ser.read(256)
        t_ms = int(time.time() * 1000)
        for b in chunk:
            block = b & 1
            status = (b >> 1) & 3
            f.write(f"{t_ms},{b:02x},{block},{status}\n")

ser.close()
print(f"UART_READER_DONE wrote log to {out_path}")
