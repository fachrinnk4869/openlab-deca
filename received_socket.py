import socket

HOST = '0.0.0.0'  # listen semua interface
PORT = 9000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))  # wajib di UDP untuk menerima data

print(f"Listening for UDP packets on {HOST}:{PORT}...")

try:
    while True:
        data, addr = sock.recvfrom(4096)  # UDP pakai recvfrom
        print(f"From {addr}:")
        try:
            print("As text:", data.decode('utf-8'))
        except UnicodeDecodeError:
            print("Bukan teks UTF-8, mungkin binary")
except KeyboardInterrupt:
    print("Stopped by user")
finally:
    sock.close()
