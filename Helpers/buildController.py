import socket
import time

HOST = "localhost"
PORT = 5050

def send_command(cmd, retry_interval=2):
    """
    Server çalışıyorsa komutu gönderir, çalışmıyorsa bekleyip tekrar dener.
    retry_interval: saniye cinsinden bekleme süresi
    """
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((HOST, PORT))
                s.sendall(cmd.encode('utf-8'))
                print(f"[Client] Komut gönderildi: {cmd}")
                break  # Gönderince döngüden çık
        except ConnectionRefusedError:
            print(f"[Client] Server bağlantısı yok, {retry_interval} sn sonra tekrar denenecek...")
            time.sleep(retry_interval)
        except Exception as e:
            print(f"[Client] Hata: {e}")
            time.sleep(retry_interval)

if __name__ == "__main__":
    while True:
        cmd = input("Gönderilecek komut: ")  # terminalden komut al
        if cmd.lower() in ["exit", "quit"]:
            print("[Client] Çıkılıyor...")
            break
        send_command(cmd)
