#!/bin/bash
# GPU服务器开HTTP代理透SSH方案（绕过华为云ModelArts notebook出站网络限制）
#
# 背景：本NPU服务器是华为云ModelArts notebook环境，出站只能走HTTP代理
# proxy-notebook.modelarts.com:8083（squid），代理白名单只允许80/443等HTTP端口，
# SSH TCP端口32222无法透出（curl --proxytunnel <GPU_IP>:32222 报 403 ERR_ACCESS_DENIED）。
#
# 方案：在GPU服务器开个HTTP代理（squid/http-proxy.py）监听80或443端口（代理白名单允许），
# SSH流量通过此HTTP代理透出——用curl --proxytunnel建立CONNECT tunnel，
# ssh ProxyCommand通过此tunnel连接GPU服务器SSH 32222。
#
# 用法（在GPU服务器跑）：
#   chmod +x gpu_http_proxy.sh
#   ./gpu_http_proxy.sh
#
# 然后告知本NPU服务器，本机测连通性后传ckpt+源码+脚本跑GPU推理。

set -e

# GPU 服务器公网 IP（自动探测，探测失败时用环境变量 GPU_PUBLIC_IP 指定；勿硬编码进脚本）
GPU_PUBLIC_IP=${GPU_PUBLIC_IP:-$(curl -s --connect-timeout 5 https://ifconfig.me 2>/dev/null || echo '<GPU_PUBLIC_IP>')}
SSH_PORT=${SSH_PORT:-32222}

echo "=== GPU服务器开HTTP代理透SSH方案 ==="

# 1. 查GPU服务器环境
echo "---1. 查GPU服务器环境---"
hostname
uname -a
nvidia-smi 2>&1 | head -10 || echo "nvidia-smi不可用"

# 2. 查可用端口（80/443是否被占）
echo "---2. 查可用HTTP端口（80/443/8080/8000）---"
for p in 80 443 8080 8000 8888; do
  if ss -tlnp 2>/dev/null | grep -q ":$p "; then
    echo "端口$p被占"
  else
    echo "端口$p可用"
  fi
done

# 3. 安装socat（转发SSH 32222→HTTP代理监听端口）
echo "---3. 安装socat---"
if ! which socat > /dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq socat 2>&1 | tail -3 || pip install socat 2>&1 | tail -3
fi
which socat && socat -V | head -1

# 4. 开HTTP代理监听80端口（用python http.server简单实现CONNECT tunnel）
echo "---4. 开HTTP代理监听80端口（CONNECT tunnel透SSH 32222）---"
cat > /tmp/http_proxy_tunnel.py <<'PYEOF'
#!/usr/bin/env python3
"""简单HTTP代理透SSH：监听80端口，CONNECT tunnel转发到SSH 32222"""
import socket, threading, sys

LISTEN_PORT = 80  # 代理白名单允许的HTTP端口
SSH_HOST = '127.0.0.1'
SSH_PORT = 32222  # GPU服务器SSH真端口

def handle_connect(client, target_host, target_port):
    try:
        upstream = socket.create_connection((target_host, target_port), timeout=10)
        client.sendall(b'HTTP/1.1 200 Connection established\r\n\r\n')
        def pipe(a, b):
            try:
                while True:
                    data = a.recv(4096)
                    if not data: break
                    b.sendall(data)
            except: pass
            finally:
                try: a.close()
                except: pass
                try: b.close()
                except: pass
        t1 = threading.Thread(target=pipe, args=(client, upstream), daemon=True)
        t2 = threading.Thread(target=pipe, args=(upstream, client), daemon=True)
        t1.start(); t2.start()
        t1.join(); t2.join()
    except Exception as e:
        client.sendall(f'HTTP/1.1 502 Bad Gateway\r\n\r\n{e}'.encode())
        client.close()

def handle_client(client):
    try:
        data = b''
        while b'\r\n\r\n' not in data and len(data) < 8192:
            chunk = client.recv(4096)
            if not chunk: break
            data += chunk
        if not data: return
        first_line = data.split(b'\r\n')[0].decode(errors='ignore')
        print(f'请求: {first_line}', flush=True)
        if first_line.startswith('CONNECT'):
            # CONNECT host:port HTTP/1.1
            target = first_line.split()[1]
            host, port = target.rsplit(':', 1)
            print(f'CONNECT tunnel → {host}:{port}', flush=True)
            handle_connect(client, host, int(port))
        else:
            # 非CONNECT请求返403（只透SSH tunnel）
            client.sendall(b'HTTP/1.1 403 Forbidden (only CONNECT tunnel for SSH)\r\n\r\n')
            client.close()
    except Exception as e:
        print(f'handle_client error: {e}', flush=True)
        client.close()

def main():
    import os
    gpu_ip = os.environ.get('GPU_PUBLIC_IP', '<GPU_PUBLIC_IP>')
    ssh_port = os.environ.get('SSH_PORT', '32222')
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', LISTEN_PORT))
    server.listen(5)
    print(f'HTTP代理透SSH监听 0.0.0.0:{LISTEN_PORT} → SSH {SSH_HOST}:{SSH_PORT}', flush=True)
    print(f'本NPU服务器用: curl --proxytunnel --proxy http://{gpu_ip}:80 http://{gpu_ip}:{ssh_port}', flush=True)
    print(f'                ssh -o ProxyCommand="curl --proxytunnel --proxy http://{gpu_ip}:80 -s http://{gpu_ip}:{ssh_port}" -p {ssh_port} root@{gpu_ip}', flush=True)
    while True:
        client, addr = server.accept()
        print(f'连接来自 {addr}', flush=True)
        threading.Thread(target=handle_client, args=(client,), daemon=True).start()

if __name__ == '__main__':
    main()
PYEOF
echo "http_proxy_tunnel.py 已写到 /tmp/http_proxy_tunnel.py"

# 5. 启动HTTP代理（后台）
echo "---5. 启动HTTP代理透SSH（后台监听80端口）---"
pkill -f "http_proxy_tunnel.py" 2>/dev/null || true
nohup python3 /tmp/http_proxy_tunnel.py > /tmp/http_proxy_tunnel.log 2>&1 &
echo "HTTP代理PID=$! 启动"
sleep 3
ss -tlnp 2>/dev/null | grep ":80 " | head -1 || echo "80端口未监听"
tail -5 /tmp/http_proxy_tunnel.log 2>&1 | head -5

echo ""
echo "=== HTTP代理透SSH已启动 ==="
echo "本NPU服务器连通性测试命令："
echo "  curl -v --proxytunnel --proxy http://$GPU_PUBLIC_IP:80 --connect-timeout 10 http://$GPU_PUBLIC_IP:$SSH_PORT"
echo "若返'HTTP/1.1 200 Connection established'→代理透SSH连通"
echo ""
echo "SSH连接命令（本NPU服务器，用户名按实际填写）："
echo "  ssh -o StrictHostKeyChecking=no -o ProxyCommand='curl --proxytunnel --proxy http://$GPU_PUBLIC_IP:80 -s http://%h:%p' -p $SSH_PORT <user>@$GPU_PUBLIC_IP"
echo ""
echo "=== 跑完后告知本NPU服务器，本机测连通性后传ckpt+源码+脚本跑GPU推理 ==="
