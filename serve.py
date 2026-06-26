#!/usr/bin/env python3
"""一键：重建数据并启动本地服务器。

    python3 serve.py          # 默认 http://localhost:8731
    python3 serve.py 9000     # 自定义端口
"""
import sys, http.server, socketserver, webbrowser
from pathlib import Path
import build  # 复用 build.py

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8731
build.main()

site = Path(__file__).resolve().parent / "site"
import functools
Handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site))
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    url = f"http://localhost:{PORT}/"
    print(f"\n▶ 网站已启动: {url}  (Ctrl+C 退出)")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    httpd.serve_forever()
