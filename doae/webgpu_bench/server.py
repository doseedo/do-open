import http.server, os
os.chdir('/tmp/webgpu_bench')

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()
    def log_message(self, *a): pass  # quiet

print('Serving at http://localhost:8877')
http.server.HTTPServer(('', 8877), Handler).serve_forever()
