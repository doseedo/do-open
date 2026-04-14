# Shared proxy settings for every staging Modal route.
# Included via `include /etc/nginx/conf.d/staging_modal_proxy.inc;`
# next to each `proxy_pass` above.
proxy_ssl_server_name on;
proxy_set_header Host arlo--doseedo-stemphonic-staging-stemphonic-wsgi.modal.run;
proxy_set_header X-Forwarded-Proto $scheme;
proxy_http_version 1.1;
# Stemphonic generations can take minutes — raise read timeouts.
proxy_connect_timeout 30s;
proxy_send_timeout    300s;
proxy_read_timeout    300s;
# Latent bytes are ~tens of KB; audio-accepting endpoints (classify,
# extract-midi) can see a few MB — 50 MB ceiling matches prod nginx.
client_max_body_size  50M;
