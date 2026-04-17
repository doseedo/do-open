FROM nginx:alpine

# build/ is pre-built locally by deploy.sh (npm run build).
# Do NOT add npm install / npm run build here — Cloud Build's npm
# produces a different bundle hash than the local build, which causes
# "Unexpected token '<'" (GCS returns HTML 404 for missing old chunks).
# See CLAUDE.md "Architecture" and DEPLOY.md for the full story.
COPY build/ /usr/share/nginx/html/
COPY login.html /usr/share/nginx/html/login.html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]
