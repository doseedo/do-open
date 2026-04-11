FROM node:18-alpine AS build
WORKDIR /app
COPY package.json ./
RUN npm install
COPY . .
# Cloud Build sets CI=true automatically, which makes CRA promote ESLint
# warnings to errors. There's a long tail of pre-existing no-unused-vars
# / react-hooks/exhaustive-deps warnings under components-demo/ that
# would fail the build without this override. Local `npm run build`
# (and ./deploy.sh step 0) doesn't set CI, so the two diverge — this
# line keeps them aligned. See DEPLOY.md "Cloud Build failed during
# step 3" for the full story.
ENV CI=false
RUN npm run build

FROM nginx:alpine

COPY --from=build /app/build /usr/share/nginx/html
COPY login.html /usr/share/nginx/html/login.html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]
