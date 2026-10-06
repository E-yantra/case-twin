FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# The browser calls its own origin; nginx proxies /api to the backend service.
ARG VITE_API_URL=/api
ARG VITE_OPENWEBUI_URL=
ENV VITE_API_URL=$VITE_API_URL VITE_OPENWEBUI_URL=$VITE_OPENWEBUI_URL
RUN npm run build

FROM nginx:1.27-alpine
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
