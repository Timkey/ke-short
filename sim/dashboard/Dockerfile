FROM nginx:alpine

# Remove default site
RUN rm /etc/nginx/conf.d/default.conf

COPY nginx.conf /etc/nginx/conf.d/default.conf

# Dashboard static files
COPY index.html /usr/share/nginx/html/

# /data is mounted from the shared volume (engine output)
# nginx will serve it at /data/
EXPOSE 8080
