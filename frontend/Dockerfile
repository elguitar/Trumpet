FROM node:current-slim

WORKDIR /usr/src/app

COPY package.json .

RUN npm install
RUN npm install --global http-server

EXPOSE 8080

COPY . .

RUN npm run build

CMD ["http-server", "dist"]
