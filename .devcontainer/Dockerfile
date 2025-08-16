# Use the official Python 3.13 image as the base image
FROM python:3.13-slim

# install git, curl and unzip
RUN apt-get update && apt-get install -y --no-install-recommends git curl unzip

# Install the 'uv' package
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# install nodejs
SHELL ["/bin/bash", "-c"]
ENV FNM_DIR=/opt/fnm
ARG FNM_INSTALL_VERSION=22
# install fnm
RUN curl -fsSL https://fnm.vercel.app/install | bash -s -- --install-dir ${FNM_DIR} --skip-shell
RUN ln -s ${FNM_DIR}/fnm /usr/bin/ && chmod +x /usr/bin/fnm
# install node
RUN fnm install ${FNM_INSTALL_VERSION}
RUN fnm alias default ${FNM_INSTALL_VERSION}
# add fnm for bash
RUN ln -s "${FNM_DIR}/aliases/default/bin/node" /usr/bin/node
RUN ln -s "${FNM_DIR}/aliases/default/bin/npm" /usr/bin/npm
RUN ln -s "${FNM_DIR}/aliases/default/bin/npx" /usr/bin/npx
