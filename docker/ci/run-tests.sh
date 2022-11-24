#!/usr/bin/env sh
set -e

cd /source
pip install -U pip
make install
make install-all
make test
