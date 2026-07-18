# PDEForge appliance (spectral models — 32 of 37; no FEM stack).
# For the FEM models (cylinder flows, NACA) use Dockerfile.fenicsx.
#
#   docker build -t pdeforge .
#   docker run -v $PWD/data:/data pdeforge \
#       pdeforge generate --preset fno_darcy_2d --n 100 \
#       --resolution x=85 y=85 --seed 0 --out /data/darcy

FROM python:3.12-slim

WORKDIR /opt/pdeforge
COPY . .

RUN pip install --no-cache-dir ".[hdf5,zarr]" \
    && pdeforge models | head -5

CMD ["pdeforge", "--help"]
