#!/usr/bin/env bash
# Install a headless FreeCAD (official AppImage, extracted, no FUSE needed)
# plus the offscreen OpenGL libraries used for rendering.
#
#   scripts/install_freecad.sh [VERSION] [PREFIX]
#
# Afterwards freecadcmd lives at $PREFIX/squashfs-root/usr/bin/freecadcmd,
# which is where cadanim looks by default (override with CADANIM_FREECADCMD).
set -euo pipefail

VERSION="${1:-1.1.4}"
PREFIX="${2:-/opt/freecad}"
ASSET="FreeCAD_${VERSION}-Linux-x86_64-py311.AppImage"
URL="https://github.com/FreeCAD/FreeCAD/releases/download/${VERSION}/${ASSET}"

if [ -x "${PREFIX}/squashfs-root/usr/bin/freecadcmd" ]; then
  echo "FreeCAD already installed in ${PREFIX}"
else
  mkdir -p "${PREFIX}"
  cd "${PREFIX}"
  echo "Downloading ${URL}"
  curl -fSL -o FreeCAD.AppImage "${URL}"
  chmod +x FreeCAD.AppImage
  ./FreeCAD.AppImage --appimage-extract > /dev/null
  rm -f FreeCAD.AppImage
fi

if command -v apt-get > /dev/null; then
  apt-get install -y -q libosmesa6 libegl1 ffmpeg > /dev/null || \
    echo "warning: could not install libosmesa6/libegl1/ffmpeg (install them manually)"
fi

"${PREFIX}/squashfs-root/usr/bin/freecadcmd" --version
