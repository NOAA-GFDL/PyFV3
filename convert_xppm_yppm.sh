#!/bin/bash
#
# Script to convert xppm.py/xtp_u.py into yppm/ytp_v.py. Can be deleted once we
# have a way to use the same codebase for x-direction and y-direction advection.
#

set -e -x

cp pyFV3/stencils/xppm.py pyFV3/stencils/yppm.py
cp pyFV3/stencils/xtp_u.py pyFV3/stencils/ytp_v.py

for fname in pyFV3/stencils/yppm.py pyFV3/stencils/ytp_v.py
do
    sed -i 's/ub/vb/g' $fname
    sed -i 's/dx/dy/g' $fname
    sed -i 's/xt/yt/g' $fname
    sed -i 's/eyternals/externals/g' $fname
    sed -i 's/xflux/yflux/g' $fname
    sed -i 's/_x/_y/g' $fname
    sed -i 's/_u/_v/g' $fname
    sed -i 's/u_/v_/g' $fname
    sed -i 's/u,/v,/g' $fname
    sed -i 's/u:/v:/g' $fname
    sed -i 's/u\[/v\[/g' $fname
    sed -i 's/u)/v)/g' $fname
    sed -i 's/iord/jord/g' $fname
    sed -i 's/\[-1, 0/\[0, -1/g' $fname
    sed -i 's/\[-2, 0/\[0, -2/g' $fname
    sed -i 's/\[1, 0/\[0, 1/g' $fname
    sed -i 's/\[2, 0/\[0, 2/g' $fname
    sed -i 's/ u / v /g' $fname
    sed -i 's/x-/y-/g' $fname
    sed -i 's/i_start/j_start/g' $fname
    sed -i 's/i_end/j_end/g' $fname
    sed -i 's/\[j_start - 1, :/\[:, j_start - 1/g' $fname
    sed -i 's/\[j_start, :/\[:, j_start/g' $fname
    sed -i 's/\[j_start + 1, :/\[:, j_start + 1/g' $fname
    sed -i 's/\[j_end - 2, :/\[:, j_end - 2/g' $fname
    sed -i 's/\[j_end - 1, :/\[:, j_end - 1/g' $fname
    sed -i 's/\[j_end, :/\[:, j_end/g' $fname
    sed -i 's/\[j_end + 1, :/\[:, j_end + 1/g' $fname
    sed -i 's/\[j_end + 2, :/\[:, j_end + 2/g' $fname
done

sed -i 's/i_start/j_start/g' pyFV3/stencils/yppm.py
sed -i 's/i_end/j_end/g' pyFV3/stencils/yppm.py
sed -i 's/XPiecewise/YPiecewise/g' pyFV3/stencils/yppm.py
sed -i 's/X Piecewise/Y Piecewise/g' pyFV3/stencils/yppm.py
sed -i 's/xppm/yppm/g' pyFV3/stencils/yppm.py
sed -i 's/u\*/v\*/g' pyFV3/stencils/yppm.py

sed -i 's/j_start - 1 : j_start + 1, j_start/i_start, j_start - 1 : j_start + 1/g' pyFV3/stencils/ytp_v.py
sed -i 's/j_start - 1 : j_start + 1, j_end + 1/i_end + 1, j_start - 1 : j_start + 1/g' pyFV3/stencils/ytp_v.py
sed -i 's/j_end : j_end + 2, j_start/i_start, j_end : j_end + 2/g' pyFV3/stencils/ytp_v.py
sed -i 's/j_end : j_end + 2, j_end + 1/i_end + 1, j_end : j_end + 2/g' pyFV3/stencils/ytp_v.py
sed -i 's/j_end, j_start, jord, j_end, j_start/i_end, i_start, j_end, j_start, jord/g' pyFV3/stencils/ytp_v.py
sed -i 's/xppm/yppm/g' pyFV3/stencils/ytp_v.py

sed -i 's/region\[j_start - 1 : j_start + 2, :\], region\[j_end - 1 : j_end + 2, :\]/region\[:, j_start - 1 : j_start + 2\], region\[:, j_end - 1 : j_end + 2\]/g' pyFV3/stencils/yppm.py
