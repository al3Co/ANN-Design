% code free =)

T = readtable('/home/disam/Documents/GitHub/ANN-Design/data/allMovements.xlsx');

IMUSq2 = [T.Quat2_1 T.Quat2_2 T.Quat2_3 T.Quat2_4];
[yaw, pitch, roll] = quat2angle(IMUSq2);
angIMU2 = [yaw, pitch, roll];

IMUSq1 = [T.Quat1 T.Quat2 T.Quat3 T.Quat4];
[yaw, pitch, roll] = quat2angle(IMUSq1);
angIMU1 = [yaw pitch roll];

clear yaw
clear pitch
clear roll

angIMUSubstract = (angIMU2 - angIMU1);
angOpti = [T.angBrazoX T.angBrazoY T.angBrazoZ];


figure
subplot(2,1,1)       % add first plot in 2 x 1 grid
plot(angIMUSubstract)
title('IMUs')

subplot(2,1,2)       % add second plot in 2 x 1 grid
plot(angOpti)        % plot using + markers
title('OptiTrack')