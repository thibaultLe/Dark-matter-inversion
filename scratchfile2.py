# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:44:05 2023

@author: Thibault
"""

import numpy as np
from matplotlib.pylab import plt

#IG3, true DM guess
# truelosses =       [9123,9215,8940,8974,8984,8990,8898,8898,8797,8634,
#                     8967,9142,9267,9032,9221,8910,8882,9094,8577,9044,
#                     9247,9198,9171,9186,9081,8977,9021,8919,9241,9194]
# recontrueDMguess = [9211,9407,9735,9183,9202,9234,8910,9706,8796,8867,
#                     9124,9780,9449,9183,9672,9154,9606,9106,9872,9293,
#                     9272,9251,9262,9186,9117,9121,9329,9663,9629,9271]


# reconNormalguess = [9193,9442,10028,9168,9168,9229,9767,9484,9178,9472,
#                     9266,9245,9255,9203,9097, 9202,8917,9799,8800,8894,
#                     9270,9610,9106,9892,9332,9174,9286,9837,9518,9264]


#Plum, BW, Zhao

#Plummer
PlumTrue = """500000
True loss: 9259.206731321317
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.35 iterations/ss/ss
Total time: 11071.17 seconds
Loss after training = 9308.542610148525
500001
True loss: 9224.797107213892
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.31 iterations/ss/ss
Total time: 11422.96 seconds
Loss after training = 9637.866747884347
500002
True loss: 9170.407069094363
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.59 iterations/ss/ss
Total time: 9433.93 seconds
Loss after training = 9279.863330547025
500003
True loss: 8984.198731374807
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.73 iterations/s/sss
Total time: 8682.61 seconds
Loss after training = 10469.507918904585
500004
True loss: 8950.865574447816
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.38 iterations/ss/ss
Total time: 10858.06 seconds
Loss after training = 8998.68101470738
500005
True loss: 8863.422888011839
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.21 iterations/ss/ss
Total time: 12436.61 seconds
Loss after training = 8935.506776762755
500006
True loss: 9203.249793071755
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.32 iterations/ss/ss
Total time: 11323.24 seconds
Loss after training = 9332.234340270756
500007
True loss: 9076.078561155895
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.39 iterations/ss/ss
Total time: 10827.73 seconds
Loss after training = 9256.941822210047
500008
True loss: 8848.726220382297
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.34 iterations/ss/ss
Total time: 11178.65 seconds
Loss after training = 8885.178499946625
500009
True loss: 9065.990607596745
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.46 iterations/ss/ss
Total time: 10264.31 seconds
Loss after training = 9333.26182990079
500010
True loss: 8839.93200621961
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.4 iterations/sns/ss
Total time: 10708.17 seconds
Loss after training = 9021.319804664272
500011
True loss: 9178.9299457118
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.47 iterations/ss/ss
Total time: 10214.7 seconds
Loss after training = 9333.316176591477
500012
True loss: 9251.112112777118
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.68 iterations/ss/ss
Total time: 8904.26 seconds
Loss after training = 9241.484788868924
500013
True loss: 9182.894759614397
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.84 iterations/ss/ss
Total time: 8144.53 seconds
Loss after training = 10014.105238528868
500014
True loss: 9152.592546419764
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.45 iterations/ss/ss
Total time: 10374.5 seconds
Loss after training = 9146.492252733075
500015
True loss: 9059.450384000062
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.33 iterations/ss/ss
Total time: 11262.71 seconds
Loss after training = 9067.127914193583
500016
True loss: 8909.412546115984
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.26 iterations/s/sss
Total time: 11929.87 seconds
Loss after training = 9203.9865390363
500017
True loss: 8972.724134827573
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.55 iterations/s/sss
Total time: 9686.89 seconds
Loss after training = 9003.1780447066
500018
True loss: 8981.80071883016
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.97 iterations/s/ss
Total time: 5046.72 seconds
Loss after training = 9065.98616158031
500019
True loss: 9108.92342398625
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4463.3 seconds
Loss after training = 10505.584952284258
500020
True loss: 9240.79357603686
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4452.04 seconds
Loss after training = 11199.881136915757
500021
True loss: 9240.096220776806
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4532.16 seconds
Loss after training = 9799.447181252594
500022
True loss: 9214.520431396386
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4454.29 seconds
Loss after training = 10587.166950987199
500023
True loss: 9121.181491994023
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4458.23 seconds
Loss after training = 9690.741699955532
500024
True loss: 8856.941579610966
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/ssss
Total time: 4553.32 seconds
Loss after training = 8870.046432799714
500025
True loss: 8945.553139128213
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4506.48 seconds
Loss after training = 9066.764520500454
500026
True loss: 8979.538359043392
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4502.37 seconds
Loss after training = 9028.296526825183
500027
True loss: 8950.333093220792
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/ssss
Total time: 4439.86 seconds
Loss after training = 8946.565456452015
500028
True loss: 9124.031290914332
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4509.51 seconds
Loss after training = 9231.365223527515
500029
True loss: 8790.554898817836
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/ssss
Total time: 4552.5 seconds
Loss after training = 9922.722029194154
500030
True loss: 8962.33218565428
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4519.91 seconds
Loss after training = 9017.75153299512
500031
True loss: 9194.040427413627
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4501.64 seconds
Loss after training = 9551.379940064335
500032
True loss: 8898.571342997016
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4536.07 seconds
Loss after training = 9517.786509319405
500033
True loss: 9039.451376636016
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 4550.59 seconds
Loss after training = 9053.658105004532
500034
True loss: 8871.56798031752
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4455.92 seconds
Loss after training = 8846.776361255263
500035
True loss: 8702.226440677136
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4464.86 seconds
Loss after training = 8715.277570291291
500036
True loss: 8965.313855861974
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4491.52 seconds
Loss after training = 8977.539084388538
500037
True loss: 8933.807605837468
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4523.33 seconds
Loss after training = 9719.473302455222
500038
True loss: 8994.140675857274
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4452.67 seconds
Loss after training = 8996.416417109582
500039
True loss: 9181.261053122891
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4506.25 seconds
Loss after training = 9353.665313097794
500040
True loss: 9195.84463358019
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4510.2 seconds
Loss after training = 9367.323034266064
500041
True loss: 9094.073324760691
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4495.05 seconds
Loss after training = 10737.291786899266
500042
True loss: 8968.64762369431
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 4574.88 seconds
Loss after training = 9502.749130593587
500043
True loss: 8756.920705409575
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 4580.05 seconds
Loss after training = 8846.048127808424
500044
True loss: 8861.757988295443
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4477.24 seconds
Loss after training = 8856.849299180834
500045
True loss: 9276.199116036754
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 4572.82 seconds
Loss after training = 10568.276949980062
500046
True loss: 9268.182458532994
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4488.39 seconds
Loss after training = 9687.298576154146
500047
True loss: 9179.198532278337
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/s/s
Total time: 4542.75 seconds
Loss after training = 9180.5091725189
500048
True loss: 8771.297817138318
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 4594.77 seconds
Loss after training = 8910.51668066815
500049
True loss: 9093.058722201426
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4510.02 seconds
Loss after training = 9103.439559360793
500050
True loss: 9194.406531241297
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4505.32 seconds
Loss after training = 9199.237479392506
500051
True loss: 9317.553024894012
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4464.64 seconds
Loss after training = 9316.419929077056
500052
True loss: 9038.74295138601
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4502.78 seconds
Loss after training = 9204.300362489974
500053
True loss: 8725.762061552177
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4480.97 seconds
Loss after training = 8724.579037884223
500054
True loss: 8768.921473626933
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4476.65 seconds
Loss after training = 9656.109954229618
500055
True loss: 8943.288557878832
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4531.65 seconds
Loss after training = 9217.276689339707
500056
True loss: 8699.571336736835
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4511.54 seconds
Loss after training = 8839.554287103747
500057
True loss: 8910.464281722576
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4528.82 seconds
Loss after training = 9045.116903329532
500058
True loss: 8889.5839261589
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/s/s
Total time: 4544.88 seconds
Loss after training = 9014.206176218675
500059
True loss: 8890.415049903662
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4492.2 seconds
Loss after training = 8961.667924020689
500060
True loss: 8879.240623367852
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 4549.43 seconds
Loss after training = 8986.00634638198
500061
True loss: 9016.75194143034
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4505.17 seconds
Loss after training = 9318.66347899464
500062
True loss: 8895.381827430443
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4518.33 seconds
Loss after training = 8953.038765892687
500063
True loss: 8771.234040427782
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 4546.87 seconds
Loss after training = 8796.161498328469
500064
True loss: 9020.066502992633
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4473.61 seconds
Loss after training = 9144.029197936887
500065
True loss: 9007.164871975643
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/ss/ss
Total time: 4540.46 seconds
Loss after training = 9192.334560140629
500066
True loss: 9144.274329207628
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/ssss
Total time: 4553.81 seconds
Loss after training = 9140.441008375772
500067
True loss: 9157.172502552645
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4480.41 seconds
Loss after training = 9282.284499415733
500068
True loss: 9183.304078171046
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4451.95 seconds
Loss after training = 9536.086876333095
500069
True loss: 8926.320059377646
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4510.24 seconds
Loss after training = 9143.722751340541
500070
True loss: 8874.911351038036
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4470.75 seconds
Loss after training = 9814.61069580414
500071
True loss: 8894.32487210288
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4456.53 seconds
Loss after training = 9123.49853668347
500072
True loss: 8800.859327424574
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 4430.63 seconds
Loss after training = 8828.565790944254
500073
True loss: 9121.132404671502
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4524.32 seconds
Loss after training = 9308.412619489876
500074
True loss: 9217.722853730676
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/ssss
Total time: 4558.07 seconds
Loss after training = 9242.120144448716
500075
True loss: 9365.032862240394
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.16 iterations/s/ss
Total time: 4739.96 seconds
Loss after training = 9522.161070367387
500076
True loss: 9222.926522929407
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.2 iterations/ss/ss
Total time: 4682.06 seconds
Loss after training = 9229.426819938026
500077
True loss: 8949.019438960839
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4496.22 seconds
Loss after training = 8983.154691642752
500078
True loss: 8999.285459363864
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4482.31 seconds
Loss after training = 9394.753030831016
500079
True loss: 8995.443925901376
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/s/ss
Total time: 4591.96 seconds
Loss after training = 9052.709217191243
500080
True loss: 8960.168494647909
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4506.97 seconds
Loss after training = 8963.68640505453
500081
True loss: 8997.327294443556
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4479.84 seconds
Loss after training = 8992.05599146706
500082
True loss: 9002.715124841432
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4529.99 seconds
Loss after training = 9038.481731601642
500083
True loss: 9276.607243761116
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4520.33 seconds
Loss after training = 10214.283449670038
500084
True loss: 8938.38316811365
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4453.0 seconds
Loss after training = 9401.7115703808
500085
True loss: 8922.276451102256
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4516.89 seconds
Loss after training = 8953.61826703364
500086
True loss: 8937.889392759482
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4470.98 seconds
Loss after training = 9456.764703638833
500087
True loss: 8946.388154667044
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4510.93 seconds
Loss after training = 8950.181003186866
500088
True loss: 9258.577865141946
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4487.7 seconds
Loss after training = 10001.298944511556
500089
True loss: 9084.551791314172
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4522.07 seconds
Loss after training = 9220.795528409542"""

#BW:
BWTrue = """500100
True loss: 8892.181903227243
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.97 iterations/ss/s
Total time: 7633.39 seconds
Loss after training = 9277.010143364618
500101
True loss: 8971.57438670103
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.28 iterations/s/sss
Total time: 11750.04 seconds
Loss after training = 9228.542178523176
500102
True loss: 9041.029806971968
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.49 iterations/ss/ss
Total time: 10062.09 seconds
Loss after training = 9084.761772642418
500103
True loss: 9064.07002604847
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.48 iterations/ss/ss
Total time: 10148.27 seconds
Loss after training = 10179.1526073837
500104
True loss: 8925.98322640005
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.32 iterations/ss/ss
Total time: 11397.72 seconds
Loss after training = 9017.852904598003
500105
True loss: 9008.204117721825
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.44 iterations/ss/ss
Total time: 10407.33 seconds
Loss after training = 9892.809650567751
500106
True loss: 9057.20360462636
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.32 iterations/ss/s
Total time: 11347.13 seconds
Loss after training = 9197.61032175811
500107
True loss: 8966.149017025802
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.24 iterations/ss/ss
Total time: 12057.82 seconds
Loss after training = 8969.263110968315
500108
True loss: 8667.644597420287
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.38 iterations/ss/ss
Total time: 10865.74 seconds
Loss after training = 9990.094976695393
500109
True loss: 9267.978932607479
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.26 iterations/ss/ss
Total time: 11890.87 seconds
Loss after training = 9947.798109117388
500110
True loss: 8780.345794696823
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.59 iterations/ss/ss
Total time: 9422.3 seconds
Loss after training = 8806.987411524242
500111
True loss: 8860.291421299977
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.74 iterations/ss/ss
Total time: 8627.69 seconds
Loss after training = 9389.873001719881
500112
True loss: 9066.53394098549
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.82 iterations/ss/ss
Total time: 8233.67 seconds
Loss after training = 9412.242057246645
500113
True loss: 9076.384554620005
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.55 iterations/ss/ss
Total time: 9698.51 seconds
Loss after training = 9661.327818559814
500114
True loss: 9041.432029026557
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.36 iterations/s/sss
Total time: 11062.84 seconds
Loss after training = 9038.203625050088
500115
True loss: 9003.44889601393
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.27 iterations/ss/ss
Total time: 11845.02 seconds
Loss after training = 9166.699670777392
500116
True loss: 8893.000647040168
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.18 iterations/ss/ss
Total time: 12680.69 seconds
Loss after training = 8947.900018917016
500117
True loss: 8836.860073870705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.44 iterations/ss/ss
Total time: 10417.22 seconds
Loss after training = 8833.597711834143
500118
True loss: 8904.322342029063
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.17 iterations/s/ss
Total time: 4736.61 seconds
Loss after training = 9893.731139572325
500119
True loss: 9058.411554731392
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4485.37 seconds
Loss after training = 9067.671508242243
500120
True loss: 9067.554716971803
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4451.15 seconds
Loss after training = 9055.346584604038
500121
True loss: 8819.771687358221
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4504.11 seconds
Loss after training = 9434.079908455222
500122
True loss: 8983.98046257182
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4512.77 seconds
Loss after training = 9068.796720552406
500123
True loss: 8899.6034744785
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4462.05 seconds
Loss after training = 8933.093396757853
500124
True loss: 9111.15494943871
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4498.26 seconds
Loss after training = 9383.900118255722
500125
True loss: 9328.81899002641
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/s/ss
Total time: 4591.04 seconds
Loss after training = 9707.297870767763
500126
True loss: 9103.145862425656
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4484.52 seconds
Loss after training = 9880.290651546211
500127
True loss: 8858.884026384823
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4467.22 seconds
Loss after training = 9615.990539492228
500128
True loss: 8851.153510060944
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4535.03 seconds
Loss after training = 8850.03796505374
500129
True loss: 8715.133278564006
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4518.88 seconds
Loss after training = 8707.01073144917
500130
True loss: 8944.714075510137
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4525.66 seconds
Loss after training = 8955.6156161007
500131
True loss: 8844.493222441391
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4475.9 seconds
Loss after training = 8860.631924692429
500132
True loss: 9112.525443807657
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/ss/ss
Total time: 4541.26 seconds
Loss after training = 9113.687626011028
500133
True loss: 9031.449003738508
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4502.32 seconds
Loss after training = 9062.855133562858
500134
True loss: 9422.393405479275
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4525.7 seconds
Loss after training = 10534.089679177972
500135
True loss: 9058.727723264514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4484.83 seconds
Loss after training = 9194.16125343541
500136
True loss: 9092.859423061413
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4495.08 seconds
Loss after training = 9096.653195991548
500137
True loss: 8849.278561261186
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4515.85 seconds
Loss after training = 8870.369368840775
500138
True loss: 9231.436162326218
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4514.0 seconds
Loss after training = 9244.995583442214
500139
True loss: 9129.785024945879
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4533.66 seconds
Loss after training = 9332.704697864408
500140
True loss: 8950.667073432096
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4535.31 seconds
Loss after training = 9127.070794964688
500141
True loss: 9206.209531826948
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4529.1 seconds
Loss after training = 9208.53513866592
500142
True loss: 9079.141037894491
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4512.58 seconds
Loss after training = 9255.329275588632
500143
True loss: 9068.950211038002
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4519.79 seconds
Loss after training = 10379.432579754955
500144
True loss: 8853.62678083281
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4501.46 seconds
Loss after training = 8857.961617862107
500145
True loss: 8758.35461937372
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/ss/s
Total time: 4616.92 seconds
Loss after training = 9223.588272740179
500146
True loss: 8968.28455542767
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4533.1 seconds
Loss after training = 10095.106410039767
500147
True loss: 8783.723632937523
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4489.31 seconds
Loss after training = 8979.615836817833
500148
True loss: 8933.746708351413
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4520.39 seconds
Loss after training = 9459.317089785352
500149
True loss: 9117.17730346635
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4531.84 seconds
Loss after training = 9405.151769173903
500150
True loss: 9073.598564318565
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 4628.96 seconds
Loss after training = 9164.41660799011
500151
True loss: 8885.205758459319
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4527.83 seconds
Loss after training = 9042.31961174326
500152
True loss: 8952.353372789774
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/ssss
Total time: 4563.41 seconds
Loss after training = 9381.5386546026
500153
True loss: 8815.075346669577
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4509.02 seconds
Loss after training = 8806.33554744747
500154
True loss: 9223.626505991879
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4496.93 seconds
Loss after training = 10012.14920700484
500155
True loss: 9054.226199851264
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4488.46 seconds
Loss after training = 9059.31336865226
500156
True loss: 8859.881160979496
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4480.74 seconds
Loss after training = 8926.21935787661
500157
True loss: 9094.354700430522
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4459.91 seconds
Loss after training = 9190.141721137134
500158
True loss: 8923.268051447674
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4509.18 seconds
Loss after training = 9083.711410654156
500159
True loss: 9159.123130760154
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4528.67 seconds
Loss after training = 9147.913740401049
500160
True loss: 8738.857550127601
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4496.37 seconds
Loss after training = 9335.514956025532
500161
True loss: 9050.50035912183
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 4600.77 seconds
Loss after training = 9347.608888573996
500162
True loss: 9072.5240325338
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4480.92 seconds
Loss after training = 9073.220239989581
500163
True loss: 8896.014899958878
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4492.42 seconds
Loss after training = 9001.816766141415
500164
True loss: 9349.401645304357
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/s/s
Total time: 4411.98 seconds
Loss after training = 9705.413717655241
500165
True loss: 8975.54457042394
Iterating: |██████████████████████████████| 100.0% complete,8 secs remaining, 0.12 iterations/sns/ss
Total time: 120132.68 seconds
Loss after training = 9239.880803915285
500166
True loss: 8928.53138233524
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 4548.21 seconds
Loss after training = 9461.838206765491
500167
True loss: 9063.907220489353
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.47 iterations/s/ss
Total time: 6076.67 seconds
Loss after training = 9258.605839502663
500168
True loss: 9190.098500741784
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.55 iterations/ss/ss
Total time: 9652.03 seconds
Loss after training = 9342.607924294976
500169
True loss: 8958.64191302387
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.35 iterations/ss/ss
Total time: 11116.14 seconds
Loss after training = 9137.345663000922
500170
True loss: 8907.147129526598
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.44 iterations/ss/ss
Total time: 10443.44 seconds
Loss after training = 9051.616434741873
500171
True loss: 9288.269348972555
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.27 iterations/s/sss
Total time: 11835.96 seconds
Loss after training = 9459.911377485745
500172
True loss: 9029.585231306639
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.2 iterations/sns/s
Total time: 12450.6 seconds
Loss after training = 10199.257194489852
500173
True loss: 8935.177227452526
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.57 iterations/ss/ss
Total time: 9572.51 seconds
Loss after training = 8935.997781983684
500174
True loss: 8861.91986684177
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.77 iterations/s/sss
Total time: 8459.36 seconds
Loss after training = 9685.459882947378
500175
True loss: 9011.389021843539
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.84 iterations/ss/ss
Total time: 8148.27 seconds
Loss after training = 9091.04037671701
500176
True loss: 9182.282667182764
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.33 iterations/ss/ss
Total time: 11264.72 seconds
Loss after training = 9204.460405139502
500177
True loss: 8953.722820361469
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.42 iterations/ss/ss
Total time: 10555.36 seconds
Loss after training = 8972.322650395883
500178
True loss: 8781.09048867889
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.18 iterations/ss/ss
Total time: 12750.37 seconds
Loss after training = 8776.853384269205
500179
True loss: 8996.816822800072
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.39 iterations/ss/ss
Total time: 10813.11 seconds
Loss after training = 9001.23878944429
500180
True loss: 9340.90622704481
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.22 iterations/ss/ss
Total time: 12287.26 seconds
Loss after training = 10479.903843676146
500181
True loss: 8690.871067322365
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.29 iterations/ss/ss
Total time: 11611.3 seconds
Loss after training = 8771.573404963841
500182
True loss: 9133.419327561885
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.24 iterations/ss/ss
Total time: 12112.82 seconds
Loss after training = 10412.550810442575
500183
True loss: 8986.164272097092
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.25 iterations/s/sss
Total time: 11978.55 seconds
Loss after training = 9081.285488549263
500184
True loss: 8682.769489100165
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.47 iterations/ss/ss
Total time: 10191.69 seconds
Loss after training = 8835.17429828713
500185
True loss: 8931.708421227228
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.23 iterations/ss/ss
Total time: 12214.08 seconds
Loss after training = 8925.059415108497
500186
True loss: 8968.768473001468
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.37 iterations/ss/ss
Total time: 10967.12 seconds
Loss after training = 8963.830503101482
500187
True loss: 9025.372489413217
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.76 iterations/ss/ss
Total time: 8521.93 seconds
Loss after training = 9908.231625435506
500188
True loss: 8691.733615316925
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.77 iterations/ss/ss
Total time: 8470.52 seconds
Loss after training = 9025.796715424021
500189
True loss: 8805.27903565785
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.31 iterations/ss/ss
Total time: 11433.89 seconds
Loss after training = 8818.340617087815"""

#Zhao:
ZhaoTrue = """500200
True loss: 8919.511107100729
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.25 iterations/ss/ss
Total time: 11999.29 seconds
Loss after training = 8908.238073568242
500201
True loss: 8884.45177328381
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.65 iterations/ss/ss
Total time: 9102.61 seconds
Loss after training = 8909.624115931638
500202
True loss: 8722.665978554107
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.42 iterations/ss/s
Total time: 10567.37 seconds
Loss after training = 10007.886587173152
500203
True loss: 9096.579415340817
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.26 iterations/ss/ss
Total time: 11898.23 seconds
Loss after training = 9364.357886352494
500204
True loss: 9284.517317380023
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.23 iterations/ss/ss
Total time: 12178.69 seconds
Loss after training = 9308.857645413862
500205
True loss: 8771.608570374236
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.27 iterations/ss/ss
Total time: 11773.93 seconds
Loss after training = 8843.090581134576
500206
True loss: 9281.350508765276
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.23 iterations/ss/ss
Total time: 12194.96 seconds
Loss after training = 10480.582924704358
500207
True loss: 8947.615633798865
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.61 iterations/ss/ss
Total time: 9289.68 seconds
Loss after training = 9006.522452491892
500208
True loss: 9107.20467597771
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.24 iterations/ss/ss
Total time: 12075.41 seconds
Loss after training = 9137.143496104698
500209
True loss: 8697.232164609553
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.57 iterations/ss/ss
Total time: 9538.04 seconds
Loss after training = 9061.082306706583
500210
True loss: 8984.949890370706
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.39 iterations/ss/ss
Total time: 10774.71 seconds
Loss after training = 8995.519427071125
500211
True loss: 9162.541760868327
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.22 iterations/ss/ss
Total time: 12245.9 seconds
Loss after training = 9647.432305877759
500212
True loss: 9093.926855921189
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.33 iterations/ss/ss
Total time: 11312.59 seconds
Loss after training = 9195.37742470067
500213
True loss: 8965.07191202816
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.61 iterations/ss/ss
Total time: 9327.87 seconds
Loss after training = 9092.63460203783
500214
True loss: 9097.548648530767
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.25 iterations/s/sss
Total time: 11971.39 seconds
Loss after training = 9884.987057815566
500215
True loss: 9039.589573476826
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.42 iterations/ss/ss
Total time: 10548.89 seconds
Loss after training = 9424.031803357102
500216
True loss: 8783.201418717495
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.32 iterations/ss/ss
Total time: 11322.77 seconds
Loss after training = 8787.724553026907
500217
True loss: 8772.590651005568
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.96 iterations/s/ss
Total time: 5070.2 seconds
Loss after training = 9155.236662358853
500218
True loss: 9119.903200563978
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 4428.55 seconds
Loss after training = 9167.665925157076
500219
True loss: 9292.883393322056
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4491.22 seconds
Loss after training = 10319.69119421241
500220
True loss: 9004.040584656908
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.42 iterations/s/ss
Total time: 4385.78 seconds
Loss after training = 9135.155183192655
500221
True loss: 8779.27319667532
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4517.04 seconds
Loss after training = 8852.869728347861
500222
True loss: 9053.982473549087
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4496.67 seconds
Loss after training = 9811.289218671574
500223
True loss: 9533.591406024294
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4454.62 seconds
Loss after training = 9595.579583763276
500224
True loss: 9196.166071394478
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4522.83 seconds
Loss after training = 9190.4788913277
500225
True loss: 9006.073592087161
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/ssss
Total time: 4444.81 seconds
Loss after training = 9774.164001421403
500226
True loss: 8886.998599529663
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4451.78 seconds
Loss after training = 9458.507854104804
500227
True loss: 9091.457808436771
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4517.83 seconds
Loss after training = 9094.424173792233
500228
True loss: 9063.60343312376
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4466.46 seconds
Loss after training = 9373.438050892524
500229
True loss: 9383.49351314638
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/ssss
Total time: 4438.28 seconds
Loss after training = 10496.987093253698
500230
True loss: 8999.620844975869
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4459.06 seconds
Loss after training = 9461.888142107624
500231
True loss: 9044.500167701197
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4499.25 seconds
Loss after training = 9548.732292016277
500232
True loss: 9312.245398095176
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/ssss
Total time: 4439.69 seconds
Loss after training = 9832.201260692744
500233
True loss: 8895.317140227029
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4470.12 seconds
Loss after training = 8893.032250182356
500234
True loss: 9220.257605840288
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4503.69 seconds
Loss after training = 10162.118563883445
500235
True loss: 8969.81728572428
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4464.77 seconds
Loss after training = 9081.459995958594
500236
True loss: 9238.644264933342
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/ss/s
Total time: 4619.05 seconds
Loss after training = 9441.5981133471
500237
True loss: 8793.98485280421
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4490.83 seconds
Loss after training = 8786.644860198376
500238
True loss: 8858.38537376315
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/ss
Total time: 4415.16 seconds
Loss after training = 10619.048962420524
500239
True loss: 9157.279975918924
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.41 iterations/s/ss
Total time: 4398.51 seconds
Loss after training = 9194.876256356341
500240
True loss: 9031.289855055644
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4488.66 seconds
Loss after training = 10628.055140676519
500241
True loss: 8994.840904160645
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4465.52 seconds
Loss after training = 9122.29097385376
500242
True loss: 8791.861022592853
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4489.1 seconds
Loss after training = 9328.850823391746
500243
True loss: 9047.058816338511
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4465.72 seconds
Loss after training = 9109.721317050251
500244
True loss: 9057.713632031448
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4489.6 seconds
Loss after training = 9088.167533886679
500245
True loss: 8795.250034744526
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 4419.46 seconds
Loss after training = 8959.175536007046
500246
True loss: 8877.284187500287
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4477.28 seconds
Loss after training = 8890.54913311949
500247
True loss: 9106.774053507595
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/ssss
Total time: 4439.02 seconds
Loss after training = 9329.467333566152
500248
True loss: 8842.21345752241
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4505.53 seconds
Loss after training = 8961.262687522527
500249
True loss: 8897.95126519101
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4475.41 seconds
Loss after training = 8914.502745557087
500250
True loss: 8951.282275660758
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4454.38 seconds
Loss after training = 9705.72882489372
500251
True loss: 8888.372372821486
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4479.0 seconds
Loss after training = 8896.383734156128
500252
True loss: 9056.47029220811
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4456.18 seconds
Loss after training = 9084.992900125588
500253
True loss: 8865.503384624371
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4504.46 seconds
Loss after training = 10109.137683731005
500254
True loss: 8833.251164182788
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 4431.3 seconds
Loss after training = 8852.225282783229
500255
True loss: 9289.558729626033
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.42 iterations/s/ss
Total time: 4380.61 seconds
Loss after training = 10236.619433619968
500256
True loss: 8887.133690381186
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/ss/s
Total time: 4613.02 seconds
Loss after training = 9135.120550058915
500257
True loss: 9255.582207626872
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4451.22 seconds
Loss after training = 9267.630651171858
500258
True loss: 8845.741056917446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4477.46 seconds
Loss after training = 9136.842151933522
500259
True loss: 8947.797406322805
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.41 iterations/s/ss
Total time: 4405.22 seconds
Loss after training = 9252.673514574517
500260
True loss: 8958.255958435851
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4459.41 seconds
Loss after training = 9925.323212112608
500261
True loss: 9241.709956876146
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4481.38 seconds
Loss after training = 9304.200765002932
500262
True loss: 8967.225383017374
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4487.34 seconds
Loss after training = 8969.640630480657
500263
True loss: 9046.374756290486
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/sss
Total time: 4414.81 seconds
Loss after training = 11004.323268470682
500264
True loss: 8857.211679655991
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4451.84 seconds
Loss after training = 8997.645815912256
500265
True loss: 8738.204022411308
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4532.86 seconds
Loss after training = 9183.760590006801
500266
True loss: 9059.9182572902
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4458.81 seconds
Loss after training = 9088.01308959664
500267
True loss: 9091.217120772173
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/ssss
Total time: 4438.41 seconds
Loss after training = 9156.144643695945
500268
True loss: 9188.133260332157
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4525.19 seconds
Loss after training = 9233.127512072264
500269
True loss: 8982.599973570437
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4502.33 seconds
Loss after training = 9009.74115872157
500270
True loss: 9040.397108011437
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/sss
Total time: 4415.02 seconds
Loss after training = 9084.828279312904
500271
True loss: 8747.906999417868
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/ssss
Total time: 4434.71 seconds
Loss after training = 8831.279995384155
500272
True loss: 9099.457191940463
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 4545.0 seconds
Loss after training = 9159.34714417009
500273
True loss: 9261.181961367334
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/ssss
Total time: 4436.71 seconds
Loss after training = 9336.555884542016
500274
True loss: 8759.168626088353
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/ssss
Total time: 4564.88 seconds
Loss after training = 10342.06093408208
500275
True loss: 9048.505951743186
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.13 iterations/s/ss
Total time: 4798.71 seconds
Loss after training = 9049.158183136933
500276
True loss: 8999.594012274198
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4479.61 seconds
Loss after training = 9001.272993777822
500277
True loss: 8731.943190935108
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4445.02 seconds
Loss after training = 9126.691742081193
500278
True loss: 9175.718269928146
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4496.93 seconds
Loss after training = 9510.830106225074
500279
True loss: 8968.626823944538
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4479.21 seconds
Loss after training = 8970.835868763694
500280
True loss: 9038.454794365724
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4504.91 seconds
Loss after training = 9306.700133313287
500281
True loss: 9225.23614155775
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4477.43 seconds
Loss after training = 9555.7727322654
500282
True loss: 9049.506048679688
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 4430.62 seconds
Loss after training = 9316.775235160672
500283
True loss: 9254.227958100706
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4485.14 seconds
Loss after training = 9514.773220781577
500284
True loss: 8842.396890710887
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/ssss
Total time: 4440.16 seconds
Loss after training = 8856.57297447927
500285
True loss: 9287.441329534895
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4475.8 seconds
Loss after training = 9849.67741701654
500286
True loss: 8845.20755799684
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4500.91 seconds
Loss after training = 9011.074549904028
500287
True loss: 8971.768066174875
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4502.79 seconds
Loss after training = 8985.710746578336
500288
True loss: 9151.716206870533
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4456.95 seconds
Loss after training = 9147.68565260733
500289
True loss: 8721.351871398465
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4475.45 seconds
Loss after training = 8898.914715148727"""



""" #######################################################
    RECONSTRUCTED NORMAL
    #######################################################
    #######################################################
    #######################################################
    #######################################################
    #######################################################
    #######################################################
    #######################################################"""

#Plummer, BW, Zhao:
PlumNormal = """500000
True loss: 9259.206731321317
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.45 iterations/ss/ss
Total time: 10376.8 seconds
Loss after training = 9297.16388260341
500001
True loss: 9224.797107213892
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.57 iterations/ss/ss
Total time: 9537.03 seconds
Loss after training = 9618.330283139658
500002
True loss: 9170.407069094363
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.37 iterations/s/sss
Total time: 10951.79 seconds
Loss after training = 9317.840688701182
500003
True loss: 8984.198731374807
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.48 iterations/s/sss
Total time: 10117.37 seconds
Loss after training = 10821.670578224474
500004
True loss: 8950.865574447816
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.68 iterations/ss/ss
Total time: 8943.39 seconds
Loss after training = 8990.612503561844
500005
True loss: 8863.422888011839
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.65 iterations/ss/ss
Total time: 9115.13 seconds
Loss after training = 8912.960036395029
500006
True loss: 9203.249793071755
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.21 iterations/s/sss
Total time: 12392.54 seconds
Loss after training = 9308.317349735646
500007
True loss: 9076.078561155895
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.4 iterations/sns/ss
Total time: 10706.64 seconds
Loss after training = 9253.371270987274
500008
True loss: 8848.726220382297
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.25 iterations/ss/ss
Total time: 12014.11 seconds
Loss after training = 8862.114714587382
500009
True loss: 9065.990607596745
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.5 iterations/sns/ss
Total time: 9987.45 seconds
Loss after training = 9383.728568322253
500010
True loss: 8839.93200621961
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.45 iterations/ss/ss
Total time: 10331.23 seconds
Loss after training = 9040.012698072824
500011
True loss: 9178.9299457118
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.52 iterations/ss/ss
Total time: 9874.83 seconds
Loss after training = 9369.82536191065
500012
True loss: 9251.112112777118
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.29 iterations/s/sss
Total time: 11583.84 seconds
Loss after training = 9246.312645926299
500013
True loss: 9182.894759614397
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.9 iterations/sns/ss
Total time: 7882.63 seconds
Loss after training = 10101.366542028785
500014
True loss: 9152.592546419764
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.51 iterations/ss/s
Total time: 9932.26 seconds
Loss after training = 9154.782832305482
500015
True loss: 9059.450384000062
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.31 iterations/ss/ss
Total time: 11418.26 seconds
Loss after training = 9087.045031740945
500016
True loss: 8909.412546115984
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.7 iterations/sns/ss
Total time: 8809.54 seconds
Loss after training = 9253.375014823841
500017
True loss: 8972.724134827573
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.24 iterations/ss/ss
Total time: 12119.52 seconds
Loss after training = 8989.980811567952
500018
True loss: 8981.80071883016
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.41 iterations/ss/ss
Total time: 6226.45 seconds
Loss after training = 9052.726220024582
500019
True loss: 9108.92342398625
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/sss
Total time: 4408.45 seconds
Loss after training = 10879.723964860983
500020
True loss: 9240.79357603686
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.41 iterations/s/ss
Total time: 4401.56 seconds
Loss after training = 11250.064390027892
500021
True loss: 9240.096220776806
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/ssss
Total time: 4438.84 seconds
Loss after training = 10232.219363555743
500022
True loss: 9214.520431396386
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/sss
Total time: 4411.89 seconds
Loss after training = 10906.716526290746
500023
True loss: 9121.181491994023
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/ssss
Total time: 4444.76 seconds
Loss after training = 9649.899629437727
500024
True loss: 8856.941579610966
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4473.67 seconds
Loss after training = 8885.977714150811
500025
True loss: 8945.553139128213
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/s
Total time: 4420.81 seconds
Loss after training = 9093.472941175402
500026
True loss: 8979.538359043392
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/sss
Total time: 4413.13 seconds
Loss after training = 9039.913413246879
500027
True loss: 8950.333093220792
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.43 iterations/s/ss
Total time: 4378.85 seconds
Loss after training = 8948.126167527364
500028
True loss: 9124.031290914332
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.42 iterations/s/ss
Total time: 4387.74 seconds
Loss after training = 9265.553474780298
500029
True loss: 8790.554898817836
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4474.39 seconds
Loss after training = 10215.133164271441
500030
True loss: 8962.33218565428
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4460.3 seconds
Loss after training = 9032.332938675514
500031
True loss: 9194.040427413627
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.41 iterations/s/ss
Total time: 4398.49 seconds
Loss after training = 9567.565034300833
500032
True loss: 8898.571342997016
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4481.68 seconds
Loss after training = 9526.084890987093
500033
True loss: 9039.451376636016
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4468.84 seconds
Loss after training = 9056.33116320642
500034
True loss: 8871.56798031752
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 4428.08 seconds
Loss after training = 8846.682907746932
500035
True loss: 8702.226440677136
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.43 iterations/s/ss
Total time: 4378.69 seconds
Loss after training = 8728.742150322143
500036
True loss: 8965.313855861974
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4458.01 seconds
Loss after training = 8973.828303619022
500037
True loss: 8933.807605837468
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.41 iterations/s/ss
Total time: 4396.65 seconds
Loss after training = 9671.699842277314
500038
True loss: 8994.140675857274
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4524.07 seconds
Loss after training = 9009.298454480513
500039
True loss: 9181.261053122891
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/sss
Total time: 4412.43 seconds
Loss after training = 9339.891921838413
500040
True loss: 9195.84463358019
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.41 iterations/s/ss
Total time: 4396.36 seconds
Loss after training = 9354.595436598465
500041
True loss: 9094.073324760691
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4479.63 seconds
Loss after training = 11350.906809545144
500042
True loss: 8968.64762369431
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/ssss
Total time: 4435.66 seconds
Loss after training = 9450.482194902519
500043
True loss: 8756.920705409575
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4453.11 seconds
Loss after training = 8868.978881904673
500044
True loss: 8861.757988295443
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4458.7 seconds
Loss after training = 8857.344215201065
500045
True loss: 9276.199116036754
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4510.77 seconds
Loss after training = 11048.31090470735
500046
True loss: 9268.182458532994
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.41 iterations/s/ss
Total time: 4400.36 seconds
Loss after training = 9667.931330083775
500047
True loss: 9179.198532278337
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4448.91 seconds
Loss after training = 9181.444301379293
500048
True loss: 8771.297817138318
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4460.41 seconds
Loss after training = 8947.055453494013
500049
True loss: 9093.058722201426
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 4545.57 seconds
Loss after training = 9131.341909512394
500050
True loss: 9194.406531241297
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.42 iterations/s/ss
Total time: 4387.31 seconds
Loss after training = 9193.56086670019
500051
True loss: 9317.553024894012
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4485.17 seconds
Loss after training = 9316.297308759938
500052
True loss: 9038.74295138601
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/sss
Total time: 4407.67 seconds
Loss after training = 9180.977458736643
500053
True loss: 8725.762061552177
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4524.81 seconds
Loss after training = 8719.101614160885
500054
True loss: 8768.921473626933
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/ss
Total time: 4411.7 seconds
Loss after training = 9593.060258899846
500055
True loss: 8943.288557878832
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.41 iterations/s/ss
Total time: 4396.48 seconds
Loss after training = 9206.789967250903
500056
True loss: 8699.571336736835
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4481.04 seconds
Loss after training = 8845.74153692033
500057
True loss: 8910.464281722576
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4467.23 seconds
Loss after training = 9117.43147069948
500058
True loss: 8889.5839261589
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 4426.37 seconds
Loss after training = 9047.50475477082
500059
True loss: 8890.415049903662
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 4430.98 seconds
Loss after training = 8992.27118352059
500060
True loss: 8879.240623367852
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 4423.92 seconds
Loss after training = 8955.375922485839
500061
True loss: 9016.75194143034
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4468.4 seconds
Loss after training = 9290.674375883067
500062
True loss: 8895.381827430443
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/ssss
Total time: 4435.64 seconds
Loss after training = 8959.513162380741
500063
True loss: 8771.234040427782
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4482.87 seconds
Loss after training = 8808.655510173849
500064
True loss: 9020.066502992633
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/sss
Total time: 4408.22 seconds
Loss after training = 9127.486973109555
500065
True loss: 9007.164871975643
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 4424.51 seconds
Loss after training = 9247.969467737714
500066
True loss: 9144.274329207628
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4481.05 seconds
Loss after training = 9141.202587774133
500067
True loss: 9157.172502552645
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/ss
Total time: 4545.33 seconds
Loss after training = 9321.326904470525
500068
True loss: 9183.304078171046
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4460.5 seconds
Loss after training = 9542.76691447408
500069
True loss: 8926.320059377646
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.42 iterations/s/ss
Total time: 4387.2 seconds
Loss after training = 9187.529244373283
500070
True loss: 8874.911351038036
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4464.82 seconds
Loss after training = 9773.131984602023
500071
True loss: 8894.32487210288
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4490.47 seconds
Loss after training = 9104.908214950423
500072
True loss: 8800.859327424574
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4524.53 seconds
Loss after training = 8814.419154483134
500073
True loss: 9121.132404671502
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4478.96 seconds
Loss after training = 9234.160898188273
500074
True loss: 9217.722853730676
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4453.46 seconds
Loss after training = 9247.685864902818
500075
True loss: 9365.032862240394
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4472.01 seconds
Loss after training = 9587.17687573877
500076
True loss: 9222.926522929407
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 4618.53 seconds
Loss after training = 9234.111637466216
500077
True loss: 8949.019438960839
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 4620.05 seconds
Loss after training = 8977.334163540894
500078
True loss: 8999.285459363864
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/ss/s
Total time: 4432.35 seconds
Loss after training = 9447.029149984573
500079
True loss: 8995.443925901376
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4465.53 seconds
Loss after training = 9044.543683345893
500080
True loss: 8960.168494647909
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4492.93 seconds
Loss after training = 8961.429193925733
500081
True loss: 8997.327294443556
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4505.71 seconds
Loss after training = 8994.537008539854
500082
True loss: 9002.715124841432
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4476.63 seconds
Loss after training = 9032.81575767555
500083
True loss: 9276.607243761116
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4459.78 seconds
Loss after training = 10176.848706390607
500084
True loss: 8938.38316811365
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/ss
Total time: 4418.1 seconds
Loss after training = 9539.340829343142
500085
True loss: 8922.276451102256
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4464.0 seconds
Loss after training = 8955.230993748777
500086
True loss: 8937.889392759482
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.42 iterations/s/ss
Total time: 4380.97 seconds
Loss after training = 9471.738246428671
500087
True loss: 8946.388154667044
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.42 iterations/s/ss
Total time: 4387.53 seconds
Loss after training = 8953.28999316842
500088
True loss: 9258.577865141946
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4456.27 seconds
Loss after training = 10155.098246676345
500089
True loss: 9084.551791314172
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4531.88 seconds
Loss after training = 9159.272924407136"""

#BW:
BWNormal = """500100
True loss: 8892.181903227243
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.64 iterations/ss/ss
Total time: 9146.84 seconds
Loss after training = 9316.190616114734
500101
True loss: 8971.57438670103
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.63 iterations/ss/ss
Total time: 9222.86 seconds
Loss after training = 9213.217875232613
500102
True loss: 9041.029806971968
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.61 iterations/ss/ss
Total time: 9300.25 seconds
Loss after training = 9088.067721628857
500103
True loss: 9064.07002604847
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.27 iterations/ss/ss
Total time: 11855.72 seconds
Loss after training = 10189.283208268746
500104
True loss: 8925.98322640005
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.41 iterations/ss/ss
Total time: 10630.91 seconds
Loss after training = 9008.430651425444
500105
True loss: 9008.204117721825
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.49 iterations/ss/ss
Total time: 10093.42 seconds
Loss after training = 9596.533090192941
500106
True loss: 9057.20360462636
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.76 iterations/ss/ss
Total time: 8526.38 seconds
Loss after training = 9195.383761422981
500107
True loss: 8966.149017025802
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.64 iterations/ss/s
Total time: 9139.68 seconds
Loss after training = 8969.279167579882
500108
True loss: 8667.644597420287
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.43 iterations/ss/ss
Total time: 10480.02 seconds
Loss after training = 10485.50816621534
500109
True loss: 9267.978932607479
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.46 iterations/ss/ss
Total time: 10240.88 seconds
Loss after training = 9911.169381070697
500110
True loss: 8780.345794696823
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.35 iterations/s/sss
Total time: 11140.68 seconds
Loss after training = 8805.350629472527
500111
True loss: 8860.291421299977
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.62 iterations/ss/ss
Total time: 9260.98 seconds
Loss after training = 9386.630559518771
500112
True loss: 9066.53394098549
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.54 iterations/ss/s
Total time: 9711.75 seconds
Loss after training = 9455.19893200432
500113
True loss: 9076.384554620005
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.68 iterations/ss/ss
Total time: 8903.17 seconds
Loss after training = 9648.231343782196
500114
True loss: 9041.432029026557
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.27 iterations/ss/ss
Total time: 11795.06 seconds
Loss after training = 9038.165662008545
500115
True loss: 9003.44889601393
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.26 iterations/ss/ss
Total time: 11925.52 seconds
Loss after training = 9166.01762791455
500116
True loss: 8893.000647040168
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.4 iterations/sns/ss
Total time: 10684.18 seconds
Loss after training = 8932.152232828486
500117
True loss: 8836.860073870705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.29 iterations/s/sss
Total time: 11649.53 seconds
Loss after training = 8833.738384418406
500118
True loss: 8904.322342029063
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.96 iterations/ss/ss
Total time: 7646.93 seconds
Loss after training = 9896.853308326148
500119
True loss: 9058.411554731392
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/ss/ss
Total time: 4418.28 seconds
Loss after training = 9067.592482002023
500120
True loss: 9067.554716971803
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/ssss
Total time: 4437.95 seconds
Loss after training = 9055.349687842348
500121
True loss: 8819.771687358221
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4472.33 seconds
Loss after training = 9425.844817217787
500122
True loss: 8983.98046257182
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.43 iterations/s/ss
Total time: 4377.3 seconds
Loss after training = 9062.41239452351
500123
True loss: 8899.6034744785
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/ssss
Total time: 4442.18 seconds
Loss after training = 8937.838686529
500124
True loss: 9111.15494943871
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4460.71 seconds
Loss after training = 9398.343294542177
500125
True loss: 9328.81899002641
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4503.27 seconds
Loss after training = 9688.130105520959
500126
True loss: 9103.145862425656
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.43 iterations/s/ss
Total time: 4372.29 seconds
Loss after training = 9896.280385620645
500127
True loss: 8858.884026384823
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 4431.57 seconds
Loss after training = 9636.851199689056
500128
True loss: 8851.153510060944
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 4428.35 seconds
Loss after training = 8850.882677995436
500129
True loss: 8715.133278564006
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.42 iterations/s/ss
Total time: 4387.04 seconds
Loss after training = 8707.322039751938
500130
True loss: 8944.714075510137
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.43 iterations/s/ss
Total time: 4376.33 seconds
Loss after training = 8955.436367605691
500131
True loss: 8844.493222441391
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4478.94 seconds
Loss after training = 8858.924922851897
500132
True loss: 9112.525443807657
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/ssss
Total time: 4446.65 seconds
Loss after training = 9113.857266941943
500133
True loss: 9031.449003738508
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/s/s
Total time: 4406.08 seconds
Loss after training = 9061.874354602702
500134
True loss: 9422.393405479275
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.41 iterations/s/ss
Total time: 4397.96 seconds
Loss after training = 10577.488911916811
500135
True loss: 9058.727723264514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/s/s
Total time: 4406.47 seconds
Loss after training = 9192.644030069347
500136
True loss: 9092.859423061413
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4460.59 seconds
Loss after training = 9098.498578906338
500137
True loss: 8849.278561261186
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4496.45 seconds
Loss after training = 8868.616718182631
500138
True loss: 9231.436162326218
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/ss/ss
Total time: 4406.35 seconds
Loss after training = 9245.144265682142
500139
True loss: 9129.785024945879
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/ss/s
Total time: 4435.51 seconds
Loss after training = 9296.710294281338
500140
True loss: 8950.667073432096
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4491.47 seconds
Loss after training = 9120.472060667054
500141
True loss: 9206.209531826948
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4450.3 seconds
Loss after training = 9210.398815172446
500142
True loss: 9079.141037894491
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/ss/s
Total time: 4432.8 seconds
Loss after training = 9258.618626555197
500143
True loss: 9068.950211038002
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/ssss
Total time: 4437.31 seconds
Loss after training = 10406.06105788031
500144
True loss: 8853.62678083281
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/ssss
Total time: 4444.04 seconds
Loss after training = 8858.611466736642
500145
True loss: 8758.35461937372
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.43 iterations/s/ss
Total time: 4371.04 seconds
Loss after training = 9311.362332000417
500146
True loss: 8968.28455542767
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/ssss
Total time: 4445.98 seconds
Loss after training = 10221.000761339852
500147
True loss: 8783.723632937523
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 4425.95 seconds
Loss after training = 9048.406928044753
500148
True loss: 8933.746708351413
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4452.36 seconds
Loss after training = 9432.19911181318
500149
True loss: 9117.17730346635
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 4433.51 seconds
Loss after training = 9382.557304126185
500150
True loss: 9073.598564318565
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.41 iterations/s/ss
Total time: 4396.49 seconds
Loss after training = 9161.761566120995
500151
True loss: 8885.205758459319
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4457.73 seconds
Loss after training = 9087.721423893981
500152
True loss: 8952.353372789774
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.43 iterations/s/ss
Total time: 4375.44 seconds
Loss after training = 9378.216782901889
500153
True loss: 8815.075346669577
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4458.69 seconds
Loss after training = 8806.316030101752
500154
True loss: 9223.626505991879
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/sss
Total time: 4411.45 seconds
Loss after training = 10009.999700704198
500155
True loss: 9054.226199851264
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/ss
Total time: 4417.74 seconds
Loss after training = 9060.053470367315
500156
True loss: 8859.881160979496
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4475.88 seconds
Loss after training = 8928.215666065347
500157
True loss: 9094.354700430522
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 4427.55 seconds
Loss after training = 9185.146958563299
500158
True loss: 8923.268051447674
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4446.12 seconds
Loss after training = 9082.389317789815
500159
True loss: 9159.123130760154
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.44 iterations/s/ss
Total time: 4358.26 seconds
Loss after training = 9147.906371968125
500160
True loss: 8738.857550127601
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.41 iterations/s/ss
Total time: 4397.95 seconds
Loss after training = 9351.585070203511
500161
True loss: 9050.50035912183
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4467.46 seconds
Loss after training = 9358.584433525082
500162
True loss: 9072.5240325338
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/ssss
Total time: 4441.95 seconds
Loss after training = 9074.825742751598
500163
True loss: 8896.014899958878
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/ssss
Total time: 4445.9 seconds
Loss after training = 9008.915294151497
500164
True loss: 9349.401645304357
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4467.17 seconds
Loss after training = 9761.067402619341
500165
True loss: 8975.54457042394
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4453.86 seconds
Loss after training = 9199.941599348807
500166
True loss: 8928.53138233524
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4501.05 seconds
Loss after training = 9818.874129654536
500167
True loss: 9063.907220489353
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4470.66 seconds
Loss after training = 9256.050384450507
500168
True loss: 9190.098500741784
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.42 iterations/s/ss
Total time: 4380.63 seconds
Loss after training = 9395.316611216682
500169
True loss: 8958.64191302387
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.42 iterations/s/ss
Total time: 4379.92 seconds
Loss after training = 9158.502594296759
500170
True loss: 8907.147129526598
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.41 iterations/s/ss
Total time: 4402.16 seconds
Loss after training = 9051.289088708003
500171
True loss: 9288.269348972555
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/sss
Total time: 4408.4 seconds
Loss after training = 9463.106922582956
500172
True loss: 9029.585231306639
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/sss
Total time: 4409.26 seconds
Loss after training = 10211.990670092004
500173
True loss: 8935.177227452526
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4522.2 seconds
Loss after training = 8972.332772768754
500174
True loss: 8861.91986684177
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.45 iterations/s/ss
Total time: 4351.84 seconds
Loss after training = 9727.661742510212
500175
True loss: 9011.389021843539
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 4425.52 seconds
Loss after training = 9089.609220159806
500176
True loss: 9182.282667182764
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/ss/ss
Total time: 4550.98 seconds
Loss after training = 9205.644430974167
500177
True loss: 8953.722820361469
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.11 iterations/ss/s
Total time: 4822.1 seconds
Loss after training = 8968.443370763174
500178
True loss: 8781.09048867889
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4484.65 seconds
Loss after training = 8778.632498002919
500179
True loss: 8996.816822800072
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 4428.51 seconds
Loss after training = 9001.240990839508
500180
True loss: 9340.90622704481
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4531.26 seconds
Loss after training = 10801.825713330147
500181
True loss: 8690.871067322365
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/sss
Total time: 4409.57 seconds
Loss after training = 8763.36094395491
500182
True loss: 9133.419327561885
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.41 iterations/s/ss
Total time: 4400.34 seconds
Loss after training = 10449.086794981526
500183
True loss: 8986.164272097092
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4479.38 seconds
Loss after training = 9077.515429543393
500184
True loss: 8682.769489100165
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4454.88 seconds
Loss after training = 8875.323046764313
500185
True loss: 8931.708421227228
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/ss
Total time: 4410.97 seconds
Loss after training = 8925.133381937658
500186
True loss: 8968.768473001468
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4457.26 seconds
Loss after training = 8964.053055462024
500187
True loss: 9025.372489413217
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4486.72 seconds
Loss after training = 9891.465028447583
500188
True loss: 8691.733615316925
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4477.16 seconds
Loss after training = 9014.370146378964
500189
True loss: 8805.27903565785
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4490.57 seconds
Loss after training = 8818.907902170708"""

#Zhao:
ZhaoNormal = """500200
True loss: 8919.511107100729
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.4 iterations/sns/ss
Total time: 10707.84 seconds
Loss after training = 8918.22776582082
500201
True loss: 8884.45177328381
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.18 iterations/ss/ss
Total time: 12705.64 seconds
Loss after training = 8941.577626202545
500202
True loss: 8722.665978554107
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.33 iterations/ss/ss
Total time: 11286.51 seconds
Loss after training = 10337.790169605934
500203
True loss: 9096.579415340817
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.12 iterations/ss/ss
Total time: 13404.59 seconds
Loss after training = 9505.810736002582
500204
True loss: 9284.517317380023
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.26 iterations/ss/ss
Total time: 11928.83 seconds
Loss after training = 9307.271552194421
500205
True loss: 8771.608570374236
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.64 iterations/ss/ss
Total time: 9140.88 seconds
Loss after training = 8881.455016572138
500206
True loss: 9281.350508765276
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.21 iterations/ss/ss
Total time: 12399.1 seconds
Loss after training = 10031.892157832954
500207
True loss: 8947.615633798865
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.34 iterations/ss/ss
Total time: 11225.92 seconds
Loss after training = 9091.892838122758
500208
True loss: 9107.20467597771
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.31 iterations/ss/ss
Total time: 11488.52 seconds
Loss after training = 9138.63621856495
500209
True loss: 8697.232164609553
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.26 iterations/ss/ss
Total time: 11926.22 seconds
Loss after training = 9035.632562469182
500210
True loss: 8984.949890370706
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.3 iterations/sns/ss
Total time: 11510.1 seconds
Loss after training = 8990.343408216388
500211
True loss: 9162.541760868327
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.54 iterations/ss/ss
Total time: 9765.94 seconds
Loss after training = 9596.50010191407
500212
True loss: 9093.926855921189
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.33 iterations/ss/ss
Total time: 11252.31 seconds
Loss after training = 9166.087681129591
500213
True loss: 8965.07191202816
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.65 iterations/ss/ss
Total time: 9080.28 seconds
Loss after training = 9110.54132825878
500214
True loss: 9097.548648530767
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.95 iterations/ss/s
Total time: 7674.91 seconds
Loss after training = 9924.78280083675
500215
True loss: 9039.589573476826
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.91 iterations/ss/ss
Total time: 7847.72 seconds
Loss after training = 9395.388184206056
500216
True loss: 8783.201418717495
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.22 iterations/s/sss
Total time: 12341.39 seconds
Loss after training = 8785.010771581403
500217
True loss: 8772.590651005568
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.28 iterations/ss/ss
Total time: 6565.2 seconds
Loss after training = 9365.89796405517
500218
True loss: 9119.903200563978
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4500.65 seconds
Loss after training = 9175.943417317825
500219
True loss: 9292.883393322056
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4513.16 seconds
Loss after training = 10254.05993535508
500220
True loss: 9004.040584656908
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4512.17 seconds
Loss after training = 9124.950330084921
500221
True loss: 8779.27319667532
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4501.77 seconds
Loss after training = 8892.176194336142
500222
True loss: 9053.982473549087
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4508.56 seconds
Loss after training = 9902.407896288878
500223
True loss: 9533.591406024294
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/ss/s
Total time: 4626.12 seconds
Loss after training = 9588.681090810627
500224
True loss: 9196.166071394478
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/ssss
Total time: 4441.68 seconds
Loss after training = 9191.11053762349
500225
True loss: 9006.073592087161
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4488.49 seconds
Loss after training = 9669.479828350313
500226
True loss: 8886.998599529663
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4506.82 seconds
Loss after training = 9326.903000361182
500227
True loss: 9091.457808436771
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4487.31 seconds
Loss after training = 9076.703710765129
500228
True loss: 9063.60343312376
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.41 iterations/s/ss
Total time: 4398.16 seconds
Loss after training = 9342.406126862315
500229
True loss: 9383.49351314638
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 4570.71 seconds
Loss after training = 10476.65013591018
500230
True loss: 8999.620844975869
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4518.2 seconds
Loss after training = 9427.007908327236
500231
True loss: 9044.500167701197
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4501.7 seconds
Loss after training = 9875.435065364909
500232
True loss: 9312.245398095176
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4457.5 seconds
Loss after training = 9865.409956742484
500233
True loss: 8895.317140227029
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4489.95 seconds
Loss after training = 8894.596247395068
500234
True loss: 9220.257605840288
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4496.24 seconds
Loss after training = 9861.8534833565
500235
True loss: 8969.81728572428
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/ss/ss
Total time: 4541.48 seconds
Loss after training = 9114.397012702024
500236
True loss: 9238.644264933342
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4535.83 seconds
Loss after training = 9418.037861492583
500237
True loss: 8793.98485280421
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/s/ss
Total time: 4591.09 seconds
Loss after training = 8791.443367936216
500238
True loss: 8858.38537376315
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4527.89 seconds
Loss after training = 10531.155195302152
500239
True loss: 9157.279975918924
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4515.51 seconds
Loss after training = 9232.618946603756
500240
True loss: 9031.289855055644
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4532.93 seconds
Loss after training = 10688.412408226826
500241
True loss: 8994.840904160645
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/s/ss
Total time: 4589.23 seconds
Loss after training = 9228.172845623383
500242
True loss: 8791.861022592853
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 4577.8 seconds
Loss after training = 9420.659908167407
500243
True loss: 9047.058816338511
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 4605.65 seconds
Loss after training = 9152.822509585934
500244
True loss: 9057.713632031448
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4459.77 seconds
Loss after training = 9111.12797959963
500245
True loss: 8795.250034744526
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4450.47 seconds
Loss after training = 8987.127206368375
500246
True loss: 8877.284187500287
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4508.9 seconds
Loss after training = 8916.475195472443
500247
True loss: 9106.774053507595
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4467.89 seconds
Loss after training = 9299.828686910592
500248
True loss: 8842.21345752241
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/ss/s
Total time: 4621.97 seconds
Loss after training = 8904.939428524514
500249
True loss: 8897.95126519101
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4507.84 seconds
Loss after training = 8910.77725626313
500250
True loss: 8951.282275660758
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4512.47 seconds
Loss after training = 9573.708530023632
500251
True loss: 8888.372372821486
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/ssss
Total time: 4557.74 seconds
Loss after training = 8916.518374217361
500252
True loss: 9056.47029220811
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4533.65 seconds
Loss after training = 9058.03201826685
500253
True loss: 8865.503384624371
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4522.88 seconds
Loss after training = 10112.40569468198
500254
True loss: 8833.251164182788
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4522.1 seconds
Loss after training = 8847.06297440793
500255
True loss: 9289.558729626033
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 4549.7 seconds
Loss after training = 10556.629990108619
500256
True loss: 8887.133690381186
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4454.77 seconds
Loss after training = 9428.742381101778
500257
True loss: 9255.582207626872
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4537.01 seconds
Loss after training = 9266.323977740141
500258
True loss: 8845.741056917446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4457.74 seconds
Loss after training = 9203.091682661186
500259
True loss: 8947.797406322805
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4478.31 seconds
Loss after training = 9230.915940874225
500260
True loss: 8958.255958435851
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4471.54 seconds
Loss after training = 9987.408017304246
500261
True loss: 9241.709956876146
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4468.92 seconds
Loss after training = 9298.237685176271
500262
True loss: 8967.225383017374
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/s/s
Total time: 4542.41 seconds
Loss after training = 8964.827212753666
500263
True loss: 9046.374756290486
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4532.73 seconds
Loss after training = 10788.343512020416
500264
True loss: 8857.211679655991
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 4598.31 seconds
Loss after training = 9022.686647191444
500265
True loss: 8738.204022411308
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4527.88 seconds
Loss after training = 9249.514873405753
500266
True loss: 9059.9182572902
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4528.99 seconds
Loss after training = 9125.161811505652
500267
True loss: 9091.217120772173
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 4547.96 seconds
Loss after training = 9148.505214209636
500268
True loss: 9188.133260332157
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4513.1 seconds
Loss after training = 9227.950846856851
500269
True loss: 8982.599973570437
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4494.43 seconds
Loss after training = 9036.127432444186
500270
True loss: 9040.397108011437
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4520.67 seconds
Loss after training = 9081.430374482436
500271
True loss: 8747.906999417868
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4484.47 seconds
Loss after training = 8805.267851174589
500272
True loss: 9099.457191940463
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4502.68 seconds
Loss after training = 9148.13408041892
500273
True loss: 9261.181961367334
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 4516.83 seconds
Loss after training = 9401.283016951771
500274
True loss: 8759.168626088353
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.18 iterations/s/ss
Total time: 4712.86 seconds
Loss after training = 10196.271812810062
500275
True loss: 9048.505951743186
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 4647.08 seconds
Loss after training = 9040.154309486004
500276
True loss: 8999.594012274198
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4493.85 seconds
Loss after training = 9016.258288156518
500277
True loss: 8731.943190935108
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4536.25 seconds
Loss after training = 9224.464595209218
500278
True loss: 9175.718269928146
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 4504.55 seconds
Loss after training = 9737.169617847683
500279
True loss: 8968.626823944538
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 4457.21 seconds
Loss after training = 8974.772579409224
500280
True loss: 9038.454794365724
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/s/ss
Total time: 4588.65 seconds
Loss after training = 9225.83954407276
500281
True loss: 9225.23614155775
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 4463.87 seconds
Loss after training = 9464.21218408721
500282
True loss: 9049.506048679688
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4475.68 seconds
Loss after training = 9383.062652318149
500283
True loss: 9254.227958100706
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 4525.78 seconds
Loss after training = 9584.445519394594
500284
True loss: 8842.396890710887
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/s/ss
Total time: 4591.46 seconds
Loss after training = 8846.91648703763
500285
True loss: 9287.441329534895
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/ssss
Total time: 4439.23 seconds
Loss after training = 9739.25718600315
500286
True loss: 8845.20755799684
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 4488.73 seconds
Loss after training = 9070.597841561314
500287
True loss: 8971.768066174875
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 4473.46 seconds
Loss after training = 8997.61534609557
500288
True loss: 9151.716206870533
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/ss/ss
Total time: 4540.51 seconds
Loss after training = 9150.76537842938
500289
True loss: 8721.351871398465
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/ssss
Total time: 4435.83 seconds
Loss after training = 8879.80628962953"""


truelosses = []
recontrueDMguess = []
reconNormalguess = []


parsedPlumTrue = PlumTrue.split("\n")
parsedBWTrue = BWTrue.split("\n")
parsedZhaoTrue = ZhaoTrue.split("\n")
parsedPlumNormal = PlumNormal.split("\n")
parsedBWNormal = BWNormal.split("\n")
parsedZhaoNormal = ZhaoNormal.split("\n")
for i in range(len(parsedPlumTrue)):
    # 1 -> True loss
    # 4 -> Loss after training
    if i % 5 == 1:
        # print(parsedPlumTrue[i])
        # print(parsedPlumTrue[i].split(' ')[-1])
        truelosses.append(float(parsedPlumTrue[i].split(' ')[-1]))
        # break
    if i % 5 == 4:
        # print(parsedPlumTrue[i])
        # print(parsedPlumTrue[i].split(' ')[-1])
        recontrueDMguess.append(float(parsedPlumTrue[i].split(' ')[-1]))
        # print(parsedPlumNormal[i])
        reconNormalguess.append(float(parsedPlumNormal[i].split(' ')[-1]))
        
truelosses = truelosses + [9123,9215,8940,8974,8984,8990,8898,8898,8797,8634]
recontrueDMguess = recontrueDMguess + [9211,9407,9735,9183,9202,9234,8910,9706,8796,8867]
reconNormalguess = reconNormalguess + [9193,9442,10028,9168,9168,9229,9767,9484,9178,9472]

for i in range(len(parsedBWTrue)):
    # 1 -> True loss
    # 4 -> Loss after training
    if i % 5 == 1:
        # print(parsedBWTrue[i])
        # print(parsedBWTrue[i].split(' ')[-1])
        truelosses.append(float(parsedBWTrue[i].split(' ')[-1]))
    if i % 5 == 4:
        # print(parsedBWTrue[i])
        # print(parsedBWTrue[i].split(' ')[-1])
        recontrueDMguess.append(float(parsedBWTrue[i].split(' ')[-1]))
        # print(parsedPlumNormal[i])
        reconNormalguess.append(float(parsedBWNormal[i].split(' ')[-1]))

truelosses = truelosses + [8967,9142,9267,9032,9221,8910,8882,9094,8577,9044]
recontrueDMguess = recontrueDMguess + [9124,9780,9449,9183,9672,9154,9606,9106,9872,9293]
reconNormalguess = reconNormalguess + [9266,9245,9255,9203,9097, 9202,8917,9799,8800,8894]

for i in range(len(parsedZhaoTrue)):
    # 1 -> True loss
    # 4 -> Loss after training
    if i % 5 == 1:
        # print(parsedZhaoTrue[i])
        # print(parsedZhaoTrue[i].split(' ')[-1])
        truelosses.append(float(parsedZhaoTrue[i].split(' ')[-1]))
    if i % 5 == 4:
        # print(parsedZhaoTrue[i])
        # print(parsedZhaoTrue[i].split(' ')[-1])
        recontrueDMguess.append(float(parsedZhaoTrue[i].split(' ')[-1]))
        # print(parsedPlumNormal[i])
        reconNormalguess.append(float(parsedZhaoNormal[i].split(' ')[-1]))
        
truelosses = truelosses + [9247,9198,9171,9186,9081,8977,9021,8919,9241,9194]
recontrueDMguess = recontrueDMguess + [9272,9251,9262,9186,9117,9121,9329,9663,9629,9271]
reconNormalguess = reconNormalguess+ [9270,9610,9106,9892,9332,9174,9286,9837,9518,9264]


# print(truelosses)
        

diffsTrueguess = []
diffsNormalguess = []
for i in range(2*100):
    diffsTrueguess.append(recontrueDMguess[i]-truelosses[i])
    diffsNormalguess.append(reconNormalguess[i]-truelosses[i])
    
print(len(truelosses))
zhaoDiffsTrue = []
zhaoDiffsNormal = []
for i in range(2*100,len(truelosses)):
    zhaoDiffsTrue.append(recontrueDMguess[i]-truelosses[i])
    zhaoDiffsNormal.append(reconNormalguess[i]-truelosses[i])
print(len(zhaoDiffsNormal))

# plt.figure()
# plt.boxplot(1,diffsNormalguess)

# plt.boxplot(2,diffsTrueguess)

import seaborn as sns
plt.figure()
sns.boxplot(data=[diffsNormalguess, diffsTrueguess]).set(xlabel="Left: IG3 starting from true DM distribution. Right: IG3.",title='{Reconstructed loss} - {True loss} for Plummer and BW')
plt.ylim(-180,2100)
plt.show()

plt.figure()
sns.boxplot(data=[zhaoDiffsNormal, zhaoDiffsTrue]).set(xlabel="Left: IG3 starting from true DM distribution. Right: IG3.",title='{Reconstructed loss} - {True loss} for Zhao')
plt.ylim(-180,2100)
plt.show()

import pandas as pd
df = pd.DataFrame(diffsNormalguess)
# df.boxplot()
# print(df.describe())
df2 = pd.DataFrame(diffsTrueguess)
# df.boxplot()
# print(df2.describe())
# df = pd.concat([df,df2],axis=1)
# print(df.describe())


df3 = pd.DataFrame(zhaoDiffsNormal)
# df.boxplot()
# print(df.describe())
df4 = pd.DataFrame(zhaoDiffsTrue)
# df.boxplot()
# print(df2.describe())
df = pd.concat([df,df2,df3,df4],axis=1)

print(df.describe())


