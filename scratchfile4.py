# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:10:55 2023

@author: Thibault
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:30:38 2023

@author: Thibault
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:44:05 2023

@author: Thibault
"""

import numpy as np
from matplotlib.pylab import plt

#Plum, BW, Zhao

#Plummer
PlumTrue = """800000
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3068.23 seconds
Loss after training = 9628.988382353666
800001
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3052.99 seconds
Loss after training = 8837.88351451101
800002
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3077.46 seconds
Loss after training = 8886.304694482711
800003
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3065.99 seconds
Loss after training = 8918.14803117189
800004
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3047.55 seconds
Loss after training = 8835.755178994663
800005
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.17 iterations/s/ss
Total time: 3155.18 seconds
Loss after training = 9003.511582841367
800006
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3018.38 seconds
Loss after training = 8871.31313947001
800007
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3089.54 seconds
Loss after training = 9267.39151515171
800008
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3067.73 seconds
Loss after training = 9931.43738402878
800009
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.21 iterations/ss/s
Total time: 3117.58 seconds
Loss after training = 8842.355424114183
800010
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3081.1 seconds
Loss after training = 10338.693680403896
800011
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3090.42 seconds
Loss after training = 9045.705464280532
800012
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3074.15 seconds
Loss after training = 8838.359305965134
800013
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3041.54 seconds
Loss after training = 11253.516783763227
800014
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/s/s
Total time: 3028.44 seconds
Loss after training = 9901.02678839187
800015
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3053.36 seconds
Loss after training = 8935.4682089202
800016
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.21 iterations/ss/s
Total time: 3112.77 seconds
Loss after training = 9010.060123316252
800017
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3076.14 seconds
Loss after training = 8842.55539225492
800018
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3090.33 seconds
Loss after training = 8965.751711424728
800019
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.19 iterations/s/ss
Total time: 3132.64 seconds
Loss after training = 9622.78121553973
800020
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3043.26 seconds
Loss after training = 9034.079348268166
800021
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3092.58 seconds
Loss after training = 9084.24281761559
800022
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.22 iterations/s/ss
Total time: 3103.17 seconds
Loss after training = 9608.992379993444
800023
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3087.82 seconds
Loss after training = 9455.926807328966
800024
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.19 iterations/sss
Total time: 3130.53 seconds
Loss after training = 10261.350492676736
800025
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/ssss
Total time: 3063.66 seconds
Loss after training = 10926.867589673666
800026
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 3033.06 seconds
Loss after training = 8854.325935878358
800027
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3068.54 seconds
Loss after training = 8924.414218407093
800028
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.21 iterations/ss/s
Total time: 3116.05 seconds
Loss after training = 9057.880926495001
800029
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3095.43 seconds
Loss after training = 8889.460991085172
800030
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3084.32 seconds
Loss after training = 8858.06467831057
800031
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3045.23 seconds
Loss after training = 8930.200900787464
800032
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3059.04 seconds
Loss after training = 8883.981020318306
800033
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3070.53 seconds
Loss after training = 8843.037409590534
800034
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.22 iterations/s/ss
Total time: 3103.63 seconds
Loss after training = 8837.112680388815
800035
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3017.36 seconds
Loss after training = 8964.780031669125
800036
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2982.89 seconds
Loss after training = 9017.674879331365
800037
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3050.76 seconds
Loss after training = 9844.909758780184
800038
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3015.11 seconds
Loss after training = 8833.336468967998
800039
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2996.41 seconds
Loss after training = 9753.987546914503
800040
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2994.71 seconds
Loss after training = 9184.981908435606
800041
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3046.41 seconds
Loss after training = 9453.490140584992
800042
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3008.2 seconds
Loss after training = 9238.266079481471
800043
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/ssss
Total time: 3035.51 seconds
Loss after training = 8857.726478707167
800044
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3072.17 seconds
Loss after training = 8936.533207358718
800045
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2959.85 seconds
Loss after training = 8927.965279444099
800046
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3021.05 seconds
Loss after training = 8986.807548755149
800047
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3001.37 seconds
Loss after training = 8875.935481634246
800048
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 2952.65 seconds
Loss after training = 9219.523392096293
800049
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/ss/ss
Total time: 3027.92 seconds
Loss after training = 9079.620769145333
800050
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3043.02 seconds
Loss after training = 9067.609039884544
800051
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3044.38 seconds
Loss after training = 8846.13251388933
800052
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3038.21 seconds
Loss after training = 9342.853559731971
800053
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3010.57 seconds
Loss after training = 9023.687484277942
800054
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3000.4 seconds
Loss after training = 8879.542467556717
800055
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3015.39 seconds
Loss after training = 10031.094665373712
800056
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2974.22 seconds
Loss after training = 9023.219502581276
800057
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3009.56 seconds
Loss after training = 10460.901753252541
800058
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2972.69 seconds
Loss after training = 8861.80322290897
800059
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2998.0 seconds
Loss after training = 8978.15593737529
800060
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3019.82 seconds
Loss after training = 8850.510336234533
800061
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3047.56 seconds
Loss after training = 9147.453798121709
800062
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3023.56 seconds
Loss after training = 10022.394867727802
800063
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2992.05 seconds
Loss after training = 8875.36095656088
800064
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2995.59 seconds
Loss after training = 9207.39716191801
800065
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/ssss
Total time: 2998.96 seconds
Loss after training = 9128.493776543031
800066
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3007.78 seconds
Loss after training = 9179.412297772386
800067
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2983.28 seconds
Loss after training = 8850.92828805537
800068
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2994.68 seconds
Loss after training = 9018.226375132426
800069
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2957.59 seconds
Loss after training = 8930.996375585122
800070
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2981.11 seconds
Loss after training = 9093.822245247417
800071
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.41 iterations/s/ss
Total time: 2935.56 seconds
Loss after training = 8833.71349537332
800072
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2978.9 seconds
Loss after training = 9107.541475426557
800073
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/ss/ss
Total time: 3026.93 seconds
Loss after training = 9042.558884630527
800074
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/s
Total time: 2948.67 seconds
Loss after training = 8839.8065169703
800075
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3005.42 seconds
Loss after training = 8957.822748313985
800076
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3015.64 seconds
Loss after training = 9194.62525559062
800077
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3004.95 seconds
Loss after training = 9262.056671138494
800078
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2979.88 seconds
Loss after training = 9265.912123911248
800079
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3007.52 seconds
Loss after training = 10639.558325108559
800080
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2996.32 seconds
Loss after training = 8826.100073534859
800081
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3046.77 seconds
Loss after training = 9343.752349588112
800082
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3020.27 seconds
Loss after training = 9139.853614537036
800083
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 2949.92 seconds
Loss after training = 9929.872838575813
800084
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3011.9 seconds
Loss after training = 8831.37583387717
800085
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3020.87 seconds
Loss after training = 8859.487506321411
800086
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.38 iterations/ss/s
Total time: 4201.24 seconds
Loss after training = 9131.393659611902
800087
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.51 iterations/ss/ss
Total time: 6614.51 seconds
Loss after training = 10123.990348912715
800088
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.2 iterations/ss/ss
Total time: 3120.34 seconds
Loss after training = 8853.243636342479
800089
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3021.15 seconds
Loss after training = 9693.858156337921
800090
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2987.54 seconds
Loss after training = 8977.933972058696
800091
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3017.91 seconds
Loss after training = 8863.88847506622
800092
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3008.97 seconds
Loss after training = 8893.348620728098
800093
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.67 iterations/s/ss
Total time: 3746.8 seconds
Loss after training = 8833.430334387087
800094
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.57 iterations/s/ss
Total time: 3890.04 seconds
Loss after training = 9110.47061048377
800095
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.68 iterations/s/ss
Total time: 3733.8 seconds
Loss after training = 9155.907458765827
800096
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.58 iterations/s/ss
Total time: 3873.26 seconds
Loss after training = 8839.987272950713
800097
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.51 iterations/s/ss
Total time: 3988.91 seconds
Loss after training = 10546.022691481387
800098
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.41 iterations/s/sss
Total time: 7073.93 seconds
Loss after training = 8938.486581193998
800099
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.48 iterations/ss/s
Total time: 6778.09 seconds
Loss after training = 8981.653451967612
"""
#BW:
    
BWTrue = """800000
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3005.76 seconds
Loss after training = 9685.016013833318
800001
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3024.16 seconds
Loss after training = 8839.085608868183
800002
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3015.42 seconds
Loss after training = 8908.662882308394
800003
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3011.72 seconds
Loss after training = 8946.33386441695
800004
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3038.51 seconds
Loss after training = 8835.787808611147
800005
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3069.99 seconds
Loss after training = 8967.84399270763
800006
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/ss
Total time: 3034.6 seconds
Loss after training = 8850.49596764037
800007
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3025.44 seconds
Loss after training = 9286.149981868699
800008
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3015.72 seconds
Loss after training = 9859.306558986427
800009
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3005.59 seconds
Loss after training = 8835.293974449747
800010
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2992.69 seconds
Loss after training = 10314.561244427547
800011
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2981.33 seconds
Loss after training = 9083.447371953594
800012
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3075.8 seconds
Loss after training = 8827.108386087668
800013
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3021.52 seconds
Loss after training = 11552.9932215991
800014
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 3033.32 seconds
Loss after training = 9851.818929623343
800015
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3042.68 seconds
Loss after training = 8903.399891841891
800016
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/ss/s
Total time: 3025.61 seconds
Loss after training = 9015.373943351653
800017
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3054.98 seconds
Loss after training = 8844.83482473785
800018
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3077.82 seconds
Loss after training = 8937.168767623374
800019
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3017.72 seconds
Loss after training = 9604.904345594532
800020
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3053.42 seconds
Loss after training = 8994.353771692464
800021
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3082.85 seconds
Loss after training = 9045.982811693275
800022
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2986.77 seconds
Loss after training = 9567.18864687783
800023
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3045.07 seconds
Loss after training = 9433.660916375547
800024
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3024.31 seconds
Loss after training = 10406.396335359475
800025
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3052.6 seconds
Loss after training = 10905.458737361672
800026
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3021.0 seconds
Loss after training = 8846.35147872137
800027
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3046.01 seconds
Loss after training = 8946.05641748162
800028
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3007.73 seconds
Loss after training = 9071.160262029785
800029
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3073.04 seconds
Loss after training = 8862.816772396001
800030
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3079.08 seconds
Loss after training = 8848.255874032171
800031
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3036.34 seconds
Loss after training = 8920.563007594332
800032
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3038.83 seconds
Loss after training = 8906.903473721017
800033
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3001.56 seconds
Loss after training = 8836.285527641823
800034
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3006.33 seconds
Loss after training = 8825.725213274194
800035
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2977.25 seconds
Loss after training = 8971.341677645367
800036
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2976.46 seconds
Loss after training = 8978.710037845985
800037
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2997.74 seconds
Loss after training = 9815.963981812773
800038
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/ss
Total time: 2944.53 seconds
Loss after training = 8823.761291547798
800039
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2976.13 seconds
Loss after training = 9697.333655943597
800040
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2973.99 seconds
Loss after training = 9137.687687024496
800041
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.44 iterations/s/ss
Total time: 2911.15 seconds
Loss after training = 9439.273617835746
800042
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2980.32 seconds
Loss after training = 9250.28049900806
800043
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2957.75 seconds
Loss after training = 8857.03141202168
800044
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.41 iterations/s/ss
Total time: 2929.84 seconds
Loss after training = 8939.975742819808
800045
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/s
Total time: 2947.06 seconds
Loss after training = 8906.999693219372
800046
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 2952.44 seconds
Loss after training = 8971.966589900667
800047
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2956.66 seconds
Loss after training = 8865.112374945506
800048
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2965.34 seconds
Loss after training = 9175.172589871898
800049
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3042.51 seconds
Loss after training = 9059.684584490784
800050
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 2949.17 seconds
Loss after training = 9043.703744355387
800051
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2971.98 seconds
Loss after training = 8840.019782924603
800052
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.42 iterations/s/ss
Total time: 2926.13 seconds
Loss after training = 9363.03593087249
800053
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/sss
Total time: 2944.87 seconds
Loss after training = 9035.173453427133
800054
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.41 iterations/s/ss
Total time: 2930.84 seconds
Loss after training = 8879.290661394452
800055
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 2951.65 seconds
Loss after training = 10071.262876377255
800056
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2965.43 seconds
Loss after training = 8983.672932299654
800057
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 2998.88 seconds
Loss after training = 10523.419063312413
800058
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3006.93 seconds
Loss after training = 8861.299525660526
800059
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2964.97 seconds
Loss after training = 8983.641271599594
800060
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2968.31 seconds
Loss after training = 8849.83494328092
800061
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3004.48 seconds
Loss after training = 9135.104799012952
800062
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3000.44 seconds
Loss after training = 9985.675270168951
800063
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3001.3 seconds
Loss after training = 8852.750151443975
800064
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2960.54 seconds
Loss after training = 9224.710349085728
800065
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.43 iterations/s/ss
Total time: 2917.9 seconds
Loss after training = 9145.181559851997
800066
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2975.77 seconds
Loss after training = 9132.950060389465
800067
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2968.49 seconds
Loss after training = 8838.55654333158
800068
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3002.52 seconds
Loss after training = 8996.821257490952
800069
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2964.82 seconds
Loss after training = 8934.856586146754
800070
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2978.89 seconds
Loss after training = 9051.004427870745
800071
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 2950.63 seconds
Loss after training = 8833.563723052366
800072
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2966.85 seconds
Loss after training = 9091.14082999627
800073
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2968.95 seconds
Loss after training = 9054.044732318951
800074
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2972.69 seconds
Loss after training = 8839.746310826804
800075
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/ss
Total time: 2942.89 seconds
Loss after training = 8961.47075275043
800076
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2974.1 seconds
Loss after training = 9149.906977152177
800077
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 2952.59 seconds
Loss after training = 9269.702175315973
800078
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3015.33 seconds
Loss after training = 9288.902434495349
800079
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2982.31 seconds
Loss after training = 10595.675973210064
800080
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3011.89 seconds
Loss after training = 8822.913714878156
800081
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2955.83 seconds
Loss after training = 9288.641682298583
800082
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2986.99 seconds
Loss after training = 9154.951478389243
800083
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2958.88 seconds
Loss after training = 9855.38217633513
800084
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3000.19 seconds
Loss after training = 8828.691239461856
800085
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3044.0 seconds
Loss after training = 8838.757297194485
800086
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2964.3 seconds
Loss after training = 9123.902226832104
800087
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.72 iterations/ss/s
Total time: 3672.33 seconds
Loss after training = 10042.901128315594
800088
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.57 iterations/s/ss
Total time: 6350.11 seconds
Loss after training = 8852.383341420811
800089
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.66 iterations/ss/s
Total time: 3766.2 seconds
Loss after training = 9671.299909865746
800090
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3055.54 seconds
Loss after training = 8970.939161259992
800091
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3100.66 seconds
Loss after training = 8854.538980465466
800092
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3000.59 seconds
Loss after training = 8913.244620589385
800093
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2970.78 seconds
Loss after training = 8824.820394684197
800094
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.84 iterations/s/ss
Total time: 3521.03 seconds
Loss after training = 9122.653528810226
800095
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.52 iterations/s/ss
Total time: 3965.17 seconds
Loss after training = 9120.61928675027
800096
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.65 iterations/s/ss
Total time: 3777.29 seconds
Loss after training = 8839.92452560295
800097
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.49 iterations/s/ss
Total time: 4009.42 seconds
Loss after training = 10497.925625033065
800098
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.69 iterations/s/ss
Total time: 3719.02 seconds
Loss after training = 8918.803702199972
800099
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.36 iterations/ss/s
Total time: 7329.81 seconds
Loss after training = 8963.751592300905
"""

#Zhao:
ZhaoTrue = """800000
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3047.29 seconds
Loss after training = 9578.661161635802
800001
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3013.03 seconds
Loss after training = 8836.569629381302
800002
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3062.84 seconds
Loss after training = 8878.296964742047
800003
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3043.52 seconds
Loss after training = 8906.371694843854
800004
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3094.78 seconds
Loss after training = 8835.53889091191
800005
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3036.41 seconds
Loss after training = 9015.493725371341
800006
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 3029.54 seconds
Loss after training = 8876.722620021406
800007
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.21 iterations/s/ss
Total time: 3111.19 seconds
Loss after training = 9254.175021466546
800008
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3078.17 seconds
Loss after training = 9960.720827780462
800009
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3047.96 seconds
Loss after training = 8852.225259427616
800010
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3097.7 seconds
Loss after training = 10342.96445042544
800011
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3093.47 seconds
Loss after training = 9022.046261986085
800012
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3072.2 seconds
Loss after training = 8842.707609972409
800013
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3059.41 seconds
Loss after training = 11417.548140192283
800014
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3085.33 seconds
Loss after training = 9986.349073357702
800015
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3095.81 seconds
Loss after training = 8945.180686190319
800016
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3080.83 seconds
Loss after training = 9002.710913743142
800017
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3083.13 seconds
Loss after training = 8840.762872865334
800018
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3085.51 seconds
Loss after training = 8971.915988803015
800019
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3044.2 seconds
Loss after training = 9618.732082980508
800020
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3090.18 seconds
Loss after training = 9047.506263430734
800021
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3071.29 seconds
Loss after training = 9094.528696841657
800022
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3062.65 seconds
Loss after training = 9627.214127548823
800023
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3089.86 seconds
Loss after training = 9442.961450077195
800024
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3070.63 seconds
Loss after training = 10343.742881413185
800025
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/s/ss
Total time: 3053.97 seconds
Loss after training = 10913.224060248247
800026
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3091.16 seconds
Loss after training = 8869.0223194189
800027
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/ssss
Total time: 3063.26 seconds
Loss after training = 8909.706259816709
800028
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.22 iterations/s/ss
Total time: 3104.9 seconds
Loss after training = 9049.169889109637
800029
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.15 iterations/ss/s
Total time: 3176.36 seconds
Loss after training = 8898.33098785352
800030
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.21 iterations/ss/s
Total time: 3119.28 seconds
Loss after training = 8860.260012518262
800031
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 3030.46 seconds
Loss after training = 8954.452538791858
800032
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3068.33 seconds
Loss after training = 8875.739443679873
800033
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.2 iterations/s/s/s
Total time: 3120.98 seconds
Loss after training = 8853.156327085933
800034
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/s/ss
Total time: 3060.88 seconds
Loss after training = 8839.767905605951
800035
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2957.38 seconds
Loss after training = 8958.886428198713
800036
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 2950.18 seconds
Loss after training = 9030.329033372063
800037
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2970.17 seconds
Loss after training = 9865.843091490677
800038
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.43 iterations/s/ss
Total time: 2917.52 seconds
Loss after training = 8836.159646146587
800039
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2964.78 seconds
Loss after training = 9822.524490850787
800040
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2986.86 seconds
Loss after training = 9201.342362556321
800041
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2959.96 seconds
Loss after training = 9423.530671014232
800042
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2957.98 seconds
Loss after training = 9225.579646337854
800043
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/sss
Total time: 2942.99 seconds
Loss after training = 8855.263707385984
800044
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2958.27 seconds
Loss after training = 8931.491144451089
800045
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.41 iterations/s/ss
Total time: 2935.03 seconds
Loss after training = 8929.80681199412
800046
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3025.27 seconds
Loss after training = 9016.238041078246
800047
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/sss
Total time: 2944.53 seconds
Loss after training = 8893.718300529796
800048
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 2953.07 seconds
Loss after training = 9240.023087207763
800049
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2956.57 seconds
Loss after training = 9118.216713523252
800050
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 3033.31 seconds
Loss after training = 9102.4463181195
800051
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2966.93 seconds
Loss after training = 8857.675420261352
800052
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2976.8 seconds
Loss after training = 9328.010964443183
800053
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3040.94 seconds
Loss after training = 9014.987020593811
800054
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2968.45 seconds
Loss after training = 8876.761675146856
800055
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3013.53 seconds
Loss after training = 10006.368479906427
800056
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3037.62 seconds
Loss after training = 9036.287745174795
800057
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2971.03 seconds
Loss after training = 10432.04886711763
800058
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2996.27 seconds
Loss after training = 8859.53124369772
800059
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 3034.92 seconds
Loss after training = 8971.77341769085
800060
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/ssss
Total time: 2945.64 seconds
Loss after training = 8848.366561372808
800061
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2991.95 seconds
Loss after training = 9149.941999738707
800062
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/ssss
Total time: 2999.24 seconds
Loss after training = 10033.179876942188
800063
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3025.27 seconds
Loss after training = 8881.537457469962
800064
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.41 iterations/s/ss
Total time: 2933.54 seconds
Loss after training = 9195.283482078243
800065
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2976.78 seconds
Loss after training = 9118.075557313436
800066
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 2950.42 seconds
Loss after training = 9195.733977194981
800067
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3005.27 seconds
Loss after training = 8854.01050875229
800068
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2974.07 seconds
Loss after training = 9018.515433646586
800069
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2993.26 seconds
Loss after training = 8926.392476962415
800070
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2981.79 seconds
Loss after training = 9108.28639122505
800071
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3016.97 seconds
Loss after training = 8834.899738105823
800072
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/ssss
Total time: 3063.0 seconds
Loss after training = 9144.247854847432
800073
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2980.81 seconds
Loss after training = 9034.285251921952
800074
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2986.38 seconds
Loss after training = 8838.208140664385
800075
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3060.34 seconds
Loss after training = 8953.057240316239
800076
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2984.28 seconds
Loss after training = 9211.125409962056
800077
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/ssss
Total time: 3035.25 seconds
Loss after training = 9267.043492022012
800078
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2959.44 seconds
Loss after training = 9252.631952556547
800079
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3000.65 seconds
Loss after training = 10609.808978286443
800080
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/ssss
Total time: 2946.27 seconds
Loss after training = 8828.208666087854
800081
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2961.52 seconds
Loss after training = 9363.469809973394
800082
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2970.66 seconds
Loss after training = 9129.200909362977
800083
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2979.36 seconds
Loss after training = 9957.71907446728
800084
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 2949.01 seconds
Loss after training = 8833.315080856919
800085
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/sss
Total time: 2943.63 seconds
Loss after training = 8865.561435317624
800086
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3022.98 seconds
Loss after training = 9124.896170807666
800087
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.6 iterations/ss/ss
Total time: 6253.16 seconds
Loss after training = 10154.589466840245
800088
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.23 iterations/s/ss
Total time: 4477.83 seconds
Loss after training = 8850.933339318699
800089
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2980.81 seconds
Loss after training = 9760.093409958987
800090
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2997.62 seconds
Loss after training = 8983.45792287981
800091
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3005.11 seconds
Loss after training = 8878.965256648078
800092
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2974.56 seconds
Loss after training = 8883.585736575647
800093
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.17 iterations/s/ss
Total time: 3158.18 seconds
Loss after training = 8837.355477905576
800094
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.47 iterations/s/ss
Total time: 4056.05 seconds
Loss after training = 9100.639721149962
800095
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.6 iterations/s/s/s
Total time: 3843.07 seconds
Loss after training = 9202.279659942344
800096
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.58 iterations/s/ss
Total time: 3882.09 seconds
Loss after training = 8838.410834880018
800097
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.64 iterations/ss/s
Total time: 3783.34 seconds
Loss after training = 10573.836388880578
800098
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.58 iterations/ss/s
Total time: 6335.31 seconds
Loss after training = 8938.841289000544
800099
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.56 iterations/ss/ss
Total time: 6394.84 seconds
Loss after training = 9010.560101873856
"""


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
PlumNormal = """800000
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3024.7 seconds
Loss after training = 9681.599956016287
800001
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/s/ss
Total time: 3053.7 seconds
Loss after training = 8839.38831870605
800002
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3057.33 seconds
Loss after training = 8909.124789167907
800003
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3037.18 seconds
Loss after training = 8949.19205522614
800004
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3048.73 seconds
Loss after training = 8835.896700153533
800005
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3064.26 seconds
Loss after training = 8961.625264793613
800006
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3046.65 seconds
Loss after training = 8855.522273302986
800007
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3023.63 seconds
Loss after training = 9275.500566611332
800008
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3057.56 seconds
Loss after training = 9831.542590252135
800009
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3069.61 seconds
Loss after training = 8836.115979963497
800010
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3067.84 seconds
Loss after training = 9836.89482514315
800011
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/ss/ss
Total time: 3026.53 seconds
Loss after training = 9090.102363160124
800012
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.21 iterations/s/ss
Total time: 3112.76 seconds
Loss after training = 8825.096085228557
800013
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3067.8 seconds
Loss after training = 10452.685311293299
800014
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.2 iterations/s/sss
Total time: 3125.25 seconds
Loss after training = 9841.14091175098
800015
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3066.35 seconds
Loss after training = 8893.53217020089
800016
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3061.91 seconds
Loss after training = 9022.898005302246
800017
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3060.29 seconds
Loss after training = 8847.577945570776
800018
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.22 iterations/s/ss
Total time: 3106.67 seconds
Loss after training = 8905.889995364372
800019
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 3034.02 seconds
Loss after training = 9612.593890681259
800020
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3059.19 seconds
Loss after training = 8983.585063043296
800021
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3073.94 seconds
Loss after training = 9010.507561891774
800022
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3080.1 seconds
Loss after training = 9428.870316830036
800023
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.22 iterations/s/ss
Total time: 3106.24 seconds
Loss after training = 9381.133746164005
800024
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3061.21 seconds
Loss after training = 9783.75045338558
800025
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3068.32 seconds
Loss after training = 10411.450213207308
800026
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3100.12 seconds
Loss after training = 8843.146784678453
800027
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/ssss
Total time: 3063.22 seconds
Loss after training = 8947.430066994437
800028
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3009.17 seconds
Loss after training = 9072.825437890398
800029
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3069.33 seconds
Loss after training = 8969.667266097833
800030
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3068.74 seconds
Loss after training = 8836.92557768906
800031
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.2 iterations/s/s/s
Total time: 3124.39 seconds
Loss after training = 8917.24544262432
800032
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3075.77 seconds
Loss after training = 8912.090994052527
800033
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3086.37 seconds
Loss after training = 8836.592806066868
800034
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3079.45 seconds
Loss after training = 8824.45243599756
800035
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/ssss
Total time: 3025.35 seconds
Loss after training = 8976.133273503368
800036
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3021.46 seconds
Loss after training = 8970.132628689418
800037
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3023.54 seconds
Loss after training = 9582.969354845713
800038
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2978.25 seconds
Loss after training = 8823.192841862841
800039
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3015.05 seconds
Loss after training = 9655.204113425618
800040
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/ss/ss
Total time: 3026.24 seconds
Loss after training = 9130.320942978535
800041
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2978.89 seconds
Loss after training = 9445.386150228534
800042
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2988.96 seconds
Loss after training = 9267.519360196893
800043
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2994.11 seconds
Loss after training = 8858.323492211064
800044
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/ss
Total time: 3034.79 seconds
Loss after training = 8943.166559196963
800045
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 2999.62 seconds
Loss after training = 8888.525625284745
800046
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3011.69 seconds
Loss after training = 8970.238788873396
800047
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3002.56 seconds
Loss after training = 8860.16354468807
800048
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3014.07 seconds
Loss after training = 9142.739337587878
800049
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2968.21 seconds
Loss after training = 9049.755149339995
800050
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2965.69 seconds
Loss after training = 9032.376648766185
800051
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3010.62 seconds
Loss after training = 8839.837174521652
800052
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3006.91 seconds
Loss after training = 9378.58167565074
800053
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3052.11 seconds
Loss after training = 9036.19256756928
800054
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3041.59 seconds
Loss after training = 8880.982698158472
800055
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2970.14 seconds
Loss after training = 10065.302923198335
800056
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/ssss
Total time: 3035.16 seconds
Loss after training = 8973.069779638041
800057
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2969.08 seconds
Loss after training = 10492.486766747897
800058
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3003.98 seconds
Loss after training = 8861.521295908753
800059
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3004.87 seconds
Loss after training = 8985.507248680411
800060
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3050.68 seconds
Loss after training = 8850.059128886653
800061
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3016.98 seconds
Loss after training = 9139.183388231253
800062
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 2990.7 seconds
Loss after training = 9675.385281306482
800063
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3019.16 seconds
Loss after training = 8852.569428594326
800064
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 2999.67 seconds
Loss after training = 9229.782789034854
800065
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3003.8 seconds
Loss after training = 9147.898324236563
800066
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2975.03 seconds
Loss after training = 9122.893551149555
800067
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2983.81 seconds
Loss after training = 8834.391101734918
800068
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2984.1 seconds
Loss after training = 8919.429474284403
800069
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3037.7 seconds
Loss after training = 8938.568897970632
800070
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2993.38 seconds
Loss after training = 9041.443488226701
800071
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2976.96 seconds
Loss after training = 8833.625840720695
800072
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3037.42 seconds
Loss after training = 9088.739175483523
800073
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3015.54 seconds
Loss after training = 9052.399427473205
800074
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/ss
Total time: 3034.36 seconds
Loss after training = 8839.74224277386
800075
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2996.32 seconds
Loss after training = 8961.427302352746
800076
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3014.94 seconds
Loss after training = 9127.795711207506
800077
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2998.07 seconds
Loss after training = 9267.076232167587
800078
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3017.12 seconds
Loss after training = 9293.209735548118
800079
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2972.36 seconds
Loss after training = 9805.73204276832
800080
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3021.55 seconds
Loss after training = 8822.966968386689
800081
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 3030.67 seconds
Loss after training = 9276.417622340938
800082
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/ssss
Total time: 2998.61 seconds
Loss after training = 9147.80970915188
800083
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2970.09 seconds
Loss after training = 9830.682104861282
800084
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/s/s
Total time: 3027.55 seconds
Loss after training = 8828.80291445364
800085
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3071.99 seconds
Loss after training = 8834.275422314033
800086
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.6 iterations/ss/ss
Total time: 3840.95 seconds
Loss after training = 8962.620115493173
800087
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.43 iterations/ss/s
Total time: 7014.73 seconds
Loss after training = 10015.319105674895
800088
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.12 iterations/ssss
Total time: 3200.96 seconds
Loss after training = 8852.770772390142
800089
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 2947.48 seconds
Loss after training = 9669.559265819926
800090
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/ss
Total time: 2942.08 seconds
Loss after training = 8974.907180118882
800091
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2996.92 seconds
Loss after training = 8854.987634878273
800092
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2967.78 seconds
Loss after training = 8912.21882920695
800093
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.82 iterations/s/ss
Total time: 3547.4 seconds
Loss after training = 8825.256270450007
800094
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.56 iterations/s/ss
Total time: 3899.37 seconds
Loss after training = 9133.569153178047
800095
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.72 iterations/s/ss
Total time: 3680.82 seconds
Loss after training = 9107.329924278582
800096
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.54 iterations/s/ss
Total time: 3931.35 seconds
Loss after training = 8839.926872303608
800097
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.68 iterations/s/ss
Total time: 3730.15 seconds
Loss after training = 10221.23192666349
800098
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.52 iterations/ss/s
Total time: 6574.82 seconds
Loss after training = 8867.10374876233
800099
True loss: 8835.514565048345
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.46 iterations/ss/s
Total time: 6829.43 seconds
Loss after training = 8963.177262662834
"""

#BW:
BWNormal = """800000
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3003.44 seconds
Loss after training = 9700.963413840364
800001
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3020.49 seconds
Loss after training = 8839.315880430386
800002
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3017.92 seconds
Loss after training = 8913.397156060537
800003
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3092.47 seconds
Loss after training = 8950.56171634964
800004
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3022.6 seconds
Loss after training = 8835.811626969287
800005
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3082.87 seconds
Loss after training = 8961.787418819596
800006
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3036.37 seconds
Loss after training = 8855.727319153639
800007
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3041.66 seconds
Loss after training = 9289.305510887963
800008
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3003.36 seconds
Loss after training = 9832.118537447126
800009
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3049.15 seconds
Loss after training = 8833.113998501982
800010
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3072.51 seconds
Loss after training = 9837.722630126773
800011
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3076.83 seconds
Loss after training = 9089.522189943225
800012
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3069.34 seconds
Loss after training = 8825.083668155186
800013
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.22 iterations/s/ss
Total time: 3108.73 seconds
Loss after training = 10453.970032270625
800014
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3042.86 seconds
Loss after training = 9819.85369764563
800015
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3085.92 seconds
Loss after training = 8893.615564405152
800016
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3051.43 seconds
Loss after training = 9024.05312909778
800017
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3053.05 seconds
Loss after training = 8847.50643085414
800018
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3048.27 seconds
Loss after training = 8906.018243525657
800019
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 3030.48 seconds
Loss after training = 9613.545677311571
800020
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/s
Total time: 3036.56 seconds
Loss after training = 8983.775190765504
800021
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3018.32 seconds
Loss after training = 9010.713029700813
800022
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/s/s
Total time: 3027.28 seconds
Loss after training = 9429.24931470452
800023
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3064.34 seconds
Loss after training = 9394.00670727974
800024
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3064.83 seconds
Loss after training = 9784.70711418218
800025
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3056.6 seconds
Loss after training = 10412.022830136248
800026
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3098.35 seconds
Loss after training = 8843.168501368607
800027
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 3029.42 seconds
Loss after training = 8947.289594809206
800028
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 3033.12 seconds
Loss after training = 9078.002387094926
800029
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3095.44 seconds
Loss after training = 8970.426512967402
800030
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3049.94 seconds
Loss after training = 8837.066760732518
800031
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3039.09 seconds
Loss after training = 8916.39749315184
800032
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3080.42 seconds
Loss after training = 8911.603414078185
800033
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3052.1 seconds
Loss after training = 8833.571804512789
800034
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3010.45 seconds
Loss after training = 8824.434627329842
800035
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2991.49 seconds
Loss after training = 8975.972638793872
800036
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2989.07 seconds
Loss after training = 8970.30974590593
800037
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2978.68 seconds
Loss after training = 9583.50000742271
800038
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/s
Total time: 2948.88 seconds
Loss after training = 8823.184954686294
800039
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 2952.3 seconds
Loss after training = 9654.956912065114
800040
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/ss/ss
Total time: 2937.6 seconds
Loss after training = 9130.586344280515
800041
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2970.66 seconds
Loss after training = 9433.626195834298
800042
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2962.9 seconds
Loss after training = 9269.19785855559
800043
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3005.42 seconds
Loss after training = 8858.223568950982
800044
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2963.08 seconds
Loss after training = 8943.972015426727
800045
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2959.75 seconds
Loss after training = 8887.980547269783
800046
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2981.69 seconds
Loss after training = 8966.310774202944
800047
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3021.87 seconds
Loss after training = 8860.130458448553
800048
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2993.46 seconds
Loss after training = 9142.947236594795
800049
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ss/s
Total time: 3055.35 seconds
Loss after training = 9048.080005336244
800050
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2963.59 seconds
Loss after training = 9030.04437474151
800051
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3023.73 seconds
Loss after training = 8838.609957553175
800052
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2970.63 seconds
Loss after training = 9382.164674392392
800053
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3008.78 seconds
Loss after training = 9041.293071055481
800054
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2970.67 seconds
Loss after training = 8880.881730205392
800055
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2998.42 seconds
Loss after training = 10092.59280330438
800056
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3023.45 seconds
Loss after training = 8973.24723342797
800057
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3001.8 seconds
Loss after training = 10531.645170793015
800058
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3039.15 seconds
Loss after training = 8861.903324347018
800059
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3017.1 seconds
Loss after training = 8989.067001529023
800060
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3017.11 seconds
Loss after training = 8849.955654648766
800061
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2989.35 seconds
Loss after training = 9122.537736738948
800062
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/ss/ss
Total time: 3025.92 seconds
Loss after training = 9675.995612758203
800063
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2983.19 seconds
Loss after training = 8848.782243921818
800064
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2965.32 seconds
Loss after training = 9236.700221578942
800065
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3010.65 seconds
Loss after training = 9154.153758774055
800066
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2958.01 seconds
Loss after training = 9123.167180423763
800067
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3072.15 seconds
Loss after training = 8834.51869058696
800068
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2997.06 seconds
Loss after training = 8919.699136500169
800069
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3011.53 seconds
Loss after training = 8938.426233568007
800070
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3023.62 seconds
Loss after training = 9041.66967736455
800071
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2968.13 seconds
Loss after training = 8833.573388428611
800072
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3010.95 seconds
Loss after training = 9087.818256070836
800073
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3020.37 seconds
Loss after training = 9060.640513411163
800074
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3023.56 seconds
Loss after training = 8839.677937256387
800075
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3056.78 seconds
Loss after training = 8967.281336174156
800076
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3036.92 seconds
Loss after training = 9128.085581732737
800077
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3012.18 seconds
Loss after training = 9257.424214226747
800078
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2975.32 seconds
Loss after training = 9300.714484688995
800079
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3025.05 seconds
Loss after training = 9806.870770313868
800080
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3004.85 seconds
Loss after training = 8822.86390344738
800081
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3009.64 seconds
Loss after training = 9276.743815224718
800082
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2960.94 seconds
Loss after training = 9159.655500610632
800083
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3055.42 seconds
Loss after training = 9831.215383664932
800084
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2961.08 seconds
Loss after training = 8828.761642132818
800085
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2967.21 seconds
Loss after training = 8834.31827612132
800086
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3040.4 seconds
Loss after training = 8963.061491453873
800087
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.78 iterations/s/ss
Total time: 5627.36 seconds
Loss after training = 10015.893047255366
800088
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.9 iterations/ss/sss
Total time: 5267.6 seconds
Loss after training = 8852.696476349822
800089
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/ssss
Total time: 2999.43 seconds
Loss after training = 9652.829088056515
800090
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3050.38 seconds
Loss after training = 8966.49965351981
800091
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3006.59 seconds
Loss after training = 8851.671808473278
800092
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3025.14 seconds
Loss after training = 8914.760506890212
800093
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.13 iterations/ssss
Total time: 3190.83 seconds
Loss after training = 8825.099571773078
800094
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.49 iterations/ssss
Total time: 4021.77 seconds
Loss after training = 9133.631162917713
800095
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.6 iterations/ss/ss
Total time: 3852.29 seconds
Loss after training = 9104.367561979623
800096
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.55 iterations/s/ss
Total time: 3926.73 seconds
Loss after training = 8839.868858183647
800097
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.63 iterations/ss/s
Total time: 3798.67 seconds
Loss after training = 10221.819769970629
800098
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.52 iterations/ss/s
Total time: 6572.15 seconds
Loss after training = 8867.326235193692
800099
True loss: 8835.514565085514
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.44 iterations/ss/ss
Total time: 6964.9 seconds
Loss after training = 8957.350014702191
"""

#Zhao:
ZhaoNormal = """800000
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2985.82 seconds
Loss after training = 9690.751455304171
800001
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3066.42 seconds
Loss after training = 8838.265848796415
800002
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3044.65 seconds
Loss after training = 8901.803921355033
800003
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 3029.95 seconds
Loss after training = 8941.50915835528
800004
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3022.31 seconds
Loss after training = 8835.556252366066
800005
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3075.25 seconds
Loss after training = 8963.939288444884
800006
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3095.86 seconds
Loss after training = 8858.022603131767
800007
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3056.95 seconds
Loss after training = 9273.235045082629
800008
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3067.53 seconds
Loss after training = 9838.636344556215
800009
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.19 iterations/s/ss
Total time: 3138.51 seconds
Loss after training = 8837.595054382778
800010
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3068.71 seconds
Loss after training = 9804.432764838755
800011
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 2999.22 seconds
Loss after training = 9090.99724903643
800012
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.22 iterations/s/ss
Total time: 3108.83 seconds
Loss after training = 8825.259904222577
800013
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3044.42 seconds
Loss after training = 10371.31566224987
800014
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.21 iterations/ss/s
Total time: 3117.15 seconds
Loss after training = 9917.253525045358
800015
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3090.89 seconds
Loss after training = 8893.468622796261
800016
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3081.69 seconds
Loss after training = 9019.04052491718
800017
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3052.35 seconds
Loss after training = 8846.14397209569
800018
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3067.11 seconds
Loss after training = 8906.709946195693
800019
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3051.32 seconds
Loss after training = 9610.07293286848
800020
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3060.89 seconds
Loss after training = 8984.592668418043
800021
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3063.06 seconds
Loss after training = 9008.67437861851
800022
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3099.43 seconds
Loss after training = 9424.441296547457
800023
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3096.86 seconds
Loss after training = 9361.566999024815
800024
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3094.81 seconds
Loss after training = 9743.227061701098
800025
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 3029.89 seconds
Loss after training = 10322.747414248961
800026
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.22 iterations/s/ss
Total time: 3101.69 seconds
Loss after training = 8846.253345768191
800027
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.2 iterations/s/ss
Total time: 3125.36 seconds
Loss after training = 8945.874331604873
800028
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3097.72 seconds
Loss after training = 9067.985227761408
800029
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3082.81 seconds
Loss after training = 8970.512440059052
800030
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3038.42 seconds
Loss after training = 8837.694977996849
800031
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3058.72 seconds
Loss after training = 8928.272937420184
800032
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3096.69 seconds
Loss after training = 8908.37203315286
800033
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3084.47 seconds
Loss after training = 8838.160751969897
800034
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3076.69 seconds
Loss after training = 8824.594050865378
800035
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3003.97 seconds
Loss after training = 8975.640996960297
800036
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.41 iterations/s/ss
Total time: 2934.14 seconds
Loss after training = 8972.173114572975
800037
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2989.36 seconds
Loss after training = 9572.856556875831
800038
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2969.44 seconds
Loss after training = 8823.63214818522
800039
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2967.73 seconds
Loss after training = 9613.85909418031
800040
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3022.23 seconds
Loss after training = 9131.450270699814
800041
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 2950.94 seconds
Loss after training = 9415.236104688698
800042
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3013.11 seconds
Loss after training = 9262.336691498585
800043
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 3029.46 seconds
Loss after training = 8858.409325195835
800044
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3010.96 seconds
Loss after training = 8940.52709005729
800045
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2995.05 seconds
Loss after training = 8894.068053292252
800046
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3018.52 seconds
Loss after training = 8980.86705209287
800047
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2987.23 seconds
Loss after training = 8866.449016828012
800048
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3018.29 seconds
Loss after training = 9139.678734428451
800049
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3019.29 seconds
Loss after training = 9064.889060696258
800050
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3038.12 seconds
Loss after training = 9057.825474482452
800051
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2984.49 seconds
Loss after training = 8842.454366638944
800052
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 2951.39 seconds
Loss after training = 9370.335066923115
800053
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2985.51 seconds
Loss after training = 9032.646857277734
800054
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 3026.89 seconds
Loss after training = 8879.344719308227
800055
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2976.12 seconds
Loss after training = 10058.878744152225
800056
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3037.36 seconds
Loss after training = 8973.548409260144
800057
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2994.55 seconds
Loss after training = 10487.469698850393
800058
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/ss
Total time: 2943.54 seconds
Loss after training = 8860.81583355893
800059
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2962.08 seconds
Loss after training = 8982.318238078607
800060
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2957.79 seconds
Loss after training = 8849.43989403427
800061
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3018.51 seconds
Loss after training = 9184.873676636573
800062
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3002.44 seconds
Loss after training = 9657.99437790456
800063
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 2999.29 seconds
Loss after training = 8856.44978218728
800064
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2989.79 seconds
Loss after training = 9223.380943095835
800065
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3010.7 seconds
Loss after training = 9142.218693792547
800066
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2976.69 seconds
Loss after training = 9125.484717037427
800067
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/ss
Total time: 3033.01 seconds
Loss after training = 8834.59601489882
800068
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2987.9 seconds
Loss after training = 8913.368241559796
800069
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3000.64 seconds
Loss after training = 8938.191607210154
800070
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3008.58 seconds
Loss after training = 9045.062214602869
800071
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2996.24 seconds
Loss after training = 8833.481733710509
800072
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3004.86 seconds
Loss after training = 9120.712714559488
800073
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/ss
Total time: 3032.57 seconds
Loss after training = 9048.699820957087
800074
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2997.7 seconds
Loss after training = 8839.747294635696
800075
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3022.83 seconds
Loss after training = 8959.342247827753
800076
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2994.06 seconds
Loss after training = 9128.558600290458
800077
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/ssss
Total time: 3063.16 seconds
Loss after training = 9310.547380268657
800078
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2969.5 seconds
Loss after training = 9286.05227155079
800079
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3049.12 seconds
Loss after training = 9738.804573632442
800080
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2976.37 seconds
Loss after training = 8823.442000127305
800081
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3052.61 seconds
Loss after training = 9276.843965074806
800082
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3009.07 seconds
Loss after training = 9145.039788119662
800083
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2979.14 seconds
Loss after training = 9835.424837909603
800084
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3021.51 seconds
Loss after training = 8828.771738107018
800085
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2984.04 seconds
Loss after training = 8834.855135900969
800086
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.89 iterations/s/ss
Total time: 3455.74 seconds
Loss after training = 8945.04423317026
800087
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.44 iterations/ss/ss
Total time: 6955.63 seconds
Loss after training = 10018.221447287391
800088
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.82 iterations/s/sss
Total time: 3545.03 seconds
Loss after training = 8852.36801758506
800089
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/ssss
Total time: 3035.52 seconds
Loss after training = 9724.022103008816
800090
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 2947.1 seconds
Loss after training = 9005.596589798273
800091
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2978.31 seconds
Loss after training = 8857.903667872299
800092
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2985.67 seconds
Loss after training = 8905.610948618341
800093
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.88 iterations/s/ss
Total time: 3478.25 seconds
Loss after training = 8826.668971275558
800094
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.52 iterations/s/ss
Total time: 3976.09 seconds
Loss after training = 9128.974866121038
800095
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.74 iterations/s/ss
Total time: 3648.24 seconds
Loss after training = 9112.977914007404
800096
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.52 iterations/s/ss
Total time: 3970.83 seconds
Loss after training = 8839.760354949754
800097
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.68 iterations/s/ss
Total time: 3730.4 seconds
Loss after training = 10208.762391321685
800098
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.63 iterations/ssss
Total time: 6117.55 seconds
Loss after training = 8864.76144594405
800099
True loss: 8835.514565105455
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.59 iterations/ss/s
Total time: 6299.06 seconds
Loss after training = 8971.916623830195
"""

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

plt.figure()
sns.boxplot(data=[zhaoDiffsNormal, zhaoDiffsTrue]).set(xlabel="Left: IG3 starting from true DM distribution. Right: IG3.",title='{Reconstructed loss} - {True loss} for Zhao')
plt.ylim(-180,2100)

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


