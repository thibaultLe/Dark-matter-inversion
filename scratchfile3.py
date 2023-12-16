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
PlumTrue = """700000
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2959.53 seconds
Loss after training = 8763.081849230892
700001
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2964.33 seconds
Loss after training = 8717.17676707396
700002
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/ss/ss
Total time: 3034.54 seconds
Loss after training = 8693.53893590968
700003
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/s/ss
Total time: 3060.74 seconds
Loss after training = 9232.180866684686
700004
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3098.74 seconds
Loss after training = 9042.26600425325
700005
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.12 iterations/ssss
Total time: 3203.24 seconds
Loss after training = 8903.14565396125
700006
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/ss/ss
Total time: 3034.22 seconds
Loss after training = 8759.584682983894
700007
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2971.6 seconds
Loss after training = 8789.224044954279
700008
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2980.45 seconds
Loss after training = 8757.604786075999
700009
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3000.36 seconds
Loss after training = 9263.62289225104
700010
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3001.15 seconds
Loss after training = 8693.121111175913
700011
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3014.19 seconds
Loss after training = 10414.943515391778
700012
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2985.99 seconds
Loss after training = 8895.903288004689
700013
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2990.71 seconds
Loss after training = 9431.638560058895
700014
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2989.58 seconds
Loss after training = 9083.091747071092
700015
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2992.93 seconds
Loss after training = 8969.998837229678
700016
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2997.16 seconds
Loss after training = 8680.197183110677
700017
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3002.29 seconds
Loss after training = 8844.099872985347
700018
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/ss/ss
Total time: 3027.15 seconds
Loss after training = 9028.165472681316
700019
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2990.79 seconds
Loss after training = 8704.60321667074
700020
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2966.31 seconds
Loss after training = 9053.669524669907
700021
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/s
Total time: 2946.42 seconds
Loss after training = 10519.53821560734
700022
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2974.82 seconds
Loss after training = 8680.727563945993
700023
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 2948.44 seconds
Loss after training = 8720.557715741306
700024
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2973.73 seconds
Loss after training = 8887.766736336307
700025
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2954.52 seconds
Loss after training = 8998.84620784693
700026
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2988.34 seconds
Loss after training = 8954.831593978915
700027
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2963.56 seconds
Loss after training = 8870.121671637697
700028
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2970.74 seconds
Loss after training = 8862.298611474582
700029
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2972.1 seconds
Loss after training = 10010.537157457624
700030
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2979.68 seconds
Loss after training = 9350.328607622154
700031
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2975.36 seconds
Loss after training = 9173.811782101357
700032
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/sss
Total time: 2941.37 seconds
Loss after training = 8867.955289886813
700033
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.21 iterations/ss/s
Total time: 3111.62 seconds
Loss after training = 8932.562768728048
700034
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.16 iterations/s/ss
Total time: 3169.32 seconds
Loss after training = 9022.05731409796
700035
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/sss
Total time: 2944.68 seconds
Loss after training = 8683.162443560996
700036
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.41 iterations/s/ss
Total time: 2930.49 seconds
Loss after training = 8850.699149518421
700037
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2967.87 seconds
Loss after training = 9188.09702060611
700038
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.77 iterations/s/ss
Total time: 5646.03 seconds
Loss after training = 10208.106869378955
700039
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.62 iterations/s/sss
Total time: 6183.25 seconds
Loss after training = 8695.108099318202
700040
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3078.89 seconds
Loss after training = 8684.149473874273
700041
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3021.91 seconds
Loss after training = 8787.429215497874
700042
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.2 iterations/s/s/s
Total time: 3121.13 seconds
Loss after training = 9014.558594567638
700043
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3036.01 seconds
Loss after training = 8681.693616819744
700044
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3038.21 seconds
Loss after training = 8898.036785212926
700045
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 3032.37 seconds
Loss after training = 8825.950044516827
700046
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3039.69 seconds
Loss after training = 8850.151292084242
700047
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3062.05 seconds
Loss after training = 9340.045085579695
700048
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/s/s
Total time: 3029.72 seconds
Loss after training = 10029.102005220362
700049
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3065.48 seconds
Loss after training = 8762.839392829184
700050
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3072.31 seconds
Loss after training = 9086.345780191175
700051
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3082.9 seconds
Loss after training = 9051.197633285798
700052
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.22 iterations/s/ss
Total time: 3102.62 seconds
Loss after training = 8752.36407533614
700053
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3044.89 seconds
Loss after training = 10257.21787900521
700054
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3090.97 seconds
Loss after training = 8853.416562184133
700055
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 3029.08 seconds
Loss after training = 8924.535838354237
700056
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3093.46 seconds
Loss after training = 10128.89894020994
700057
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3072.35 seconds
Loss after training = 8810.949059625147
700058
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3077.04 seconds
Loss after training = 8954.72174202234
700059
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/ssss
Total time: 3098.25 seconds
Loss after training = 8708.37120575409
700060
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.04 iterations/s/sss
Total time: 9579.49 seconds
Loss after training = 8681.443421641125
700061
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.35 iterations/ss/ss
Total time: 7428.93 seconds
Loss after training = 8730.37838774644
700062
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.25 iterations/ss/ss
Total time: 7996.22 seconds
Loss after training = 9161.238803116985
700063
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.27 iterations/ss/ss
Total time: 7847.83 seconds
Loss after training = 9228.955114556085
700064
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.37 iterations/ss/ss
Total time: 7291.4 seconds
Loss after training = 8737.101160408958
700065
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.25 iterations/ss/ss
Total time: 7995.48 seconds
Loss after training = 8724.474003955254
700066
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.28 iterations/ss/s
Total time: 7814.21 seconds
Loss after training = 10566.986364413782
700067
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.31 iterations/ss/ss
Total time: 7608.8 seconds
Loss after training = 8749.842940520577
700068
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.3 iterations/sns/ss
Total time: 7694.56 seconds
Loss after training = 8724.928475042117
700069
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.42 iterations/ss/ss
Total time: 7039.68 seconds
Loss after training = 9298.07418896826
700070
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.58 iterations/ss/s
Total time: 6311.33 seconds
Loss after training = 9155.661084339386
700071
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.2 iterations/ss/ss
Total time: 3125.38 seconds
Loss after training = 8958.679666276896
700072
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3010.31 seconds
Loss after training = 8942.572421669727
700073
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2989.11 seconds
Loss after training = 8682.043002643213
700074
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.31 iterations/s/ss
Total time: 4334.71 seconds
Loss after training = 9035.191203786517
700075
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.58 iterations/s/ss
Total time: 3878.99 seconds
Loss after training = 8681.671379691465
700076
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.3 iterations/ss/ss
Total time: 4348.21 seconds
Loss after training = 9211.900978080579
700077
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.2 iterations/sns/ss
Total time: 8317.56 seconds
Loss after training = 8731.854054783724
700078
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.18 iterations/ss/ss
Total time: 8467.23 seconds
Loss after training = 8697.569233823373
700079
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.21 iterations/ss/s
Total time: 8282.16 seconds
Loss after training = 8732.097943860907
700080
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.19 iterations/ss/ss
Total time: 8392.96 seconds
Loss after training = 8972.12507494678
700081
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.27 iterations/s/sss
Total time: 4412.62 seconds
Loss after training = 8722.781571306657
700082
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/ss/ss
Total time: 3026.57 seconds
Loss after training = 8683.03007375624
700083
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 3031.17 seconds
Loss after training = 9461.258781023607
700084
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3077.21 seconds
Loss after training = 8837.00561463032
700085
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3074.06 seconds
Loss after training = 8745.792702648128
700086
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.22 iterations/s/ss
Total time: 3105.08 seconds
Loss after training = 8686.870535044483
700087
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3069.26 seconds
Loss after training = 9273.175001326972
700088
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3098.55 seconds
Loss after training = 9219.435746733334
700089
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3065.22 seconds
Loss after training = 9053.392089307026
700090
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.24 iterations/s/ss
Total time: 4458.0 seconds
Loss after training = 8680.178432332277
700091
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.17 iterations/ss/ss
Total time: 8537.95 seconds
Loss after training = 9121.506055103653
700092
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.16 iterations/ss/ss
Total time: 8592.88 seconds
Loss after training = 8698.13947512345
700093
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.22 iterations/ss/s
Total time: 8187.44 seconds
Loss after training = 9166.225856054469
700094
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.13 iterations/ss/ss
Total time: 8866.34 seconds
Loss after training = 8696.256035645922
700095
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.32 iterations/ss/ss
Total time: 7587.18 seconds
Loss after training = 8683.29718632734
700096
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.17 iterations/ss/ss
Total time: 8567.7 seconds
Loss after training = 9078.447115640827
700097
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.71 iterations/s/ss
Total time: 3694.58 seconds
Loss after training = 11724.986881009994
700098
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3081.04 seconds
Loss after training = 9517.462424137308
700099
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3037.9 seconds
Loss after training = 8686.389995588603
"""

#BW:
    
BWTrue = """700000
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2960.27 seconds
Loss after training = 8763.157215088368
700001
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2984.19 seconds
Loss after training = 8717.174213921659
700002
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3017.83 seconds
Loss after training = 8693.447690178446
700003
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 3031.08 seconds
Loss after training = 9227.912006079616
700004
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3039.85 seconds
Loss after training = 9042.223031320498
700005
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.1 iterations/ss/ss
Total time: 3226.57 seconds
Loss after training = 8906.320723109868
700006
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3009.11 seconds
Loss after training = 8759.660497104249
700007
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2989.41 seconds
Loss after training = 8793.287064051186
700008
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2976.02 seconds
Loss after training = 8757.68192423213
700009
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/ssss
Total time: 2999.29 seconds
Loss after training = 9258.64623241068
700010
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2974.93 seconds
Loss after training = 8693.148329798534
700011
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2972.77 seconds
Loss after training = 10415.221577180972
700012
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2991.59 seconds
Loss after training = 8896.154504271502
700013
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2984.78 seconds
Loss after training = 9432.143335800542
700014
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2975.38 seconds
Loss after training = 9088.873418527979
700015
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3020.48 seconds
Loss after training = 8941.34535643159
700016
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2964.46 seconds
Loss after training = 8680.211840227978
700017
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3006.19 seconds
Loss after training = 8849.128882975947
700018
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2969.52 seconds
Loss after training = 8986.827711685082
700019
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2985.8 seconds
Loss after training = 8704.556172863517
700020
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2956.48 seconds
Loss after training = 9053.817131994165
700021
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.42 iterations/s/ss
Total time: 2926.91 seconds
Loss after training = 10558.484094950436
700022
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2971.24 seconds
Loss after training = 8680.111449121723
700023
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/ss
Total time: 2943.95 seconds
Loss after training = 8720.980628381052
700024
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2988.01 seconds
Loss after training = 8887.871981986016
700025
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 2948.19 seconds
Loss after training = 8999.175946023548
700026
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2993.56 seconds
Loss after training = 8946.481395788394
700027
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2996.94 seconds
Loss after training = 8872.247265365007
700028
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2973.79 seconds
Loss after training = 8862.268452964656
700029
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/ssss
Total time: 3000.27 seconds
Loss after training = 10010.21364903387
700030
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2978.19 seconds
Loss after training = 9350.543045462031
700031
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2998.34 seconds
Loss after training = 9175.085548199087
700032
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2977.15 seconds
Loss after training = 8868.095964896884
700033
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.17 iterations/ss/s
Total time: 3159.58 seconds
Loss after training = 8932.689896277236
700034
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.12 iterations/s/ss
Total time: 3200.78 seconds
Loss after training = 9031.54874170948
700035
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2979.6 seconds
Loss after training = 8683.187645248061
700036
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3059.44 seconds
Loss after training = 8848.52811869212
700037
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3035.86 seconds
Loss after training = 9181.823250260595
700038
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.4 iterations/sns/s
Total time: 7168.33 seconds
Loss after training = 10208.754133167271
700039
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.03 iterations/s/sss
Total time: 4927.16 seconds
Loss after training = 8695.401997950315
700040
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3039.38 seconds
Loss after training = 8684.180383979186
700041
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3049.25 seconds
Loss after training = 8787.582909021863
700042
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.2 iterations/s/sss
Total time: 3123.48 seconds
Loss after training = 9014.703362416889
700043
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3042.32 seconds
Loss after training = 8681.716104539664
700044
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.22 iterations/s/ss
Total time: 3101.76 seconds
Loss after training = 8897.528368635087
700045
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/ss
Total time: 3031.01 seconds
Loss after training = 8832.853082354723
700046
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3065.77 seconds
Loss after training = 8850.307283933107
700047
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3076.65 seconds
Loss after training = 9340.182989272842
700048
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.2 iterations/s/ss
Total time: 3122.15 seconds
Loss after training = 10029.065894029092
700049
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.21 iterations/ss/s
Total time: 3117.0 seconds
Loss after training = 8766.918046216744
700050
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3062.18 seconds
Loss after training = 9088.689704855688
700051
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.21 iterations/ss/s
Total time: 3111.53 seconds
Loss after training = 9051.545550758969
700052
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3099.46 seconds
Loss after training = 8752.437030258952
700053
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3080.31 seconds
Loss after training = 10270.937841096676
700054
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3074.07 seconds
Loss after training = 8853.520177345006
700055
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3078.53 seconds
Loss after training = 8930.98587532279
700056
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.2 iterations/s/ss
Total time: 3127.33 seconds
Loss after training = 10128.887186105752
700057
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.21 iterations/ss/s
Total time: 3118.45 seconds
Loss after training = 8811.136737189425
700058
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3079.1 seconds
Loss after training = 8941.239571025406
700059
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.97 iterations/s/ss
Total time: 5064.21 seconds
Loss after training = 8708.371512074073
700060
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.09 iterations/ss/ss
Total time: 9138.82 seconds
Loss after training = 8681.471773795913
700061
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.3 iterations/sns/s
Total time: 7683.25 seconds
Loss after training = 8732.745134143914
700062
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.33 iterations/ss/s
Total time: 7502.05 seconds
Loss after training = 9170.237627549755
700063
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.33 iterations/ss/ss
Total time: 7518.06 seconds
Loss after training = 9224.57685597402
700064
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.27 iterations/ss/ss
Total time: 7862.86 seconds
Loss after training = 8737.16773164966
700065
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.22 iterations/ss/ss
Total time: 8217.26 seconds
Loss after training = 8727.368576542642
700066
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.3 iterations/sns/ss
Total time: 7684.12 seconds
Loss after training = 10616.212711875645
700067
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.19 iterations/ss/ss
Total time: 8426.1 seconds
Loss after training = 8752.297246083072
700068
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.35 iterations/ss/ss
Total time: 7430.07 seconds
Loss after training = 8726.03348057427
700069
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.19 iterations/ss/ss
Total time: 8408.99 seconds
Loss after training = 9298.25790428473
700070
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.39 iterations/s/sss
Total time: 4178.76 seconds
Loss after training = 9155.815754094869
700071
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3011.01 seconds
Loss after training = 8956.856674722982
700072
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3017.6 seconds
Loss after training = 8942.558177711791
700073
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.37 iterations/ss/s
Total time: 4219.13 seconds
Loss after training = 8681.960446920282
700074
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.41 iterations/s/ss
Total time: 4153.0 seconds
Loss after training = 9035.189284796294
700075
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.8 iterations/s/s/s
Total time: 3566.74 seconds
Loss after training = 8681.659966037238
700076
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.27 iterations/ss/s
Total time: 7879.53 seconds
Loss after training = 9212.060666656373
700077
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.31 iterations/ss/ss
Total time: 7643.31 seconds
Loss after training = 8731.99955762589
700078
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.25 iterations/ss/ss
Total time: 7998.69 seconds
Loss after training = 8697.573081974537
700079
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.17 iterations/ss/ss
Total time: 8524.56 seconds
Loss after training = 8732.16268986941
700080
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.62 iterations/s/sss
Total time: 6190.35 seconds
Loss after training = 8972.057067407626
700081
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3068.1 seconds
Loss after training = 8734.381841929731
700082
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.22 iterations/s/ss
Total time: 3104.23 seconds
Loss after training = 8683.060982403049
700083
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.21 iterations/ss/s
Total time: 3113.82 seconds
Loss after training = 9461.491239282741
700084
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3066.33 seconds
Loss after training = 8835.500072960836
700085
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.22 iterations/s/ss
Total time: 3110.37 seconds
Loss after training = 8745.933062082946
700086
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3040.09 seconds
Loss after training = 8686.802319534618
700087
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3070.75 seconds
Loss after training = 9273.39096497971
700088
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3063.87 seconds
Loss after training = 9215.058504275501
700089
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3077.52 seconds
Loss after training = 9040.856394262526
700090
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.34 iterations/ss/s
Total time: 7444.55 seconds
Loss after training = 8680.204474322549
700091
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.34 iterations/ss/ss
Total time: 7474.49 seconds
Loss after training = 9121.691419575118
700092
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.17 iterations/ss/ss
Total time: 8557.37 seconds
Loss after training = 8698.596902252728
700093
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.24 iterations/ss/s
Total time: 8046.12 seconds
Loss after training = 9166.397466991368
700094
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.19 iterations/ss/ss
Total time: 8377.91 seconds
Loss after training = 8696.262644254602
700095
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.17 iterations/ss/s
Total time: 8513.84 seconds
Loss after training = 8683.292999305564
700096
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.57 iterations/s/sss
Total time: 6379.26 seconds
Loss after training = 9087.668504024943
700097
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.22 iterations/s/ss
Total time: 3101.64 seconds
Loss after training = 11799.402130248773
700098
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3089.04 seconds
Loss after training = 9517.624071905422
700099
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.22 iterations/s/ss
Total time: 3108.54 seconds
Loss after training = 8684.468797754234
"""

#Zhao:
ZhaoTrue = """700000
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2975.45 seconds
Loss after training = 8763.926037889922
700001
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3007.35 seconds
Loss after training = 8716.469926792157
700002
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3065.42 seconds
Loss after training = 8695.367957442675
700003
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/ss/ss
Total time: 3033.12 seconds
Loss after training = 9285.195604862183
700004
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3099.75 seconds
Loss after training = 9039.836647484763
700005
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.98 iterations/s/ss
Total time: 3350.29 seconds
Loss after training = 8898.749699077622
700006
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/ssss
Total time: 2999.78 seconds
Loss after training = 8760.783299108001
700007
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3020.49 seconds
Loss after training = 8786.653892813989
700008
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2994.46 seconds
Loss after training = 8760.02630695563
700009
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2985.6 seconds
Loss after training = 9320.012656833065
700010
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3066.58 seconds
Loss after training = 8691.93517808558
700011
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3017.61 seconds
Loss after training = 10390.492976672045
700012
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 3031.27 seconds
Loss after training = 8878.807611789947
700013
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3005.22 seconds
Loss after training = 9397.60124484505
700014
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3008.13 seconds
Loss after training = 9076.534541512108
700015
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3046.57 seconds
Loss after training = 8986.000894642992
700016
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3001.87 seconds
Loss after training = 8680.275706652943
700017
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/s/s
Total time: 3028.68 seconds
Loss after training = 8840.531957652478
700018
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3042.94 seconds
Loss after training = 9114.800021234129
700019
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3038.63 seconds
Loss after training = 8702.451901477538
700020
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2978.79 seconds
Loss after training = 9039.649179563687
700021
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/ss
Total time: 3034.67 seconds
Loss after training = 10509.785160939091
700022
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3036.89 seconds
Loss after training = 8682.187700015369
700023
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2983.65 seconds
Loss after training = 8719.367092390194
700024
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/s/s
Total time: 3028.13 seconds
Loss after training = 8887.930904513883
700025
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ss/s
Total time: 3057.29 seconds
Loss after training = 8973.004380312215
700026
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3014.25 seconds
Loss after training = 8962.723080832995
700027
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3042.53 seconds
Loss after training = 8864.328305038072
700028
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/s/s
Total time: 3028.81 seconds
Loss after training = 8861.43015884421
700029
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3018.75 seconds
Loss after training = 9965.437571563914
700030
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 3028.77 seconds
Loss after training = 9353.398780936279
700031
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3016.71 seconds
Loss after training = 9171.506311979156
700032
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/ssss
Total time: 3087.65 seconds
Loss after training = 8888.35863877665
700033
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.19 iterations/s/ss
Total time: 3139.37 seconds
Loss after training = 8934.440282959049
700034
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.22 iterations/ss/s
Total time: 3109.05 seconds
Loss after training = 9016.33398372212
700035
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3025.22 seconds
Loss after training = 8683.213648129113
700036
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2998.22 seconds
Loss after training = 8851.201100847911
700037
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3006.92 seconds
Loss after training = 9233.510395759802
700038
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.11 iterations/ss/ss
Total time: 9039.46 seconds
Loss after training = 10105.629097471114
700039
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ss/s
Total time: 3059.9 seconds
Loss after training = 8693.057638832563
700040
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.22 iterations/s/ss
Total time: 3106.08 seconds
Loss after training = 8684.791403061585
700041
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3099.38 seconds
Loss after training = 8777.511773222925
700042
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3099.41 seconds
Loss after training = 9015.823874436499
700043
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3037.94 seconds
Loss after training = 8682.368603027591
700044
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/ssss
Total time: 3064.23 seconds
Loss after training = 8919.042093035463
700045
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3060.81 seconds
Loss after training = 8822.90070112399
700046
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3075.17 seconds
Loss after training = 8838.179420964867
700047
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3066.41 seconds
Loss after training = 9308.452480418859
700048
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3084.51 seconds
Loss after training = 10046.044127089539
700049
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.2 iterations/s/s/s
Total time: 3122.99 seconds
Loss after training = 8755.342285024026
700050
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3059.68 seconds
Loss after training = 9087.927904534898
700051
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3059.95 seconds
Loss after training = 9014.009160366151
700052
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3059.29 seconds
Loss after training = 8753.625377884498
700053
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.19 iterations/sss
Total time: 3130.82 seconds
Loss after training = 10266.89796056671
700054
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3050.6 seconds
Loss after training = 8854.686423532821
700055
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3041.58 seconds
Loss after training = 8918.36411904991
700056
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3079.14 seconds
Loss after training = 10057.993422992902
700057
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3095.17 seconds
Loss after training = 8811.28406904395
700058
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3086.4 seconds
Loss after training = 9010.863339607988
700059
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.27 iterations/ss/s
Total time: 7851.7 seconds
Loss after training = 8708.150589218592
700060
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.32 iterations/ss/ss
Total time: 7588.99 seconds
Loss after training = 8681.51710148151
700061
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.24 iterations/ss/ss
Total time: 8046.31 seconds
Loss after training = 8729.040995697864
700062
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.22 iterations/ss/ss
Total time: 8229.03 seconds
Loss after training = 9154.778817356755
700063
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.26 iterations/ss/ss
Total time: 7905.11 seconds
Loss after training = 9261.092147802914
700064
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.38 iterations/ss/s
Total time: 7261.39 seconds
Loss after training = 8738.317344954105
700065
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.41 iterations/ss/ss
Total time: 7101.32 seconds
Loss after training = 8728.878896568744
700066
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.39 iterations/s/sss
Total time: 7193.32 seconds
Loss after training = 10565.782504603605
700067
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.4 iterations/sns/s
Total time: 7150.92 seconds
Loss after training = 8744.031581125228
700068
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.21 iterations/ss/ss
Total time: 8249.15 seconds
Loss after training = 8723.997797872302
700069
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.19 iterations/ss/s
Total time: 8426.83 seconds
Loss after training = 9298.298526205836
700070
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.89 iterations/s/ss
Total time: 3466.09 seconds
Loss after training = 9168.73840682694
700071
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2988.44 seconds
Loss after training = 8986.308514973458
700072
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/sss
Total time: 2940.43 seconds
Loss after training = 8940.425406505587
700073
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.3 iterations/ss/ss
Total time: 4353.88 seconds
Loss after training = 8683.599567373349
700074
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.47 iterations/s/ss
Total time: 4050.52 seconds
Loss after training = 9016.312933820043
700075
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.13 iterations/s/ss
Total time: 4695.77 seconds
Loss after training = 8681.71094582679
700076
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.21 iterations/ss/ss
Total time: 8272.31 seconds
Loss after training = 9227.233411410289
700077
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.16 iterations/ss/ss
Total time: 8590.49 seconds
Loss after training = 8734.098084461188
700078
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.18 iterations/ss/ss
Total time: 8460.29 seconds
Loss after training = 8696.85451134236
700079
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.18 iterations/ss/ss
Total time: 8505.05 seconds
Loss after training = 8732.693750634435
700080
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.52 iterations/s/ss
Total time: 3962.13 seconds
Loss after training = 8955.349153643232
700081
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/ss
Total time: 3032.14 seconds
Loss after training = 8719.286505247896
700082
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3098.55 seconds
Loss after training = 8683.294745270166
700083
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3022.57 seconds
Loss after training = 9448.558470623506
700084
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3096.83 seconds
Loss after training = 8840.257480968137
700085
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3023.53 seconds
Loss after training = 8743.879507836355
700086
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3096.81 seconds
Loss after training = 8687.867587161827
700087
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3020.38 seconds
Loss after training = 9277.022489040599
700088
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3080.72 seconds
Loss after training = 9270.804052436502
700089
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.55 iterations/s/ss
Total time: 3916.83 seconds
Loss after training = 9059.152643700072
700090
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.16 iterations/ss/ss
Total time: 8608.72 seconds
Loss after training = 8680.379089410973
700091
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.26 iterations/ss/ss
Total time: 7932.77 seconds
Loss after training = 9124.9505969953
700092
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.15 iterations/ss/ss
Total time: 8685.34 seconds
Loss after training = 8697.792057492163
700093
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.24 iterations/ss/ss
Total time: 8047.7 seconds
Loss after training = 9171.647602160194
700094
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.22 iterations/ss/ss
Total time: 8206.39 seconds
Loss after training = 8695.829416623586
700095
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.27 iterations/ss/ss
Total time: 7881.13 seconds
Loss after training = 8683.160593499833
700096
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.06 iterations/s/sss
Total time: 4845.07 seconds
Loss after training = 9079.074396556414
700097
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3023.5 seconds
Loss after training = 11726.045805661783
700098
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3048.66 seconds
Loss after training = 9514.142686948233
700099
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3077.14 seconds
Loss after training = 8688.249346838911
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
PlumNormal = """700000
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2969.84 seconds
Loss after training = 8797.495550612955
700001
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2959.81 seconds
Loss after training = 8714.988732628046
700002
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3022.89 seconds
Loss after training = 8705.71627624257
700003
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2978.23 seconds
Loss after training = 9258.207441627634
700004
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3093.71 seconds
Loss after training = 9014.26917401424
700005
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.05 iterations/ss/s
Total time: 3279.65 seconds
Loss after training = 8886.063423108353
700006
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3077.25 seconds
Loss after training = 8797.32762987063
700007
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3025.66 seconds
Loss after training = 8783.939278465537
700008
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2964.03 seconds
Loss after training = 8793.13080369222
700009
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3022.14 seconds
Loss after training = 9321.139849566609
700010
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3036.51 seconds
Loss after training = 8710.300021615141
700011
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 2999.2 seconds
Loss after training = 10914.945417078683
700012
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2962.43 seconds
Loss after training = 9029.954775963879
700013
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2974.94 seconds
Loss after training = 9948.107091730431
700014
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2979.51 seconds
Loss after training = 9054.681240644044
700015
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2973.93 seconds
Loss after training = 8923.769204279593
700016
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 3028.4 seconds
Loss after training = 8687.81555892381
700017
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2962.09 seconds
Loss after training = 8833.334941804222
700018
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3009.71 seconds
Loss after training = 9005.673284097113
700019
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2972.52 seconds
Loss after training = 8695.705102891587
700020
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2974.27 seconds
Loss after training = 9264.3880246898
700021
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2955.17 seconds
Loss after training = 10467.815662775434
700022
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2983.48 seconds
Loss after training = 8685.888133197639
700023
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2960.04 seconds
Loss after training = 8718.124440431715
700024
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2954.95 seconds
Loss after training = 8939.670639466422
700025
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.41 iterations/s/ss
Total time: 2929.58 seconds
Loss after training = 9242.026305898788
700026
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3015.08 seconds
Loss after training = 8991.726105040792
700027
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/sss
Total time: 2942.95 seconds
Loss after training = 8835.547735364062
700028
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 2947.23 seconds
Loss after training = 8843.79934372514
700029
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 2946.79 seconds
Loss after training = 10037.862142059088
700030
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2980.83 seconds
Loss after training = 9432.148928269546
700031
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2984.82 seconds
Loss after training = 9192.644140151544
700032
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3023.18 seconds
Loss after training = 8769.654015425807
700033
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.2 iterations/ss/ss
Total time: 3123.29 seconds
Loss after training = 8972.026011047732
700034
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/s/ss
Total time: 3054.65 seconds
Loss after training = 9000.894831795173
700035
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2983.67 seconds
Loss after training = 8705.499636201295
700036
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3002.33 seconds
Loss after training = 8844.887033715471
700037
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3007.76 seconds
Loss after training = 9256.833258740473
700038
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.62 iterations/s/ss
Total time: 6185.92 seconds
Loss after training = 11048.029784096638
700039
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.69 iterations/ss/ss
Total time: 5900.67 seconds
Loss after training = 8690.520672544726
700040
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3024.23 seconds
Loss after training = 8703.477869027194
700041
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3062.33 seconds
Loss after training = 8898.464545687093
700042
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3058.74 seconds
Loss after training = 9077.03075503885
700043
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3043.44 seconds
Loss after training = 8711.758959695611
700044
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3044.89 seconds
Loss after training = 8919.892426817016
700045
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3046.71 seconds
Loss after training = 8818.036312526257
700046
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3059.83 seconds
Loss after training = 9003.260285044073
700047
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3010.58 seconds
Loss after training = 9691.765677875544
700048
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3060.98 seconds
Loss after training = 10122.734268722508
700049
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3074.38 seconds
Loss after training = 8738.44511973865
700050
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3045.49 seconds
Loss after training = 9079.701061632939
700051
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3062.55 seconds
Loss after training = 9157.022074057317
700052
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3009.26 seconds
Loss after training = 8787.22609120178
700053
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.21 iterations/ss/s
Total time: 3113.88 seconds
Loss after training = 10186.672484290679
700054
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/s
Total time: 3037.57 seconds
Loss after training = 8902.874229174839
700055
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ss/s
Total time: 3055.83 seconds
Loss after training = 8892.343040197038
700056
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3013.39 seconds
Loss after training = 10736.956171869948
700057
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 3032.17 seconds
Loss after training = 8756.402853178692
700058
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/ss/ss
Total time: 3027.5 seconds
Loss after training = 8965.889491296919
700059
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3049.59 seconds
Loss after training = 8708.33136527691
700060
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.0 iterations/ss/sss
Total time: 10012.88 seconds
Loss after training = 8701.392302758662
700061
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.45 iterations/ss/ss
Total time: 6907.07 seconds
Loss after training = 8729.423478333578
700062
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.27 iterations/ss/ss
Total time: 7869.46 seconds
Loss after training = 9133.469889631018
700063
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.58 iterations/s/ss
Total time: 6332.58 seconds
Loss after training = 9285.952824106866
700064
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.26 iterations/ss/s
Total time: 7923.39 seconds
Loss after training = 8770.82280230296
700065
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.3 iterations/sns/ss
Total time: 7669.75 seconds
Loss after training = 8727.463270613975
700066
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.23 iterations/ss/s
Total time: 8159.78 seconds
Loss after training = 10546.95376241314
700067
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.13 iterations/ss/ss
Total time: 8867.43 seconds
Loss after training = 8731.511296219747
700068
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.14 iterations/ss/ss
Total time: 8756.45 seconds
Loss after training = 8723.990900870573
700069
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.15 iterations/ss/ss
Total time: 8696.32 seconds
Loss after training = 9372.024108259966
700070
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.87 iterations/s/sss
Total time: 5358.58 seconds
Loss after training = 9090.306647942223
700071
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2963.24 seconds
Loss after training = 9017.743545234012
700072
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/sss
Total time: 2938.31 seconds
Loss after training = 8921.58936704639
700073
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.41 iterations/s/ss
Total time: 4151.77 seconds
Loss after training = 8694.630320438666
700074
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.54 iterations/s/ss
Total time: 3942.41 seconds
Loss after training = 9156.582382833827
700075
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.15 iterations/s/ss
Total time: 3176.33 seconds
Loss after training = 8683.28741230256
700076
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.28 iterations/s/ss
Total time: 7801.1 seconds
Loss after training = 9167.465816045153
700077
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.18 iterations/ss/ss
Total time: 8464.39 seconds
Loss after training = 8754.9583365151
700078
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.22 iterations/ss/ss
Total time: 8229.75 seconds
Loss after training = 8691.847055397548
700079
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.23 iterations/ss/ss
Total time: 8158.99 seconds
Loss after training = 8747.71889135146
700080
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.58 iterations/s/sss
Total time: 6335.46 seconds
Loss after training = 9003.348423717687
700081
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2991.63 seconds
Loss after training = 8715.078574024872
700082
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3009.78 seconds
Loss after training = 8702.056750914691
700083
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3038.82 seconds
Loss after training = 9765.673151760779
700084
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/ss
Total time: 3034.67 seconds
Loss after training = 8881.27345597479
700085
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3051.65 seconds
Loss after training = 8725.017868812447
700086
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3006.68 seconds
Loss after training = 8695.364045237757
700087
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3099.74 seconds
Loss after training = 9351.95039404734
700088
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3070.93 seconds
Loss after training = 9273.034784945623
700089
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3023.27 seconds
Loss after training = 9079.05099838426
700090
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.73 iterations/s/ss
Total time: 5777.61 seconds
Loss after training = 8693.683245525173
700091
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.2 iterations/sns/ss
Total time: 8342.47 seconds
Loss after training = 9183.690444999214
700092
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.16 iterations/ss/ss
Total time: 8605.3 seconds
Loss after training = 8697.833332993849
700093
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.21 iterations/ss/ss
Total time: 8263.08 seconds
Loss after training = 9238.515548423162
700094
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.2 iterations/sns/ss
Total time: 8354.95 seconds
Loss after training = 8695.759254414414
700095
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.3 iterations/sns/ss
Total time: 7721.36 seconds
Loss after training = 8683.156040153191
700096
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.29 iterations/ss/ss
Total time: 7727.79 seconds
Loss after training = 9052.277377086979
700097
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.2 iterations/ss/ss
Total time: 3129.81 seconds
Loss after training = 11706.83706921372
700098
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3087.43 seconds
Loss after training = 9681.800527488558
700099
True loss: 8682.601097676705
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3038.67 seconds
Loss after training = 8698.466343718745
"""

#BW:
BWNormal = """700000
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2965.22 seconds
Loss after training = 8768.68783256067
700001
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 2949.12 seconds
Loss after training = 8715.745349667084
700002
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 3026.92 seconds
Loss after training = 8696.489799384917
700003
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/s/ss
Total time: 2990.84 seconds
Loss after training = 9237.071861907958
700004
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.22 iterations/ss/s
Total time: 3102.72 seconds
Loss after training = 9031.160071054634
700005
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.07 iterations/s/ss
Total time: 3256.34 seconds
Loss after training = 8897.290688624516
700006
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/ss/s
Total time: 3051.94 seconds
Loss after training = 8768.087393934231
700007
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 3028.9 seconds
Loss after training = 8788.703236793748
700008
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2973.55 seconds
Loss after training = 8766.853251847942
700009
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 3029.48 seconds
Loss after training = 9278.55187532636
700010
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3022.94 seconds
Loss after training = 8691.60234715991
700011
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3002.08 seconds
Loss after training = 10854.225064911829
700012
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3018.51 seconds
Loss after training = 9043.599944545644
700013
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3000.24 seconds
Loss after training = 9908.905742292578
700014
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3043.88 seconds
Loss after training = 9077.181197682747
700015
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3015.6 seconds
Loss after training = 8919.079307111815
700016
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2983.16 seconds
Loss after training = 8681.449572895926
700017
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3004.71 seconds
Loss after training = 8843.100744165791
700018
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3017.52 seconds
Loss after training = 8999.125525011312
700019
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3036.69 seconds
Loss after training = 8703.248421694025
700020
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2995.32 seconds
Loss after training = 9243.227023055366
700021
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2997.03 seconds
Loss after training = 10531.785041267169
700022
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/s/s
Total time: 3027.04 seconds
Loss after training = 8679.454826666657
700023
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3020.44 seconds
Loss after training = 8719.28390785444
700024
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2983.28 seconds
Loss after training = 8898.428484762182
700025
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2991.2 seconds
Loss after training = 9295.834932207152
700026
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2957.12 seconds
Loss after training = 8951.127348324542
700027
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2970.32 seconds
Loss after training = 8868.515804470298
700028
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/ss
Total time: 3032.46 seconds
Loss after training = 8856.539592473719
700029
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2961.25 seconds
Loss after training = 10003.274215542102
700030
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3049.86 seconds
Loss after training = 9369.588955768468
700031
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2981.42 seconds
Loss after training = 9233.972309977526
700032
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2965.76 seconds
Loss after training = 8775.359610581034
700033
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.21 iterations/ss/s
Total time: 3113.46 seconds
Loss after training = 8933.647353317041
700034
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.18 iterations/s/ss
Total time: 3144.81 seconds
Loss after training = 9021.488763283218
700035
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3023.24 seconds
Loss after training = 8688.33039240627
700036
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2997.67 seconds
Loss after training = 8845.936076389882
700037
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3007.8 seconds
Loss after training = 9203.8821555234
700038
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.27 iterations/ss/s
Total time: 7859.09 seconds
Loss after training = 11419.655691501459
700039
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.42 iterations/ss/ss
Total time: 4140.69 seconds
Loss after training = 8694.504093878228
700040
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3043.49 seconds
Loss after training = 8687.583067772039
700041
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3046.67 seconds
Loss after training = 8870.339075533695
700042
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3083.9 seconds
Loss after training = 9028.708043469234
700043
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3067.38 seconds
Loss after training = 8701.335555657522
700044
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.2 iterations/s/sss
Total time: 3121.79 seconds
Loss after training = 8905.734068827494
700045
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3067.21 seconds
Loss after training = 8827.151560757833
700046
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3048.67 seconds
Loss after training = 8981.826642693486
700047
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/ssss
Total time: 3063.05 seconds
Loss after training = 9668.995029734277
700048
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3081.53 seconds
Loss after training = 10133.282636156628
700049
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3063.0 seconds
Loss after training = 8762.974470171252
700050
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3002.88 seconds
Loss after training = 9090.35494378777
700051
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3068.03 seconds
Loss after training = 9162.41976303323
700052
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.21 iterations/ss/s
Total time: 3112.32 seconds
Loss after training = 8759.24126328531
700053
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3054.52 seconds
Loss after training = 10251.605846663564
700054
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3079.86 seconds
Loss after training = 8864.556576267654
700055
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.2 iterations/s/sss
Total time: 3121.84 seconds
Loss after training = 8927.005139641307
700056
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3051.38 seconds
Loss after training = 10748.794323360007
700057
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3057.73 seconds
Loss after training = 8798.14246583257
700058
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3040.25 seconds
Loss after training = 8954.995113422377
700059
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.0 iterations/s/s/s
Total time: 5010.57 seconds
Loss after training = 8707.423536658272
700060
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.11 iterations/ss/ss
Total time: 9008.25 seconds
Loss after training = 8684.56317178103
700061
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.35 iterations/ss/s
Total time: 7430.51 seconds
Loss after training = 8730.630683717052
700062
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.33 iterations/ss/ss
Total time: 7518.37 seconds
Loss after training = 9154.602995369736
700063
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.25 iterations/ss/s
Total time: 7970.9 seconds
Loss after training = 9245.76469620583
700064
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.5 iterations/sns/ss
Total time: 6684.18 seconds
Loss after training = 8744.197150835398
700065
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.26 iterations/ss/s
Total time: 7946.44 seconds
Loss after training = 8725.519419139786
700066
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.33 iterations/ss/ss
Total time: 7503.21 seconds
Loss after training = 10614.64263763452
700067
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.21 iterations/ss/s
Total time: 8291.86 seconds
Loss after training = 8751.186540989576
700068
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.26 iterations/ss/ss
Total time: 7918.01 seconds
Loss after training = 8724.581428436313
700069
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.34 iterations/ss/ss
Total time: 7482.64 seconds
Loss after training = 9311.091928572028
700070
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.88 iterations/s/sss
Total time: 5315.39 seconds
Loss after training = 9151.876527875418
700071
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 3030.81 seconds
Loss after training = 8976.824621374655
700072
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3019.45 seconds
Loss after training = 8928.305760288382
700073
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.14 iterations/ssss
Total time: 3185.02 seconds
Loss after training = 8682.57901320775
700074
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.14 iterations/s/ss
Total time: 4678.27 seconds
Loss after training = 9135.15269196219
700075
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.87 iterations/s/ss
Total time: 3486.39 seconds
Loss after training = 8682.095412648472
700076
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.32 iterations/ss/s
Total time: 7549.59 seconds
Loss after training = 9211.868124819674
700077
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.23 iterations/ss/ss
Total time: 8140.15 seconds
Loss after training = 8738.946259541472
700078
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.26 iterations/ss/ss
Total time: 7962.34 seconds
Loss after training = 8694.461102822384
700079
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.33 iterations/ss/ss
Total time: 7522.53 seconds
Loss after training = 8728.768143354002
700080
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.28 iterations/ss/ss
Total time: 7799.56 seconds
Loss after training = 8981.917341789163
700081
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3071.64 seconds
Loss after training = 8731.739911828101
700082
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/ssss
Total time: 3036.58 seconds
Loss after training = 8685.884625258534
700083
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3059.31 seconds
Loss after training = 9725.200897640298
700084
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.2 iterations/s/sss
Total time: 3129.03 seconds
Loss after training = 8849.69624133773
700085
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3041.28 seconds
Loss after training = 8745.783584826182
700086
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3077.62 seconds
Loss after training = 8689.359638668762
700087
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3083.16 seconds
Loss after training = 9293.399939692816
700088
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3077.38 seconds
Loss after training = 9234.074588939186
700089
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.21 iterations/ss/s
Total time: 3117.06 seconds
Loss after training = 9046.951435976716
700090
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.53 iterations/s/ss
Total time: 6555.5 seconds
Loss after training = 8681.567771459828
700091
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.22 iterations/ss/ss
Total time: 8173.86 seconds
Loss after training = 9132.178913033553
700092
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.14 iterations/ss/ss
Total time: 8787.62 seconds
Loss after training = 8697.836975337561
700093
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.17 iterations/ss/ss
Total time: 8531.3 seconds
Loss after training = 9182.995471396262
700094
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.15 iterations/ss/ss
Total time: 8704.5 seconds
Loss after training = 8695.774573653294
700095
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.25 iterations/ss/ss
Total time: 8023.84 seconds
Loss after training = 8683.131536349656
700096
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.48 iterations/s/sss
Total time: 6762.43 seconds
Loss after training = 9077.709418618168
700097
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.17 iterations/s/ss
Total time: 3154.46 seconds
Loss after training = 11797.630255944361
700098
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3084.81 seconds
Loss after training = 9631.839678107708
700099
True loss: 8682.60109769344
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.22 iterations/s/ss
Total time: 3103.15 seconds
Loss after training = 8685.478791934227
"""

#Zhao:
ZhaoNormal = """700000
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/sss
Total time: 2940.29 seconds
Loss after training = 8806.141412888912
700001
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.42 iterations/s/ss
Total time: 2923.18 seconds
Loss after training = 8711.486938261398
700002
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3005.52 seconds
Loss after training = 8717.64288930706
700003
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3002.28 seconds
Loss after training = 9309.193366862493
700004
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3072.54 seconds
Loss after training = 9002.597150390957
700005
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.08 iterations/s/ss
Total time: 3243.73 seconds
Loss after training = 8876.757013965547
700006
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3008.91 seconds
Loss after training = 8806.628870173501
700007
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 2950.57 seconds
Loss after training = 8777.800126392363
700008
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2956.41 seconds
Loss after training = 8802.620267098437
700009
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3008.49 seconds
Loss after training = 9385.298770046755
700010
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2956.45 seconds
Loss after training = 8713.950603291883
700011
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2968.27 seconds
Loss after training = 10930.457288385987
700012
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.44 iterations/s/ss
Total time: 2907.9 seconds
Loss after training = 9045.796308234481
700013
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.33 iterations/s/ss
Total time: 3000.06 seconds
Loss after training = 9933.808079250286
700014
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2973.33 seconds
Loss after training = 9041.371535150927
700015
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.41 iterations/s/ss
Total time: 2935.88 seconds
Loss after training = 8923.14564584369
700016
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2963.42 seconds
Loss after training = 8693.62509069302
700017
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2955.72 seconds
Loss after training = 8825.517827282463
700018
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2971.01 seconds
Loss after training = 9014.09775625321
700019
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.36 iterations/s/ss
Total time: 2973.63 seconds
Loss after training = 8692.445401160772
700020
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.44 iterations/s/ss
Total time: 2906.29 seconds
Loss after training = 9262.470353717124
700021
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.44 iterations/s/ss
Total time: 2905.38 seconds
Loss after training = 10436.41481258826
700022
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2997.33 seconds
Loss after training = 8686.94389819129
700023
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/ss/ss
Total time: 2936.96 seconds
Loss after training = 8714.41556764041
700024
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.34 iterations/ssss
Total time: 2991.51 seconds
Loss after training = 8952.557887042933
700025
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/sss
Total time: 2942.59 seconds
Loss after training = 9272.37090968722
700026
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2967.67 seconds
Loss after training = 9003.642890894025
700027
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.43 iterations/s/ss
Total time: 2913.04 seconds
Loss after training = 8815.216943736243
700028
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.39 iterations/s/ss
Total time: 2950.83 seconds
Loss after training = 8835.603038239233
700029
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2986.08 seconds
Loss after training = 10015.596465062521
700030
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.37 iterations/s/ss
Total time: 2967.64 seconds
Loss after training = 9453.915935740593
700031
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.38 iterations/s/ss
Total time: 2955.62 seconds
Loss after training = 9151.779731165272
700032
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.44 iterations/s/ss
Total time: 2909.36 seconds
Loss after training = 8764.00006230698
700033
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3066.64 seconds
Loss after training = 8984.954970029383
700034
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3069.23 seconds
Loss after training = 8988.810881256444
700035
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2983.67 seconds
Loss after training = 8708.688765224417
700036
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.4 iterations/s/s/s
Total time: 2938.17 seconds
Loss after training = 8838.39290696757
700037
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.35 iterations/s/ss
Total time: 2982.77 seconds
Loss after training = 9306.276390934121
700038
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.61 iterations/s/ss
Total time: 3838.52 seconds
Loss after training = 11221.022348645838
700039
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.22 iterations/ss/ss
Total time: 8188.48 seconds
Loss after training = 8688.078557002098
700040
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3069.84 seconds
Loss after training = 8707.390712499764
700041
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3088.96 seconds
Loss after training = 8909.072632319334
700042
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3058.32 seconds
Loss after training = 9093.174835624675
700043
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3025.27 seconds
Loss after training = 8710.84140380696
700044
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3073.04 seconds
Loss after training = 8956.120312442281
700045
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3021.42 seconds
Loss after training = 8810.132362149905
700046
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3048.43 seconds
Loss after training = 8996.201892500168
700047
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3072.28 seconds
Loss after training = 9680.176272721686
700048
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3058.94 seconds
Loss after training = 10127.357904351546
700049
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3057.18 seconds
Loss after training = 8728.487808414857
700050
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3036.69 seconds
Loss after training = 9074.588995796044
700051
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3039.58 seconds
Loss after training = 9165.021770754363
700052
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.31 iterations/s/ss
Total time: 3018.4 seconds
Loss after training = 8795.668349504907
700053
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.29 iterations/s/ss
Total time: 3041.82 seconds
Loss after training = 10155.622893740243
700054
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.2 iterations/s/sss
Total time: 3123.17 seconds
Loss after training = 8915.201216343283
700055
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.28 iterations/s/ss
Total time: 3051.36 seconds
Loss after training = 8868.438426636068
700056
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3087.14 seconds
Loss after training = 10734.840787789592
700057
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3077.27 seconds
Loss after training = 8743.825389521717
700058
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3076.02 seconds
Loss after training = 8960.80921886602
700059
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.21 iterations/ss/s
Total time: 3117.4 seconds
Loss after training = 8704.400777331757
700060
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.23 iterations/ss/s
Total time: 8098.03 seconds
Loss after training = 8704.545161058964
700061
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.51 iterations/ss/ss
Total time: 6603.6 seconds
Loss after training = 8725.375119129678
700062
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.24 iterations/ss/ss
Total time: 8037.16 seconds
Loss after training = 9118.636425596063
700063
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.14 iterations/ss/ss
Total time: 8784.02 seconds
Loss after training = 9344.74445192028
700064
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.16 iterations/ss/ss
Total time: 8606.49 seconds
Loss after training = 8778.727476411368
700065
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.17 iterations/ss/ss
Total time: 8565.9 seconds
Loss after training = 8724.340840193292
700066
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.21 iterations/ss/ss
Total time: 8257.15 seconds
Loss after training = 10515.173377911433
700067
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.36 iterations/ss/ss
Total time: 7327.94 seconds
Loss after training = 8721.980416522656
700068
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.29 iterations/ss/ss
Total time: 7766.13 seconds
Loss after training = 8720.171596506176
700069
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.53 iterations/ss/s
Total time: 6535.17 seconds
Loss after training = 9392.214501970733
700070
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.38 iterations/ss/ss
Total time: 7252.82 seconds
Loss after training = 9054.050784817682
700071
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3011.24 seconds
Loss after training = 9060.568467693383
700072
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/sss
Total time: 3029.65 seconds
Loss after training = 8919.03073564266
700073
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.3 iterations/s/s/s
Total time: 3026.55 seconds
Loss after training = 8697.41142266332
700074
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.14 iterations/s/ss
Total time: 4671.37 seconds
Loss after training = 9145.490929742436
700075
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.66 iterations/s/ss
Total time: 3754.0 seconds
Loss after training = 8685.949427645175
700076
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.77 iterations/s/ss
Total time: 5651.55 seconds
Loss after training = 9127.23082585424
700077
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.23 iterations/ss/ss
Total time: 8134.3 seconds
Loss after training = 8774.931750260306
700078
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.26 iterations/ss/ss
Total time: 7923.06 seconds
Loss after training = 8689.31794459291
700079
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.23 iterations/ss/ss
Total time: 8124.68 seconds
Loss after training = 8754.509225653746
700080
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.29 iterations/s/sss
Total time: 7751.99 seconds
Loss after training = 8995.27841325172
700081
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.39 iterations/s/sss
Total time: 4180.01 seconds
Loss after training = 8707.60435595642
700082
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3082.42 seconds
Loss after training = 8705.584592090563
700083
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.32 iterations/s/ss
Total time: 3013.61 seconds
Loss after training = 9778.04644469611
700084
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/s/ss
Total time: 3053.92 seconds
Loss after training = 8888.382779757165
700085
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3071.68 seconds
Loss after training = 8717.330499256856
700086
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.26 iterations/s/ss
Total time: 3065.99 seconds
Loss after training = 8706.203836906285
700087
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.27 iterations/ssss
Total time: 3060.44 seconds
Loss after training = 9372.85952205925
700088
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3079.4 seconds
Loss after training = 9323.796814781323
700089
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.23 iterations/s/ss
Total time: 3095.79 seconds
Loss after training = 9115.72670231466
700090
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 2.28 iterations/s/ss
Total time: 4388.58 seconds
Loss after training = 8696.24635294
700091
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.16 iterations/ss/ss
Total time: 8654.41 seconds
Loss after training = 9200.659453050468
700092
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.17 iterations/ss/ss
Total time: 8539.6 seconds
Loss after training = 8695.03471961574
700093
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.19 iterations/ss/ss
Total time: 8399.32 seconds
Loss after training = 9257.6809271328
700094
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.16 iterations/ss/ss
Total time: 8613.74 seconds
Loss after training = 8692.935629494237
700095
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.16 iterations/ss/ss
Total time: 8595.4 seconds
Loss after training = 8683.790615314078
700096
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,1 secs remaining, 1.19 iterations/ss/ss
Total time: 8386.55 seconds
Loss after training = 9039.388522768431
700097
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.01 iterations/ss/s
Total time: 3326.61 seconds
Loss after training = 11665.869345640174
700098
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.25 iterations/s/ss
Total time: 3076.29 seconds
Loss after training = 9704.886083264411
700099
True loss: 8682.601097667446
Iterating: |██████████████████████████████| 100.0% complete,0 secs remaining, 3.24 iterations/s/ss
Total time: 3082.16 seconds
Loss after training = 8701.211898331092
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


