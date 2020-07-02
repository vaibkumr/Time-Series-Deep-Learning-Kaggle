import os
import time
# f1 = "/home/timetraveller/Work/m5data/output/2020-04-27_03:53:02.774404/2020-04-27_03:53:02.774404.csv" #X_test shape: (853720, 88)
f2 = "/home/timetraveller/Work/m5data/output/2020-04-27_06:07:01.190064/2020-04-27_06:07:01.190064-submission.csv" #X_test shape: (853720, 88)
f3 = "/home/timetraveller/Work/m5data/output/2020-04-27_08:46:38.657025/2020-04-27_08:46:38.657025-submission.csv" #X_test shape: (853720, 66)

# m1 = "400-with-target-encoding"
m2 = "800-with-target-encoding"
m3 = "800-without-target-encoding"

for m, f in zip([m2, m3], [f2, f3]):
        os.system(f"kaggle competitions submit -f {f} -m {m} -c m5-forecasting-accuracy")        
        time.sleep(60*5)