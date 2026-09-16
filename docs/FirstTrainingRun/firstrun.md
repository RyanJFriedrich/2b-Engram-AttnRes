Below is the first test run. Un-Optimized.

The true throughput was 4x the listed Tokens/second because it wasn't properly using gradient accumulation when factoring in the data. 

[2026-09-03 12:09:23] [INFO] prebuilt base checkpoint loaded: exports/llama-9b-base-v1
[2026-09-03 12:10:39] [INFO] step 1/12735 loss 6.7477 lr 1.00e-05 gnorm 202.913 window 4096 theta 1.000 tok/s 428 eng_round 0.000
[2026-09-03 12:15:34] [INFO] step 5/12735 loss 6.6958 lr 5.00e-05 gnorm 16.194 window 4096 theta 1.000 tok/s 442 eng_round 0.007
[2026-09-03 12:21:37] [INFO] step 10/12735 loss 6.8891 lr 1.00e-04 gnorm 41.689 window 4096 theta 1.000 tok/s 446 eng_round 0.010
[2026-09-03 12:27:42] [INFO] step 15/12735 loss 13.2775 lr 1.50e-04 gnorm 275.066 window 4096 theta 1.000 tok/s 447 eng_round 0.009
[2026-09-03 12:33:45] [INFO] step 20/12735 loss 9.9112 lr 2.00e-04 gnorm 147.718 window 4096 theta 1.000 tok/s 448 eng_round 0.009
[2026-09-03 12:39:48] [INFO] step 25/12735 loss 8.3387 lr 2.00e-04 gnorm 44.448 window 4096 theta 1.000 tok/s 449 eng_round 0.012
[2026-09-03 12:45:50] [INFO] step 30/12735 loss 7.5902 lr 1.99e-04 gnorm 19.709 window 4096 theta 1.000 tok/s 449 eng_round 0.022
[2026-09-03 12:51:53] [INFO] step 35/12735 loss 7.4805 lr 1.97e-04 gnorm 9.811 window 4096 theta 1.000 tok/s 450 eng_round 0.030
[2026-09-03 12:57:55] [INFO] step 40/12735 loss 7.2846 lr 1.95e-04 gnorm 9.863 window 4096 theta 1.000 tok/s 450 eng_round 0.034
[2026-09-03 13:03:59] [INFO] step 45/12735 loss 7.2757 lr 1.92e-04 gnorm 17.186 window 4096 theta 1.000 tok/s 450 eng_round 0.033
[2026-09-03 13:10:02] [INFO] step 50/12735 loss 7.4540 lr 1.89e-04 gnorm 19.664 window 4096 theta 1.000 tok/s 450 eng_round 0.039
[2026-09-03 13:16:05] [INFO] step 55/12735 loss 7.0503 lr 1.85e-04 gnorm 10.292 window 4096 theta 1.000 tok/s 450 eng_round 0.046
[2026-09-03 13:22:07] [INFO] step 60/12735 loss 7.2147 lr 1.80e-04 gnorm 6.278 window 4096 theta 1.000 tok/s 450 eng_round 0.050
[2026-09-03 13:28:10] [INFO] step 65/12735 loss 7.2939 lr 1.75e-04 gnorm 12.946 window 4096 theta 1.000 tok/s 451 eng_round 0.053
[2026-09-03 13:34:13] [INFO] step 70/12735 loss 7.2487 lr 1.69e-04 gnorm 18.968 window 4096 theta 1.000 tok/s 451 eng_round 0.057
[2026-09-03 13:40:15] [INFO] step 75/12735 loss 7.3302 lr 1.63e-04 gnorm 25.357 window 4096 theta 1.000 tok/s 451 eng_round 0.058
[2026-09-03 13:46:17] [INFO] step 80/12735 loss 6.9805 lr 1.56e-04 gnorm 14.364 window 4096 theta 1.000 tok/s 451 eng_round 0.060
[2026-09-03 13:52:19] [INFO] step 85/12735 loss 7.0991 lr 1.49e-04 gnorm 7.627 window 4096 theta 1.000 tok/s 451 eng_round 0.058
[2026-09-03 13:58:21] [INFO] step 90/12735 loss 7.1179 lr 1.42e-04 gnorm 13.126 window 4096 theta 1.000 tok/s 451 eng_round 0.058
[2026-09-03 14:04:23] [INFO] step 95/12735 loss 6.8846 lr 1.35e-04 gnorm 6.368 window 4096 theta 1.000 tok/s 451 eng_round 0.057
[2026-09-03 14:10:25] [INFO] step 100/12735 loss 7.0554 lr 1.27e-04 gnorm 15.927 window 4096 theta 1.000 tok/s 451 eng_round 0.054
[2026-09-03 14:11:03] [INFO] checkpoint saved: train/runs/real_base_v1/ckpt_step100.pt (step 100)
